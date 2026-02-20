"""
DQI (Discourse Quality Index) Assessor using AQuA.

This module provides a wrapper around the AQuA repository to assess
deliberation quality of transcripts.
"""

import os
import sys
import tempfile
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
from datetime import datetime
import asyncio

try:
    import torch
    from transformers import AutoTokenizer, AutoConfig
    # Try new adapters package first (adapter-transformers 4.0.0+)
    try:
        from adapters import AutoAdapterModel
    except ImportError:
        # Fall back to old adapter-transformers API
        try:
            from transformers import AutoAdapterModel
        except ImportError:
            # Last resort: try adapter_transformers package
            from adapter_transformers import AutoAdapterModel
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    AQUA_AVAILABLE = True
    AQUA_IMPORT_ERROR = None
except ImportError as e:
    AQUA_AVAILABLE = False
    AQUA_IMPORT_ERROR = str(e)


class DQIAssessor:
    """
    Assessor for Discourse Quality Index using AQuA.
    
    This class wraps the AQuA functionality to assess deliberation quality
    of transcripts. It handles format conversion, translation (if needed),
    and provides a clean API for assessing transcripts.
    """
    
    def __init__(self, aqua_path: Optional[str] = None, batch_size: int = 8, max_translation_concurrent: int = 10):
        """
        Initialize DQI assessor.
        
        Args:
            aqua_path: Path to AQuA repository. If None, tries to find it
                      relative to this module's location.
            batch_size: Batch size for AQUA model inference (default: 8)
            max_translation_concurrent: Maximum concurrent translations (default: 10)
        """
        if not AQUA_AVAILABLE:
            error_msg = "AQuA dependencies not available. Install with: pip install adapter-transformers torch transformers"
            if AQUA_IMPORT_ERROR:
                error_msg += f"\nImport error: {AQUA_IMPORT_ERROR}"
            raise ImportError(error_msg)
        
        # Find AQuA path
        if aqua_path is None:
            # Try to find AQuA relative to this file
            current_file = Path(__file__).resolve()
            # Go up: analysis -> delib_sim -> delib_sim -> external -> AQuA
            # Also try: analysis -> delib_sim -> delib_sim -> .. -> delib_sim -> external -> AQuA
            possible_paths = [
                # From dqi_assessor.py: delib_sim/delib_sim/analysis/dqi_assessor.py
                # Go up 3 levels to delib_sim/, then external/AQuA
                current_file.parent.parent.parent / "external" / "AQuA",
                # From project root
                Path("delib_sim/external/AQuA"),
                # Absolute path from project root (if running from project root)
                Path(__file__).parent.parent.parent.parent / "delib_sim" / "external" / "AQuA",
                # Old nested path (for backward compatibility)
                current_file.parent.parent.parent.parent / "external" / "delib_sim" / "external" / "AQuA",
            ]
            
            for path in possible_paths:
                path = path.resolve()
                if path.exists() and (path / "inference_parallel_de.py").exists():
                    aqua_path = str(path)
                    break
            
            if aqua_path is None:
                # Try to find it anywhere in the project
                project_root = current_file.parent.parent.parent.parent
                for path in project_root.rglob("AQuA"):
                    if (path / "inference_parallel_de.py").exists():
                        aqua_path = str(path.resolve())
                        break
                
                if aqua_path is None:
                    raise FileNotFoundError(
                        "AQuA repository not found. Please specify aqua_path or "
                        "ensure AQuA is available. Tried:\n" +
                        "\n".join(f"  - {p}" for p in possible_paths)
                    )
        
        self.aqua_path = Path(aqua_path)
        if not self.aqua_path.exists():
            raise FileNotFoundError(f"AQuA path does not exist: {aqua_path}")
        
        # Patch transformers.adapters BEFORE adding AQuA to path
        # AQuA's code uses transformers.adapters, but new API uses adapters
        try:
            import adapters
            import adapters.composition
            # Create a fake transformers.adapters module for compatibility
            import types
            fake_adapters_module = types.ModuleType('transformers.adapters')
            fake_adapters_module.composition = adapters.composition
            fake_adapters_module.Parallel = adapters.composition.Parallel
            
            # Ensure transformers module exists
            if 'transformers' not in sys.modules:
                import transformers
                sys.modules['transformers'] = transformers
            
            # Patch transformers.adapters
            sys.modules['transformers.adapters'] = fake_adapters_module
            sys.modules['transformers.adapters.composition'] = adapters.composition
        except Exception as e:
            print(f"[WARN] Could not patch transformers.adapters: {e}")
            import traceback
            traceback.print_exc()
        
        # Add AQuA to path for imports
        if str(self.aqua_path) not in sys.path:
            sys.path.insert(0, str(self.aqua_path))
        
        # Import AQuA modules
        try:
            from data import InferenceDataset
            from utils import get_dynamic_parallel
            self.InferenceDataset = InferenceDataset
            self.get_dynamic_parallel = get_dynamic_parallel
        except ImportError as e:
            raise ImportError(f"Failed to import AQuA modules: {e}")
        
        # Initialize model and tokenizer (lazy loading)
        self._model = None
        self._tokenizer = None
        self._device = None
        self._task2identifier = None
        self._task2weight = None
        self._weights = None
        self._normalization_params = None
        
        # Translation settings
        self.translation_model = "moonshotai/kimi-k2"
        self.translation_api_key_env = "OPENROUTER_API_KEY"
        self.translation_api_endpoint = "https://openrouter.ai/api/v1"
        self.max_translation_concurrent = max_translation_concurrent
        
        # AQUA inference settings
        self.batch_size = batch_size
        
        # Load AQuA configuration
        self._load_aqua_config()
    
    def _load_aqua_config(self):
        """Load AQuA configuration (task identifiers, weights, etc.)."""
        # These are defined in inference_parallel_de.py
        self._task2identifier = {
            "relevance": "trained adapters/relevance",
            "fact": "trained adapters/fact",
            "opinion": "trained adapters/opinion",
            "justification": "trained adapters/justification",
            "solproposal": "trained adapters/solproposal",
            "addknowledge": "trained adapters/addknowledge",
            "question": "trained adapters/question",
            "refusers": "trained adapters/refusers",
            "refmedium": "trained adapters/refmedium",
            "refcontents": "trained adapters/refcontents",
            "refpersonal": "trained adapters/refpersonal",
            "refformat": "trained adapters/refformat",
            "address": "trained adapters/address",
            "respect": "trained adapters/respect",
            "screaming": "trained adapters/screaming",
            "vulgar": "trained adapters/vulgar",
            "insult": "trained adapters/insult",
            "sarcasm": "trained adapters/sarcasm",
            "discrimination": "trained adapters/discrimination",
            "storytelling": "trained adapters/storytelling"
        }
        
        self._task2weight = {
            "relevance": 0.20908452,
            "fact": 0.18285757,
            "opinion": -0.11069402,
            "justification": 0.29000763,
            "solproposal": 0.39535126,
            "addknowledge": 0.14655912,
            "question": -0.07331445,
            "refusers": -0.03768367,
            "refmedium": 0.07019062,
            "refcontents": -0.02847408,
            "refpersonal": 0.21126469,
            "refformat": -0.02674237,
            "address": 0.01482095,
            "respect": 0.00732909,
            "screaming": -0.01900971,
            "vulgar": -0.04995486,
            "insult": -0.05884586,
            "sarcasm": -0.15170863,
            "discrimination": 0.02934227,
            "storytelling": 0.10628146
        }
        
        self._weights = list(self._task2weight.values())
        
        # Normalization parameters
        maxval = 4.989267539999999
        minval = -1.66928295
        self._normalization_params = {
            "minval": minval,
            "maxval": maxval,
            "minmaxdif": maxval - minval,
            "bound": 5
        }
    
    def _initialize_model(self):
        """Initialize the AQuA model and tokenizer (lazy loading)."""
        if self._model is not None:
            return
        
        print("[DQI] Initializing AQuA model (this may take a moment)...")
        
        # Check for GPU
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            print("[DQI] Using GPU")
        else:
            self._device = torch.device("cpu")
            print("[DQI] Using CPU")
        
        # Suppress the warning about uninitialized weights (this is normal for adapter models)
        # The bias term gets randomly initialized and then adapter weights are loaded
        # This warning is harmless - the adapter heads are trained separately
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")
            warnings.filterwarnings("ignore", message=".*You should probably TRAIN.*")
            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            self._model = AutoAdapterModel.from_pretrained("bert-base-multilingual-cased")
        
        # Load adapters
        adapter_counter = 0
        adapter_paths = []
        for task, adapter_path in self._task2identifier.items():
            full_path = self.aqua_path / adapter_path
            if not full_path.exists():
                raise FileNotFoundError(
                    f"Adapter not found: {full_path}. "
                    "Ensure AQuA repository is complete with trained adapters."
                )
            adapter_paths.append(str(full_path))
            adapter_counter += 1
        
        # Load all adapters
        for i, (task, adapter_path) in enumerate(self._task2identifier.items()):
            full_path = self.aqua_path / adapter_path
            print(f"[DQI] Loading adapter {i+1}/{len(self._task2identifier)}: {task}")
            self._model.load_adapter(
                str(full_path),
                load_as=f"adapter{i}",
                with_head=True,
                set_active=True,
                source="hf"
            )
        
        # Set up parallel adapters
        adapter_setup = self.get_dynamic_parallel(adapter_number=len(self._task2identifier))
        self._model.active_adapters = adapter_setup
        
        # Move to device
        if self._device.type == "cuda":
            self._model = self._model.cuda()
        
        print(f"[DQI] Model initialized with {len(self._task2identifier)} adapters")
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, bound] interval."""
        params = self._normalization_params
        return ((scores - params["minval"]) / params["minmaxdif"]) * params["bound"]
    
    async def _translate_to_german(self, text: str, comment_index: Optional[int] = None) -> str:
        """
        Translate text from English to German using OpenRouter API with moonshotai/kimi-k2.
        
        Args:
            text: English text to translate
            comment_index: Optional index of comment being translated (for logging)
        
        Returns:
            German translation of the text
        """
        try:
            from openai import AsyncOpenAI
            
            api_key = os.getenv(self.translation_api_key_env)
            if not api_key:
                raise ValueError(f"API key not found for environment variable {self.translation_api_key_env}")
            
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.translation_api_endpoint
            )
            
            # Prepare translation prompt
            messages = [
                {
                    "role": "system",
                    "content": "Translate from English to German. Keep wording as literal as possible. Do not paraphrase. Preserve line breaks, numbering, punctuation, and formatting exactly. Keep proper nouns unchanged. If a phrase is ambiguous, choose the most literal rendering and add no commentary."
                    # "content": "You are a professional translator. Translate the English text to German. CRITICAL RULES: 0) Preserve the structure, tone, formatting, and meaning. 1) Translate EVERY sentence completely - do NOT summarize or shorten. 2) The German output should be similar length to the English input. 3) Output ONLY the German translation, nothing else."
                },
                {
                    "role": "user",
                    "content": f"{text}"
                }
            ]
            
            if comment_index is None:
                print(f"[DQI] Translating text to German (length: {len(text)} chars) using {self.translation_model}...")
            
            response = await client.chat.completions.create(
                model=self.translation_model,
                messages=messages,
                temperature=0.1,  # Lower temperature for more consistent translations
                max_tokens=16384  # Generous limit to avoid truncation
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            if comment_index is None:
                print(f"[DQI] Translation complete (length: {len(translated_text)} chars)")
            
            return translated_text
            
        except Exception as e:
            if comment_index is not None:
                print(f"[DQI] Translation failed for comment {comment_index}: {e}")
            else:
                print(f"[DQI] Translation failed: {e}")
            raise
    
    def _translate_to_german_sync(self, text: str) -> str:
        """
        Synchronous wrapper for translation (for use in non-async contexts).
        
        Args:
            text: English text to translate
        
        Returns:
            German translation of the text
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to use a different approach
                # Create a new event loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_translation_in_new_loop, text)
                    return future.result()
            else:
                return loop.run_until_complete(self._translate_to_german(text))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._translate_to_german(text))
    
    def _run_translation_in_new_loop(self, text: str) -> str:
        """Helper to run translation in a new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._translate_to_german(text))
        finally:
            loop.close()
    
    def _run_parallel_translation(self, comments: List[Dict[str, Any]], max_concurrent: int = 10):
        """
        Run parallel translation of comments (synchronous wrapper).
        
        Args:
            comments: List of comment dictionaries
            max_concurrent: Maximum number of concurrent translations
        
        Returns:
            Tuple of (translated_comments, truncation_warnings)
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, create a new event loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_parallel_translation_in_new_loop, comments, max_concurrent)
                    return future.result()
            else:
                return loop.run_until_complete(self._translate_comments_parallel(comments, max_concurrent))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._translate_comments_parallel(comments, max_concurrent))
    
    def _run_parallel_translation_in_new_loop(self, comments: List[Dict[str, Any]], max_concurrent: int):
        """Helper to run parallel translation in a new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._translate_comments_parallel(comments, max_concurrent))
        finally:
            loop.close()
    
    async def _translate_comments_parallel(self, comments: List[Dict[str, Any]], max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """
        Translate multiple comments in parallel.
        
        Args:
            comments: List of comment dictionaries with 'turn', 'instance_id', and 'comment'
            max_concurrent: Maximum number of concurrent translations
        
        Returns:
            List of translated comment dictionaries
        """
        from openai import AsyncOpenAI
        
        api_key = os.getenv(self.translation_api_key_env)
        if not api_key:
            raise ValueError(f"API key not found for environment variable {self.translation_api_key_env}")
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.translation_api_endpoint
        )
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Track truncation warnings
        truncation_warnings = []
        
        async def translate_single_comment(comment: Dict[str, Any], index: int) -> Dict[str, Any]:
            """Translate a single comment with rate limiting."""
            # Note: <thinking> blocks are already stripped in _transcript_to_comments
            comment_text = comment['comment']
            
            # Skip empty comments
            if not comment_text.strip():
                return comment
            
            # Warn if comment is very long
            if len(comment_text) > 10000:
                print(f"[DQI] Warning: Comment {index} is very long ({len(comment_text)} chars), may be truncated")
            
            async with semaphore:
                try:
                    # Prepare translation prompt
                    # Note: <thinking> blocks are already stripped in _transcript_to_comments
                    messages = [
                        {
                            "role": "system",
                            "content": "Translate from English to German. Keep wording as literal as possible. Do not paraphrase. Preserve line breaks, numbering, punctuation, and formatting exactly. Keep proper nouns unchanged. If a phrase is ambiguous, choose the most literal rendering and add no commentary."
                            # "content": "You are a professional translator. Translate the English text to German. CRITICAL RULES: 0) Preserve the structure, tone, formatting, and meaning. 1) Translate EVERY sentence completely - do NOT summarize or shorten. 2) The German output should be similar length to the English input. 3) Output ONLY the German translation, nothing else."
                        },
                        {
                            "role": "user",
                            "content": comment_text
                        }
                    ]
                    
                    response = await client.chat.completions.create(
                        model=self.translation_model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=16384  # Generous limit to avoid truncation
                    )
                    
                    translated_text = response.choices[0].message.content.strip()
                    
                    # Check if translation seems truncated
                    if len(translated_text) < len(comment_text) * 0.5 and len(comment_text) > 1000:
                        print(f"\n[DQI] ⚠️  TRUNCATION WARNING - Comment {index}")
                        print(f"[DQI] Original length: {len(comment_text)} chars -> Translated: {len(translated_text)} chars ({len(translated_text)/len(comment_text)*100:.1f}%)")
                        print(f"[DQI] === ENGLISH ORIGINAL ===")
                        print(comment_text[:2000] + ("..." if len(comment_text) > 2000 else ""))
                        print(f"[DQI] === GERMAN TRANSLATION ===")
                        print(translated_text[:2000] + ("..." if len(translated_text) > 2000 else ""))
                        print(f"[DQI] === END TRUNCATION WARNING ===\n")
                        
                        # Track for logging
                        truncation_warnings.append({
                            'comment_index': index,
                            'turn': comment['turn'],
                            'instance_id': comment['instance_id'],
                            'original_length': len(comment_text),
                            'translated_length': len(translated_text),
                            'ratio': len(translated_text) / len(comment_text),
                            'original_text': comment_text,
                            'translated_text': translated_text
                        })
                    
                    return {
                        'turn': comment['turn'],
                        'instance_id': comment['instance_id'],
                        'comment': translated_text,
                        'original_comment': comment_text  # Keep original for reference
                    }
                except Exception as e:
                    print(f"[DQI] Warning: Failed to translate comment {index}: {e}")
                    # Use original text if translation fails
                    return comment
        
        # Create tasks for all comments
        tasks = [translate_single_comment(comment, i+1) for i, comment in enumerate(comments)]
        total = len(tasks)
        
        # Use asyncio.gather to run all tasks concurrently (semaphore will limit concurrency)
        # This is more efficient than batching since the semaphore already handles concurrency control
        print(f"[DQI] Starting parallel translation of {total} comments...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and track progress
        translated_comments = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[DQI] Error translating comment {i+1}: {result}")
                # Use original comment on error
                translated_comments.append(comments[i])
            else:
                translated_comments.append(result)
            
            # Show progress every 10 comments or at completion
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[DQI] Translated {i+1}/{total} comments...")
        
        print(f"[DQI] Translation completed: {total} comments processed")
        if truncation_warnings:
            print(f"[DQI] ⚠️  {len(truncation_warnings)} comment(s) may have been truncated during translation")
        
        # Return both translated comments and truncation warnings
        return translated_comments, truncation_warnings
    
    def _strip_thinking_blocks(self, text: str) -> str:
        """
        Remove <thinking>...</thinking> blocks from text.
        
        These blocks contain internal reasoning that should not be:
        - Counted in character counts
        - Passed to translation
        - Assessed for DQI
        """
        # Remove <thinking>...</thinking> blocks (case-insensitive, handles multiline)
        stripped = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Also handle variants like <Thinking> or unclosed tags
        stripped = re.sub(r'<thinking>.*?$', '', stripped, flags=re.DOTALL | re.IGNORECASE)
        # Clean up any resulting double whitespace
        stripped = re.sub(r'\n\s*\n\s*\n', '\n\n', stripped)
        return stripped.strip()
    
    def _transcript_to_comments(self, transcript_text: str, strip_thinking: bool = True) -> List[Dict[str, Any]]:
        """
        Convert turn-based transcript to individual comments.
        
        Transcript format:
        Turn 1 (instance_id): response text...
        Turn 2 (instance_id): response text...
        
        Args:
            transcript_text: The raw transcript text
            strip_thinking: If True, remove <thinking>...</thinking> blocks from comments
        
        Returns list of dicts with 'comment' and metadata.
        """
        comments = []
        lines = transcript_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match "Turn N (instance_id): text"
            match = re.match(r'Turn\s+(\d+)\s+\(([^)]+)\):\s*(.*)', line, re.DOTALL)
            if match:
                turn_num = int(match.group(1))
                instance_id = match.group(2)
                comment_text = match.group(3).strip()
                
                if comment_text:  # Only add non-empty comments
                    comments.append({
                        'turn': turn_num,
                        'instance_id': instance_id,
                        'comment': comment_text
                    })
            else:
                # If line doesn't match pattern, treat as continuation or standalone comment
                if comments:
                    # Append to last comment
                    comments[-1]['comment'] += ' ' + line
                else:
                    # Standalone comment
                    comments.append({
                        'turn': len(comments) + 1,
                        'instance_id': 'unknown',
                        'comment': line
                    })
        
        # Strip <thinking>...</thinking> blocks from each comment if requested
        if strip_thinking:
            for comment in comments:
                comment['comment'] = self._strip_thinking_blocks(comment['comment'])
        
        return comments
    
    def assess_transcript(self, transcript_text: str, language: str = "en", translate: bool = True) -> Dict[str, Any]:
        """
        Assess a single transcript and return DQI scores.
        
        Args:
            transcript_text: The transcript text (turn-based format)
            language: Language of transcript ("en" or "de"). If "en" and translate=True,
                     will translate to German before assessment.
            translate: Whether to translate English text to German (default: True)
        
        Returns:
            Dictionary with DQI scores and quality indices
        """
        if not transcript_text or not transcript_text.strip():
            return {
                "error": "Empty transcript",
                "aqua_score": None,
                "quality_indices": {},
                "per_turn_scores": []
            }
        
        # Track if translation was applied
        translation_applied = False
        original_text = transcript_text
        truncation_warnings = []
        
        # Convert transcript to comments first (before translation)
        # This allows us to translate each comment individually
        comments = self._transcript_to_comments(transcript_text)
        
        # Translate each comment individually if needed (in parallel)
        if language == "en" and translate and comments:
            try:
                # Calculate original transcript length (without thinking blocks)
                original_clean_text = self._strip_thinking_blocks(transcript_text)
                original_length = len(original_clean_text)
                print(f"[DQI] Translating transcript from English to German ({len(comments)} comments, {original_length:,} chars excl. <thinking> blocks) in parallel (max {self.max_translation_concurrent} concurrent)...")
                
                # Run parallel translation (returns tuple of comments and warnings)
                translated_comments, truncation_warnings = self._run_parallel_translation(comments, self.max_translation_concurrent)
                
                # Reconstruct transcript from translated comments
                transcript_lines = []
                for comment in translated_comments:
                    transcript_lines.append(f"Turn {comment['turn']} ({comment['instance_id']}): {comment['comment']}")
                transcript_text = '\n'.join(transcript_lines)
                
                # Calculate translated transcript length
                translated_length = len(transcript_text)
                length_change = ((translated_length - original_length) / original_length * 100) if original_length > 0 else 0
                
                # Re-parse to ensure consistency
                comments = self._transcript_to_comments(transcript_text)
                translation_applied = True
                print(f"[DQI] Translation completed ({len(comments)} comments, {original_length:,} -> {translated_length:,} chars, {length_change:+.1f}%), proceeding with DQI assessment...")
            except Exception as e:
                print(f"[DQI] Warning: Translation failed: {e}")
                print("[DQI] Attempting assessment with original English text (may produce inaccurate results)")
                # Continue with original text, but note in results
                transcript_text = original_text
                comments = self._transcript_to_comments(transcript_text)
        
        if not comments:
            return {
                "error": "No comments found in transcript",
                "aqua_score": None,
                "quality_indices": {},
                "per_turn_scores": []
            }
        
        # Initialize model if needed
        self._initialize_model()
        
        # Create temporary CSV file for AQuA
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            # Create DataFrame
            df = pd.DataFrame(comments)
            # AQuA expects a 'comment' or 'comment_de' column
            df['comment_de'] = df['comment']  # Assume text is already in German or will work
            df.to_csv(f.name, sep='\t', index=False)
            temp_csv_path = f.name
        
        try:
            # Create dataset
            dataset = self.InferenceDataset(
                path_to_dataset=temp_csv_path,
                tokenizer=self._tokenizer,
                text_col='comment_de'
            )
            
            # Create dataloader (use configured batch size)
            dataloader = DataLoader(dataset, batch_size=self.batch_size)
            
            # Run inference
            output_dic = {task: [] for task in self._task2identifier.keys()}
            
            for batch in dataloader:
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self._device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self._model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )
                
                # Process outputs (one per adapter)
                for i, output in enumerate(outputs):
                    task = list(self._task2identifier.keys())[i]
                    predictions = output.logits
                    prediction = torch.argmax(predictions, axis=1).detach().cpu().numpy()
                    output_dic[task].extend(prediction)
            
            # Calculate scores per comment
            per_turn_scores = []
            for idx, comment in enumerate(comments):
                comment_scores = {}
                comment_weighted_sum = 0.0
                
                for task in self._task2identifier.keys():
                    score = output_dic[task][idx]
                    weight = self._task2weight[task]
                    comment_scores[task] = int(score)
                    comment_weighted_sum += score * weight
                
                # Normalize score
                normalized_score = self._normalize_scores(np.array([comment_weighted_sum]))[0]
                
                per_turn_scores.append({
                    'turn': comment['turn'],
                    'instance_id': comment['instance_id'],
                    'aqua_score': float(normalized_score),
                    'quality_indices': comment_scores
                })
            
            # Calculate aggregate scores
            all_aqua_scores = [s['aqua_score'] for s in per_turn_scores]
            aggregate_aqua_score = float(np.mean(all_aqua_scores)) if all_aqua_scores else None
            
            # Aggregate quality indices (mean across all comments)
            aggregate_indices = {}
            for task in self._task2identifier.keys():
                task_scores = [s['quality_indices'][task] for s in per_turn_scores]
                aggregate_indices[task] = float(np.mean(task_scores)) if task_scores else 0.0
            
            result = {
                "assessment_timestamp": datetime.now().isoformat(),
                "aqua_score": aggregate_aqua_score,
                "quality_indices": aggregate_indices,
                "per_turn_scores": per_turn_scores,
                "num_comments": len(comments),
                "num_turns": len(set(c['turn'] for c in comments)),
                "translation_applied": translation_applied,
                "original_language": language,
                "truncation_warnings": truncation_warnings if truncation_warnings else None
            }
            return result
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
    
    def assess_transcript_file(self, transcript_path: str) -> Dict[str, Any]:
        """
        Assess a transcript file and return DQI scores.
        
        Args:
            transcript_path: Path to transcript file
        
        Returns:
            Dictionary with DQI scores and quality indices
        """
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        result = self.assess_transcript(transcript_text)
        result["transcript_path"] = transcript_path
        return result
    
    def assess_transcripts_batch(self, transcript_paths: List[str], batch_size: int = 8) -> Dict[str, Dict[str, Any]]:
        """
        Assess multiple transcripts in batch.
        
        Args:
            transcript_paths: List of paths to transcript files
            batch_size: Batch size for processing (not used currently, kept for API compatibility)
        
        Returns:
            Dictionary mapping transcript paths to their DQI scores
        """
        results = {}
        for transcript_path in transcript_paths:
            try:
                results[transcript_path] = self.assess_transcript_file(transcript_path)
            except Exception as e:
                results[transcript_path] = {
                    "error": str(e),
                    "transcript_path": transcript_path
                }
        return results

