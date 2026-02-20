import asyncio
import json
import yaml
import numpy as np
from typing import Dict, Any, List, Optional
import random
from datetime import datetime
import os
from enum import Enum

from .utils.config import SimulationConfig, DeliberationProtocol
from .llm.manager import LLMManager, LLMInstance
from .survey.dri_survey import DRISurvey, DRISurveyManager, SurveyResponse
from .deliberation.engine import DeliberationEngine
from .analysis.dri_calculator import DRICalculator
from .visualization.dri_plots import generate_dri_plots_for_iteration

class SimulationRunner:
    """Orchestrates the complete deliberation simulation workflow."""

    def __init__(self, config: SimulationConfig, survey: DRISurvey = None):
        self.config = config
        self.survey = survey
        self.survey_manager = DRISurveyManager(survey, config.deliberation_tool_choice) if survey else None
        self.deliberation_engine = DeliberationEngine(
            config.group_size,
            config.group_assignment_strategy,
            config.deliberation_tool_choice,
            survey,
            deliberation_prompt_type=getattr(config, 'deliberation_prompt_type', 'control_1')
        )
        self.dri_calculator = DRICalculator() if survey else None
        
    # --- Wrapper for running a single repetition (for parallel execution) ---
    async def _run_repetition_wrapper(self, repetition_index: int, 
                                    repetition_semaphore: Optional[asyncio.Semaphore]) -> Optional[Dict[str, Any]]:
        """Handles the execution for one repetition, creating its own LLM instances and an optional semaphore."""
        MAX_REPETITION_RETRIES = 3
        RETRY_BASE_DELAY = 60  # seconds
        
        for attempt in range(MAX_REPETITION_RETRIES):
            print(f"--- Preparing Repetition {repetition_index + 1}/{self.config.num_repetitions} (Parallel Task, attempt {attempt + 1}) ---")
            
            # Create separate LLM instances for this repetition
            repetition_llm_manager = LLMManager(self.config.llm_setup)
            try:
                print(f"  [INFO] Initializing separate LLM instances for repetition {repetition_index + 1}...")
                await repetition_llm_manager.initialize_instances()
                repetition_instances = repetition_llm_manager.get_instances()
                
                if not repetition_instances:
                    print(f"  [ERROR] No LLM instances available for repetition {repetition_index + 1}. Skipping.")
                    return { 'iteration': repetition_index, 'error': 'Failed to initialize LLM instances' }
                
                iteration_result = None
                # Acquire repetition semaphore if provided
                if repetition_semaphore:
                    async with repetition_semaphore:
                        print(f"  Rep {repetition_index + 1} acquired semaphore. Starting execution.")
                        iteration_result = await self.run_single_iteration(repetition_index, repetition_instances)
                        print(f"  Rep {repetition_index + 1} finished execution, releasing semaphore.")
                else: # Run without semaphore if not provided
                    iteration_result = await self.run_single_iteration(repetition_index, repetition_instances)
                
                # Check if the result indicates a failed run (empty transcripts due to rate limits)
                if iteration_result and 'deliberation_transcripts' in iteration_result:
                    transcripts = iteration_result.get('deliberation_transcripts', {})
                    if not transcripts or all(t is None or (isinstance(t, dict) and not t.get('main')) for t in transcripts.values()):
                        raise Exception("All deliberation groups failed - empty transcripts")
                
                return iteration_result
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = 'ratelimit' in error_str or 'rate_limit' in error_str or 'rate limit' in error_str or 'retryerror' in error_str or 'empty transcripts' in error_str
                
                if is_rate_limit and attempt < MAX_REPETITION_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)  # Exponential backoff: 60s, 120s, 240s
                    print(f"[RETRY] Repetition {repetition_index + 1} failed with rate limit error, attempt {attempt + 1}/{MAX_REPETITION_RETRIES}. Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"[ERROR] Repetition {repetition_index + 1} failed after {attempt + 1} attempts: {e}")
                    import traceback
                    print(traceback.format_exc())
                    return { 'iteration': repetition_index, 'error': str(e) }
            finally:
                # Cleanup LLM instances for this repetition
                if repetition_llm_manager:
                    print(f"  [INFO] Cleaning up LLM instances for repetition {repetition_index + 1}...")
                    await repetition_llm_manager.cleanup_assistants()
        
        print(f"--- Finished Repetition {repetition_index + 1}/{self.config.num_repetitions} (Parallel Task) ---")
        return { 'iteration': repetition_index, 'error': 'Max retries exceeded' }

    async def run(self, output_dir: str) -> Dict[str, Any]:
        """Run the complete simulation, potentially in parallel."""
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

        results = []
        aggregated_results = {}
        print(f"\nStarting {self.config.num_repetitions} simulation repetition(s)...")

        # Each repetition will create its own LLM instances (no shared instances)
        try:
            # --- Conditional Parallel vs Sequential Execution ---
            if self.config.parallelize_repetitions:
                # Setup semaphore for limiting parallel repetitions if configured
                repetition_semaphore = None
                if isinstance(self.config.max_parallel_repetitions, int) and self.config.max_parallel_repetitions > 0:
                    repetition_semaphore = asyncio.Semaphore(self.config.max_parallel_repetitions)
                    print(f"[INFO] Running {self.config.num_repetitions} repetitions in PARALLEL with concurrency limit: {self.config.max_parallel_repetitions} (Separate LLM Instances per Repetition).")
                else:
                     print(f"[INFO] Running {self.config.num_repetitions} repetitions in PARALLEL (unlimited concurrency) (Separate LLM Instances per Repetition).")

                tasks = [
                    self._run_repetition_wrapper(
                        i, 
                        repetition_semaphore # Pass the semaphore to the wrapper
                    ) 
                    for i in range(self.config.num_repetitions)
                ]
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)

                results = []
                for i, res in enumerate(raw_results):
                    if isinstance(res, Exception):
                        print(f"[ERROR] Repetition {i+1} (parallel) returned an exception: {res}")
                    elif isinstance(res, dict):
                        if res.get('error'):
                             print(f"[WARN] Repetition {i+1} (parallel) reported an error: {res['error']}")
                        else:
                             results.append(res)
                    else:
                        print(f"[WARN] Repetition {i+1} (parallel) returned unexpected type or None: {type(res)}")

            else:
                print(f"[INFO] Running {self.config.num_repetitions} repetitions SEQUENTIALLY (Separate LLM Instances per Repetition).")
                MAX_REPETITION_RETRIES = 3
                RETRY_BASE_DELAY = 60  # seconds
                
                for i in range(self.config.num_repetitions):
                    for attempt in range(MAX_REPETITION_RETRIES):
                        print(f"\n--- Repetition {i + 1}/{self.config.num_repetitions} (Sequential, attempt {attempt + 1}) ---")
                        
                        # Create separate LLM instances for this repetition
                        repetition_llm_manager = LLMManager(self.config.llm_setup)
                        try:
                            print(f"  [INFO] Initializing separate LLM instances for repetition {i + 1}...")
                            await repetition_llm_manager.initialize_instances()
                            repetition_instances = repetition_llm_manager.get_instances()
                            
                            if not repetition_instances:
                                print(f"  [ERROR] No LLM instances available for repetition {i + 1}. Skipping.")
                                break  # Skip to next repetition
                            
                            iteration_result = await self.run_single_iteration(i, repetition_instances)
                            
                            # Check if the result indicates a failed run (empty transcripts due to rate limits)
                            if iteration_result and 'deliberation_transcripts' in iteration_result:
                                transcripts = iteration_result.get('deliberation_transcripts', {})
                                if not transcripts or all(t is None or (isinstance(t, dict) and not t.get('main')) for t in transcripts.values()):
                                    raise Exception("All deliberation groups failed - empty transcripts")
                            
                            results.append(iteration_result)
                            break  # Success - move to next repetition
                        except Exception as e:
                            error_str = str(e).lower()
                            is_rate_limit = 'ratelimit' in error_str or 'rate_limit' in error_str or 'rate limit' in error_str or 'retryerror' in error_str or 'empty transcripts' in error_str
                            
                            if is_rate_limit and attempt < MAX_REPETITION_RETRIES - 1:
                                delay = RETRY_BASE_DELAY * (2 ** attempt)
                                print(f"[RETRY] Repetition {i + 1} failed with rate limit error, attempt {attempt + 1}/{MAX_REPETITION_RETRIES}. Waiting {delay}s before retry...")
                                await asyncio.sleep(delay)
                                # Continue to next attempt
                            else:
                                print(f"[ERROR] Repetition {i + 1} failed after {attempt + 1} attempts: {e}")
                                import traceback
                                print(traceback.format_exc())
                                break  # Give up on this repetition
                        finally:
                            # Cleanup LLM instances for this repetition
                            if repetition_llm_manager:
                                print(f"  [INFO] Cleaning up LLM instances for repetition {i + 1}...")
                                await repetition_llm_manager.cleanup_assistants()
            
        except Exception as e_run_setup: # Catch errors during repetition loop dispatch
            print(f"[CRITICAL_ERROR] Failed during main repetition loop dispatch: {e_run_setup}")
            import traceback
            print(traceback.format_exc())
            return { 'error': f'Critical error during run setup: {e_run_setup}' }

        print(f"\nFinished {self.config.num_repetitions} simulation repetition(s). Total successful repetitions processed: {len(results)}")

        if results:
            aggregated_results = self._aggregate_results(results)
            
            output_path = os.path.join(output_dir, "simulation_results.json")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.json"
            output_path = os.path.join(output_dir, filename)
            
            print(f"\nSaving results to: {output_path}")
            with open(output_path, 'w') as f:
                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, (datetime)):
                            return obj.isoformat()
                        if hasattr(obj, 'dict'):
                            return obj.dict()
                        return super(NpEncoder, self).default(obj)
                json.dump(aggregated_results, f, indent=2, cls=NpEncoder)
            
            # Export CSV for easy analysis
            self._export_csv(results, output_dir, timestamp)
        else:
            print("[Warning] No iterations completed successfully, skipping aggregation and saving.")

        # Add timestamp to transcript filenames
        transcript_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for repetition_idx, result in enumerate(results):
            # If deliberation transcripts are present, write them to readable files
            transcripts = result.get('deliberation_transcripts')
            if transcripts:
                for group_id, transcript_data in transcripts.items():
                    # Sanitize group_id for use in filenames (replace path separators and other unsafe characters)
                    safe_group_id = group_id.replace('/', '-').replace('\\', '-').replace(' ', '_')
                    
                    # transcript_data is now a dict with 'main' and 'with_reasoning'
                    if isinstance(transcript_data, dict):
                        transcript_filename_main = f"deliberation_transcript_{repetition_idx}_{safe_group_id}_{transcript_timestamp}_main.txt"
                        transcript_filename_reasoning = f"deliberation_transcript_{repetition_idx}_{safe_group_id}_{transcript_timestamp}_with_reasoning.txt"
                        transcript_path_main = os.path.join(output_dir, transcript_filename_main)
                        transcript_path_reasoning = os.path.join(output_dir, transcript_filename_reasoning)
                        with open(transcript_path_main, 'w', encoding='utf-8') as f:
                            f.write(transcript_data.get('main', str(transcript_data)))
                        with open(transcript_path_reasoning, 'w', encoding='utf-8') as f:
                            f.write(transcript_data.get('with_reasoning', str(transcript_data)))
                        # Save detailed turn logs (prompt + response) if available
                        turn_logs = transcript_data.get('turn_logs')
                        if turn_logs:
                            turn_log_filename = f"deliberation_turn_log_{repetition_idx}_{safe_group_id}_{transcript_timestamp}.json"
                            turn_log_path = os.path.join(output_dir, turn_log_filename)
                            with open(turn_log_path, 'w', encoding='utf-8') as f:
                                json.dump(turn_logs, f, indent=2)
                    else:
                        # Fallback for old format (string) - should not happen but handle gracefully
                        print(f"Warning: transcript_data for {group_id} is not a dict, converting to dict format")
                        transcript_filename_main = f"deliberation_transcript_{repetition_idx}_{safe_group_id}_{transcript_timestamp}_main.txt"
                        transcript_path_main = os.path.join(output_dir, transcript_filename_main)
                        with open(transcript_path_main, 'w', encoding='utf-8') as f:
                            f.write(str(transcript_data))
            
            # Save DRI survey prompts if available
            pre_delib = result.get('pre_deliberation', {})
            post_delib = result.get('post_deliberation', {})
            
            if pre_delib and pre_delib.get('prompts'):
                pre_prompts_filename = f"dri_survey_prompts_pre_{repetition_idx}_{transcript_timestamp}.json"
                pre_prompts_path = os.path.join(output_dir, pre_prompts_filename)
                with open(pre_prompts_path, 'w', encoding='utf-8') as f:
                    json.dump(pre_delib['prompts'], f, indent=2)
            
            if post_delib and post_delib.get('prompts'):
                post_prompts_filename = f"dri_survey_prompts_post_{repetition_idx}_{transcript_timestamp}.json"
                post_prompts_path = os.path.join(output_dir, post_prompts_filename)
                with open(post_prompts_path, 'w', encoding='utf-8') as f:
                    json.dump(post_delib['prompts'], f, indent=2)

        # Generate DRI plots if DRI assessment is enabled
        if self.survey is not None:  # DRI assessment is enabled
            print("\n--- Generating DRI Scatter Plots ---")
            for repetition_idx, result in enumerate(results):
                try:
                    generate_dri_plots_for_iteration(result, output_dir, transcript_timestamp)
                except Exception as e:
                    print(f"Warning: Failed to generate DRI plots for repetition {repetition_idx}: {e}")

        # Assess DQI for transcripts if enabled
        if getattr(self.config, 'assess_dqi', False):
            print("\n--- Assessing DQI for transcripts ---")
            try:
                from .analysis.dqi_assessor import DQIAssessor
                dqi_assessor = DQIAssessor()
                
                for repetition_idx, result in enumerate(results):
                    transcripts = result.get('deliberation_transcripts')
                    if transcripts:
                        for group_id, transcript_data in transcripts.items():
                            # Use main transcript (without reasoning)
                            transcript_text = transcript_data.get('main', '') if isinstance(transcript_data, dict) else str(transcript_data)
                            
                            if transcript_text:
                                try:
                                    dqi_scores = dqi_assessor.assess_transcript(transcript_text)
                                    
                                    # Save DQI scores to JSON file
                                    safe_group_id = group_id.replace('/', '-').replace('\\', '-').replace(' ', '_')
                                    dqi_filename = f"dqi_scores_{repetition_idx}_{safe_group_id}_{transcript_timestamp}.json"
                                    dqi_path = os.path.join(output_dir, dqi_filename)
                                    with open(dqi_path, 'w', encoding='utf-8') as f:
                                        json.dump(dqi_scores, f, indent=2)
                                    
                                    # Also add to results for aggregation
                                    if 'dqi_scores' not in result:
                                        result['dqi_scores'] = {}
                                    result['dqi_scores'][group_id] = dqi_scores
                                    
                                    print(f"  ✓ Assessed DQI for {group_id} (AQuA score: {dqi_scores.get('aqua_score', 'N/A'):.3f})")
                                except Exception as e:
                                    print(f"  ✗ Failed to assess DQI for {group_id}: {e}")
                                    import traceback
                                    print(traceback.format_exc())
            except ImportError as e:
                print(f"[WARN] AQuA not available for DQI assessment: {e}")
                print("  Install AQuA dependencies or set assess_dqi=false")
            except Exception as e:
                print(f"[ERROR] Failed to initialize DQI assessor: {e}")
                import traceback
                print(traceback.format_exc())

        return aggregated_results

    async def run_single_iteration(self, iteration: int, instances: List[LLMInstance]) -> Dict[str, Any]:
        print(f"Starting iteration {iteration + 1}/{self.config.num_repetitions} with {len(instances)} instances...")
        
        if not instances:
            print("[Warning] No instances available for iteration {iteration+1}")
            return { 'iteration': iteration, 'error': 'No available LLM instances' }

        # Always administer pre-deliberation survey

        print("Administering pre-deliberation survey...")
        pre_survey_results_raw = await self.survey_manager.administer_survey_to_group(instances)
        pre_parsed_responses: Dict[str, Optional[SurveyResponse]] = {
             k: v.get('parsed') for k, v in pre_survey_results_raw.items()
        }
        
        # --- Calculate Pre-Deliberation DRI ---
        pre_individual_scores, pre_group_score, pre_ic_df = self.dri_calculator.calculate_dri_scores(
             pre_parsed_responses
        )

        # --- Analyze Pre-Deliberation Variance (Store Full Dicts) ---
        pre_consideration_variances = self.dri_calculator.analyze_response_variance(
            pre_parsed_responses, self.survey.considerations, 'consideration'
        )
        pre_preference_variances = self.dri_calculator.analyze_response_variance(
            pre_parsed_responses, self.survey.preferences, 'preference'
        )
        pre_high_c_id, pre_high_c_var = self.dri_calculator.find_highest_variance(pre_consideration_variances)
        pre_high_p_id, pre_high_p_var = self.dri_calculator.find_highest_variance(pre_preference_variances)

        # Store highest variance info for results file
        pre_variance_analysis_summary = {
            'highest_consideration': {'id': pre_high_c_id, 'variance': pre_high_c_var},
            'highest_preference': {'id': pre_high_p_id, 'variance': pre_high_p_var}
        }
        print(f"  Pre-Variance Analysis Summary: {pre_variance_analysis_summary}") 
        print("\nPre-deliberation DRI Scores:")
        print(f"Group Score: {pre_group_score:.3f}")
        print("Individual Scores:")
        for instance_id, score in pre_individual_scores.items():
            print(f"  {instance_id}: {score:.3f}")
        
        if self.config.pre_deliberation_only:
            return {
                'iteration': iteration,
                'pre_deliberation': {
                    'individual_scores': pre_individual_scores,
                    'group_score': pre_group_score,
                    'raw_responses': {k: v.get('raw') for k, v in pre_survey_results_raw.items()},
                    'parsed_responses': pre_parsed_responses,
                    'prompts': {k: v.get('prompt') for k, v in pre_survey_results_raw.items()},
                    'variance_analysis': pre_variance_analysis_summary
                }
            }
        
        # Run deliberation if enabled
        deliberation_transcripts = {}
        groups = self.deliberation_engine.assign_groups(instances)

        if getattr(self.config, 'dri_conversation_enabled', True):
            print("\nRunning deliberation sessions...")
            deliberation_transcripts = await self.deliberation_engine.run_deliberation(
                groups=groups,
                topic=self.config.deliberation_topic,
                num_turns=self.config.num_deliberation_turns,
                protocol=self.config.deliberation_protocol,
                display_conversation=self.config.display_conversation,
                consideration_variances=pre_consideration_variances,
                preference_variances=pre_preference_variances,
            )
        else:
            print("\nDeliberation disabled. Skipping deliberation sessions.")
            for group in groups:
                deliberation_transcripts[group.group_id] = {"main": "", "with_reasoning": ""}

        print("\nAdministering post-deliberation survey...")
        post_survey_results_raw = {}
        for group in groups:
            group_transcript = deliberation_transcripts.get(group.group_id, {})
            transcript_text = group_transcript.get("main", "") if isinstance(group_transcript, dict) else str(group_transcript)

            group_responses = await self.survey_manager.administer_survey_to_group(
                group.instances,
                context=transcript_text,
            )
            post_survey_results_raw.update(group_responses)

        # Extract parsed responses
        post_parsed_responses: Dict[str, Optional[SurveyResponse]] = {
            k: v.get('parsed') for k, v in post_survey_results_raw.items()
        }

        # --- Calculate Post-Deliberation DRI ---
        post_individual_scores, post_group_score, post_ic_df = self.dri_calculator.calculate_dri_scores(
            post_parsed_responses # Pass dict of SurveyResponse objects
        )

        # --- Analyze Post-Deliberation Variance (Store Full Dicts) ---
        post_consideration_variances = self.dri_calculator.analyze_response_variance(
            post_parsed_responses, self.survey.considerations, 'consideration'
        )
        post_preference_variances = self.dri_calculator.analyze_response_variance(
            post_parsed_responses, self.survey.preferences, 'preference'
        )
        post_high_c_id, post_high_c_var = self.dri_calculator.find_highest_variance(post_consideration_variances)
        post_high_p_id, post_high_p_var = self.dri_calculator.find_highest_variance(post_preference_variances)

        # Store highest variance info for results file
        post_variance_analysis_summary = {
            'highest_consideration': {'id': post_high_c_id, 'variance': post_high_c_var},
            'highest_preference': {'id': post_high_p_id, 'variance': post_high_p_var}
        }
        print(f"  Post-Variance Analysis Summary: {post_variance_analysis_summary}") 
        print("\nPost-deliberation DRI Scores:")
        print(f"Group Score: {post_group_score:.3f}")
        print("Individual Scores:")
        for instance_id, score in post_individual_scores.items():
            print(f"  {instance_id}: {score:.3f}")
        
        changes = self.dri_calculator.analyze_dri_changes(
            pre_individual_scores,
            post_individual_scores
        )
        
        print("\nDRI Score Changes:")
        print(f"Mean Change: {changes['mean_change']:.3f}")
        print(f"Standard Deviation: {changes['std_change']:.3f}")
        
        return {
            'iteration': iteration,
            'pre_deliberation': {
                'individual_scores': pre_individual_scores,
                'group_score': pre_group_score,
                'raw_responses': {k: v.get('raw') for k, v in pre_survey_results_raw.items()},
                'parsed_responses': pre_parsed_responses,
                'prompts': {k: v.get('prompt') for k, v in pre_survey_results_raw.items()},
                'variance_analysis': pre_variance_analysis_summary,
                'ic_dataframe': pre_ic_df.to_dict('records') if not pre_ic_df.empty else None
            },
            'post_deliberation': {
                'individual_scores': post_individual_scores,
                'group_score': post_group_score,
                'raw_responses': {k: v.get('raw') for k, v in post_survey_results_raw.items()},
                'parsed_responses': post_parsed_responses,
                'prompts': {k: v.get('prompt') for k, v in post_survey_results_raw.items()},
                'variance_analysis': post_variance_analysis_summary,
                'ic_dataframe': post_ic_df.to_dict('records') if not post_ic_df.empty else None
            },
            'changes': changes,
            'deliberation_transcripts': deliberation_transcripts
        }
        
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all iterations."""
        # If results are deliberation-only (no DRI), aggregate accordingly
        if not results or 'pre_deliberation' not in results[0]:
            return {
                'configuration': self.config.model_dump(),
                'summary': {
                    'num_observations': len(results),
                    'deliberation_only': True
                },
                'detailed_results': results
            }

        pre_group_scores = np.array([r['pre_deliberation']['group_score'] for r in results])
        post_group_scores = np.array([r['post_deliberation']['group_score'] for r in results])
        delta_group_scores = post_group_scores - pre_group_scores # Calculate change in group score for each iteration
        
        # Calculate individual change stats if available (might be missing if pre_deliberation_only=True)
        individual_mean_changes = [r.get('changes', {}).get('mean_change') for r in results]
        individual_std_changes = [r.get('changes', {}).get('std_change') for r in results]
        # Filter out None values in case 'changes' was missing
        valid_individual_mean_changes = [c for c in individual_mean_changes if c is not None]
        valid_individual_std_changes = [c for c in individual_std_changes if c is not None]

        return {
            'configuration': self.config.dict(), # All config parameters
            'summary': {
                'num_observations': len(results), # Number of iterations/observations
                'pre_deliberation': {
                    'mean_group_score': np.mean(pre_group_scores),
                    'median_group_score': np.median(pre_group_scores), # Added median
                    'std_group_score': np.std(pre_group_scores)
                },
                'post_deliberation': {
                    'mean_group_score': np.mean(post_group_scores),
                    'median_group_score': np.median(post_group_scores), # Added median
                    'std_group_score': np.std(post_group_scores)
                },
                'group_dri_change': { # Renamed and recalculated based on group score changes
                    'mean_delta_group_dri': np.mean(delta_group_scores), 
                    'median_delta_group_dri': np.median(delta_group_scores),
                    'std_dev_delta_group_dri': np.std(delta_group_scores) # Std Dev of Group DRI Change
                },
                'individual_dri_change': { # Keep individual stats separate
                    'mean_of_mean_individual_change': np.mean(valid_individual_mean_changes) if valid_individual_mean_changes else None,
                    'std_dev_of_mean_individual_change': np.std(valid_individual_mean_changes) if valid_individual_mean_changes else None,
                    'mean_of_std_dev_individual_change': np.mean(valid_individual_std_changes) if valid_individual_std_changes else None # Mean of the per-iteration std dev
                }
            },
            'detailed_results': results # DRI per simulation (within each dict) and other details
        }
    
    def _export_csv(self, results: List[Dict[str, Any]], output_dir: str, timestamp: str):
        """Export results to CSV for easy analysis."""
        import pandas as pd
        
        csv_data = []
        
        for result in results:
            iteration = result.get('iteration', 0)
            
            # Extract configuration info
            model_name = "unknown"
            temperature = self.config.global_temperature
            condition = "treatment" if getattr(self.config, 'dri_assessment_enabled', True) else "control"
            topic = getattr(self.config, 'survey_name', 'unknown')
            
            # Extract from LLM setup if available
            if hasattr(self.config, 'llm_setup') and self.config.llm_setup:
                enabled_models = [setup.model_name for setup in self.config.llm_setup if setup.enabled]
                if enabled_models:
                    model_name = enabled_models[0]  # Take first enabled model
            
            # Extract DRI scores
            pre_dri = None
            post_dri = None
            
            if 'pre_deliberation' in result:
                pre_dri = result['pre_deliberation'].get('group_score')
            if 'post_deliberation' in result:
                post_dri = result['post_deliberation'].get('group_score')
            
            # Create row for this iteration
            row = {
                'model': model_name,
                'condition': condition,
                'topic': topic,
                'run_id': iteration,
                'timestamp': timestamp,
                'pre_dri': pre_dri,
                'post_dri': post_dri,
                'temperature': temperature,
                'group_size': getattr(self.config, 'group_size', None),
                'num_turns': getattr(self.config, 'num_deliberation_turns', None),
                'protocol': getattr(self.config, 'deliberation_protocol', None)
            }
            
            csv_data.append(row)
        
        # Create DataFrame and save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_filename = f"pilot_data_{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"Exported CSV data to: {csv_path}")
            print(f"CSV contains {len(df)} rows with columns: {list(df.columns)}")
        else:
            print("[Warning] No data to export to CSV")

# --- Helper Function for Parameter Sweep Output Directory --- 
def _generate_run_dir_name(base_dir: str, sweep_params: Dict[str, Any]) -> str:
    """Generates a descriptive directory name from sweep parameters."""
    name_parts = []
    # Define a consistent order or sort keys for reproducibility
    for key in sorted(sweep_params.keys()):
        value = sweep_params[key]
        # Shorten key names if needed
        short_key = key.replace("num_deliberation_turns", "turns")\
                     .replace("group_size", "size")\
                     .replace("deliberation_protocol", "proto")\
                     .replace("global_temperature", "temp")
        # Format value nicely
        if isinstance(value, float):
            value_str = f"{value:.1f}".replace(".", "p") # Replace decimal point
        else:
            value_str = str(value)
        name_parts.append(f"{short_key}_{value_str}")
    run_name = "__".join(name_parts)
    return os.path.join(base_dir, run_name)

# --- Restore Original Main Function Signature (or similar) --- 
async def main(config: SimulationConfig, survey: DRISurvey, output_dir_base: Optional[str] = None):
    """DEPRECATED/INTERNAL: Runs the simulation for a SINGLE config/survey.
    
    This function is now primarily intended to be called by an external 
    script (like the main run_simulation.py) that handles config loading, 
    survey loading, and parameter sweep orchestration.
    
    Args:
        config: The specific SimulationConfig for this run.
        survey: The loaded DRISurvey object.
        output_dir_base: The base directory where results for this specific run 
                         should be saved. If None, uses a default.
    """
    # Determine output directory for this specific run
    if output_dir_base is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("delib_sim", "data", "simulation_results", f"run_{timestamp}")
        print(f"[WARN] No output directory provided to main(), using default: {output_dir}")
    else:
        output_dir = output_dir_base
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Running simulation with output to: {output_dir} ---")
    
    runner = SimulationRunner(config, survey)
    try:
        # Pass the specific output directory to the run method
        results = await runner.run(output_dir=output_dir) 
        
        if results: # Check if results were returned
            print("\nSimulation Completed!")
            print(f"Results saved in: {output_dir}")
            print("\nSummary for this run:")
            # Access summary stats safely
            summary = results.get('summary', {})
            pre_summary = summary.get('pre_deliberation', {})
            post_summary = summary.get('post_deliberation', {})
            change_summary = summary.get('group_dri_change', {})
            
            print(f"Mean pre-deliberation group DRI: {pre_summary.get('mean_group_score', float('nan')):.3f}")
            print(f"Mean post-deliberation group DRI: {post_summary.get('mean_group_score', float('nan')):.3f}")
            print(f"Mean change in DRI: {change_summary.get('mean_delta_group_dri', float('nan')):.3f}")
            return results # Return results for potential aggregation upstream
        else:
            print("[WARN] Runner did not return results for this configuration.")
            return None
            
    except Exception as e:
        print(f"[ERROR] Simulation run failed: {e}")
        import traceback
        print(traceback.format_exc())
        # Log the error to a file in the run_output_dir
        error_log_path = os.path.join(output_dir, "error.log")
        try:
            with open(error_log_path, 'w') as f_err:
                 f_err.write(f"Run failed for configuration:\n{yaml.dump(config.dict())}\n")
                 f_err.write(f"\nError:\n{e}\n")
                 f_err.write(f"\nTraceback:\n{traceback.format_exc()}")
        except Exception as log_e:
            print(f"[ERROR] Failed to write error log to {error_log_path}: {log_e}")
        return None # Indicate failure

# --- NpEncoder Class (Remains the same) --- 
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime)):
            return obj.isoformat()
        if hasattr(obj, 'dict'): # For Pydantic models like SimulationConfig
            return obj.dict()
        # Add handling for Enum types like DeliberationProtocol if needed
        if isinstance(obj, Enum):
             return obj.value
        return super(NpEncoder, self).default(obj)

# --- Entry Point Logic (Should be removed or commented out) ---
# This module is not intended to be run directly anymore.
# The main entry point is the top-level run_simulation.py script.
# if __name__ == "__main__":
#     import sys
#     from enum import Enum # Added for NpEncoder
# 
#     if len(sys.argv) > 1:
#         config_file_path = sys.argv[1]
#     else:
#         config_file_path = os.path.join("delib_sim", "config", "simulation_config.yaml") # Default path
#         print(f"No config path provided, using default: {config_file_path}")
# 
#     if not os.path.exists(config_file_path):
#         print(f"[ERROR] Configuration file not found at: {config_file_path}")
#         sys.exit(1)
# 
#     # This direct call is no longer correct as main needs config/survey objects
#     # asyncio.run(main(config_path=config_file_path)) 