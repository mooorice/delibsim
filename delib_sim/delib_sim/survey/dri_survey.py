from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from pydantic import BaseModel
from ..llm.manager import LLMInstance
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import re # Import regex module
import traceback # Import traceback
import asyncio # Import asyncio for parallel execution

class SurveyGenerationFailedError(Exception):
    """Custom exception for when LLM generation fails before parsing."""
    pass

class Consideration(BaseModel):
    id: str
    text: str
    description: Optional[str] = None

class Preference(BaseModel):
    id: str
    text: str
    description: Optional[str] = None

class DRISurvey(BaseModel):
    issue_topic: str
    considerations: List[Consideration]
    preferences: List[Preference]
    response_format: Dict[str, Any]

class SurveyResponse(BaseModel):
    instance_id: str
    consideration_ratings: Dict[str, int]  # consideration_id -> rating (1-5)
    preference_rankings: Dict[str, int]    # preference_id -> rank (1-n)

class DRISurveyManager:
    def __init__(self, survey: DRISurvey, deliberation_tool_choice: Union[Literal["auto", "none", "required"], Dict[str, Any]] = "auto"):
        self.survey = survey
        self.deliberation_tool_choice = deliberation_tool_choice
        self.context_doc_warning_shown = False
    
    def _format_consideration_prompt(self) -> str:
        """Format the considerations section of the survey prompt."""
        scale_min = self.survey.response_format['consideration_scale']['min']
        scale_max = self.survey.response_format['consideration_scale']['max']
        considerations_text = f"Indicate your agreement to each consideration on a scale of {scale_min}-{scale_max} with {scale_min} being strongly disagree and {scale_max} being strongly agree:\\n\\n"
        for consideration in self.survey.considerations:
            considerations_text += f"{consideration.id}. {consideration.text}\\n"
            if consideration.description:
                considerations_text += f"   Description: {consideration.description}\\n"
            considerations_text += f"   Rating ({scale_min}-{scale_max}): " # Use dynamic scale
        return considerations_text
    
    def _format_preference_prompt(self) -> str:
        """Format the preferences section of the survey prompt."""
        num_preferences = len(self.survey.preferences)
        preferences_text = f"\\n Rank the following {num_preferences} preferences (1 being highest priority):\\n\\n"
        for preference in self.survey.preferences:
            preferences_text += f"{preference.id}. {preference.text}\\n"
            if preference.description:
                preferences_text += f"   Description: {preference.description}\\n"
            preferences_text += "   Rank: "
        return preferences_text
    
    def _format_pre_response(self, pre_response: SurveyResponse) -> str:
        """Formats the pre-deliberation response for inclusion in the prompt."""
        if not pre_response:
            return ""
            
        response_text = "Here were your previous answers:\n\n"
        response_text += "Considerations Ratings:\n"
        if pre_response.consideration_ratings:
            for c_id, rating in pre_response.consideration_ratings.items():
                response_text += f"- {c_id}: {rating}\n"
        else:
            response_text += "- (No previous ratings found)\n"
        
        response_text += "\nPreference Rankings:\n"
        if pre_response.preference_rankings:
            for p_id, rank in pre_response.preference_rankings.items():
                response_text += f"- {p_id}: {rank}\n"
        else:
            response_text += "- (No previous rankings found)\n"
            
        response_text += "\nPlease review the discussion context and update your ratings and rankings below based on the deliberation.\n"
        return response_text
    
    def _format_survey_prompt(self, context: Optional[str] = None, pre_response: Optional[SurveyResponse] = None) -> str:
        """Format the complete survey prompt, potentially including context and previous answers."""
        # Use the actual issue topic from the loaded survey data
        prompt = f"Topic: {self.survey.issue_topic}\\n\\n"
        
        if context:
            prompt += f"Context from previous discussion:\\n---\n{context}\\n---\\n\\n"
        
        # Add previous response section if provided
        if pre_response:
            prompt += self._format_pre_response(pre_response) + "\\n\\n"
            
        prompt += self._format_consideration_prompt()
        prompt += self._format_preference_prompt()
        
        prompt += "\\nPlease provide your updated responses in excactly the following format:\\n" # Added 'updated'
        prompt += "Considerations:\\n"
        for consideration in self.survey.considerations:
            prompt += f"{consideration.id}: <rating>\\n"
            
        prompt += "\\nPreferences:\\n"
        for preference in self.survey.preferences:
            prompt += f"{preference.id}: <rank>\\n"
        
        prompt += "\\nDo not include any other text than the format above.\\n" # Added 'updated'
        return prompt
    
    def _validate_response_consistency(self, consideration_ratings: Dict[str, int], 
                                       preference_rankings: Dict[str, int], 
                                       instance_id: str) -> None:
        """Validate that the survey response is consistent according to the requirements.
        
        Checks:
        a) Preferences: Each rank appears only once and ranks are within valid range (1 to number of preferences)
        b) Considerations: Ratings are within Likert range (min to max from survey config) and not all the same number
        
        Raises ValueError if response is inconsistent.
        """
        # Get valid ranges from survey config
        scale_min = self.survey.response_format['consideration_scale']['min']
        scale_max = self.survey.response_format['consideration_scale']['max']
        num_preferences = len(self.survey.preferences)
        
        issues = []
        
        # Check preferences: a) Each rank appears only once, b) Ranks within valid range (1 to num_preferences)
        if len(preference_rankings) != num_preferences:
            issues.append(f"Expected {num_preferences} preference rankings, got {len(preference_rankings)}")
        else:
            # Check for duplicate ranks
            rank_values = list(preference_rankings.values())
            seen_ranks = set()
            duplicate_ranks = []
            out_of_range_ranks = []
            
            for pref_id, rank in preference_rankings.items():
                # Check range
                if rank < 1 or rank > num_preferences:
                    out_of_range_ranks.append(f"{pref_id}: rank {rank} (valid range: 1-{num_preferences})")
                
                # Check duplicates
                if rank in seen_ranks:
                    duplicate_ranks.append(f"{pref_id}: rank {rank} (duplicate)")
                seen_ranks.add(rank)
            
            if duplicate_ranks:
                issues.append(f"Duplicate preference ranks found: {', '.join(duplicate_ranks)}")
            if out_of_range_ranks:
                issues.append(f"Preference ranks out of range: {', '.join(out_of_range_ranks)}")
        
        # Check considerations: a) Ratings within Likert range, b) Not all the same number
        if len(consideration_ratings) != len(self.survey.considerations):
            issues.append(f"Expected {len(self.survey.considerations)} consideration ratings, got {len(consideration_ratings)}")
        else:
            out_of_range_ratings = []
            all_ratings = list(consideration_ratings.values())
            
            # Check range
            for cons_id, rating in consideration_ratings.items():
                if rating < scale_min or rating > scale_max:
                    out_of_range_ratings.append(f"{cons_id}: rating {rating} (valid range: {scale_min}-{scale_max})")
            
            if out_of_range_ratings:
                issues.append(f"Consideration ratings out of range: {', '.join(out_of_range_ratings)}")
            
            # Check if all ratings are the same
            if len(set(all_ratings)) == 1:
                issues.append(f"All consideration ratings are the same ({all_ratings[0]}), indicating non-serious response")
        
        # Raise ValueError if any issues found
        if issues:
            error_msg = f"Inconsistent survey response for {instance_id}: {'; '.join(issues)}"
            print(f"[CONSISTENCY CHECK FAILED] {error_msg}")
            raise ValueError(error_msg)
        
        print(f"[CONSISTENCY CHECK PASSED] Response for {instance_id} is consistent")
    
    def _parse_response(self, response_text: str, instance_id: str) -> SurveyResponse:
        """Parse the LLM response into a structured SurveyResponse object."""
        print(f"--- Parsing response for {instance_id} ---") # DEBUG
        print(f"Raw Text:\n{response_text}") # DEBUG - Show full raw text
        
        consideration_ratings = {}
        preference_rankings = {}
        
        expected_considerations = len(self.survey.considerations)
        expected_preferences = len(self.survey.preferences)
        
        # Allows whitespace, optional markdown like *, _, #, ` around ID/value
        # Assumes ID starts with C or P followed by digits
        line_regex = re.compile(r"^\s*[*_#`]*([CP]\d+)[*_#`]*\s*:\s*[*_#`]*(\d+)[*_#`]*\s*$") # Final attempt at correcting regex

        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            cleaned_line = line.strip()
            if not cleaned_line:
                continue
                
            # Check for section headers
            if cleaned_line.lower().startswith('considerations:'):
                current_section = 'considerations'
                print(f"  Found considerations section") # Debug
                continue
            elif cleaned_line.lower().startswith('preferences:'):
                current_section = 'preferences'
                print(f"  Found preferences section") # Debug
                continue
                
            # Attempt to parse using regex based on current section
            match = line_regex.match(cleaned_line)
            if match:
                item_id = match.group(1)
                item_value = match.group(2)
                try:
                    item_value_int = int(item_value)
                    if current_section == 'considerations' and item_id.startswith('C'):
                        consideration_ratings[item_id] = item_value_int
                    elif current_section == 'preferences' and item_id.startswith('P'):
                        preference_rankings[item_id] = item_value_int
                    else:
                        print(f"  [WARN] Parsed line '{line}' but ID '{item_id}' doesn't match section '{current_section}'") # Debug
                except ValueError:
                    print(f"  [WARN] Parsed line '{line}' but value '{item_value}' is not an integer.") # Debug
            elif current_section: # Only print skip warning if we are inside a section
                # Print skipped lines only if they aren't empty or section headers
                lower_cleaned = cleaned_line.lower()
                if not lower_cleaned.startswith('considerations:') and not lower_cleaned.startswith('preferences:'):
                    print(f"  Skipping line in {current_section}: {line}") # DEBUG - Now uncommented
        
        print(f"Parsed Considerations: {consideration_ratings}") # DEBUG
        print(f"Parsed Preferences: {preference_rankings}") # DEBUG
        
        # --- Assertion for successful parsing --- 
        if len(consideration_ratings) != expected_considerations or \
           len(preference_rankings) != expected_preferences:
            print(f"[ASSERTION FAILED] Parsing incomplete for {instance_id}. Expected C:{expected_considerations}, P:{expected_preferences}. Got C:{len(consideration_ratings)}, P:{len(preference_rankings)}")
            # Raise an error to potentially trigger retry logic later
            raise ValueError(f"Parsing failed for {instance_id}: Incomplete data extracted.")
        # --- End Assertion --- 
        
        # --- Consistency validation ---
        self._validate_response_consistency(consideration_ratings, preference_rankings, instance_id)
        # --- End Consistency validation ---
            
        print(f"--- End parsing for {instance_id} ---") # DEBUG
                    
        parsed_obj = SurveyResponse(
            instance_id=instance_id,
            consideration_ratings=consideration_ratings,
            preference_rankings=preference_rankings
        )
        return parsed_obj
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=6), retry=retry_if_exception_type(ValueError))
    async def administer_survey(self,
                              instance: LLMInstance,
                              context: Optional[str] = None,
                              pre_response: Optional[SurveyResponse] = None,
                              file_content_base64: Optional[str] = None,
                              file_media_type: Optional[str] = None
                              ) -> Tuple[str, Optional[SurveyResponse], str]:
        """Administer the survey to a single LLM instance. Returns raw response text, parsed response object, and prompt string. Retries up to 5 times on parsing failure."""
        print(f"Administering survey to {instance.instance_id} (Attempt: {self.administer_survey.retry.statistics.get('attempt_number', 1)})" ) # Log attempt number
        
        prompt_string = self._format_survey_prompt(context, pre_response)

        if file_content_base64 and not self.context_doc_warning_shown:
            print("[WARN] Context documents cannot be attached via OpenRouter chat completions. "
                  "Include any necessary excerpts directly in the survey prompt.")
            self.context_doc_warning_shown = True

        messages_payload: List[Dict[str, Any]] = [{"role": "user", "content": prompt_string}]


        # Determine tool_choice based on survey phase (pre vs post)
        if context is None: # Pre-survey
            run_tool_choice = "none" # Type: ignore
            print(f"  Survey phase detected: PRE. Setting tool_choice='none'")
        else: # Post-survey
            run_tool_choice = self.deliberation_tool_choice # Use configured setting
            print(f"  Survey phase detected: POST. Setting tool_choice='{run_tool_choice}' from config.")

        print(f"  Attempting to generate response for {instance.instance_id}...") # DEBUG
        raw_response = None # Initialize raw_response
        try:
            raw_response = await instance.generate(
                messages=messages_payload,
                tool_choice=run_tool_choice # Pass dynamic tool_choice
            )
            print(f"  Generation successful for {instance.instance_id}.") # DEBUG

            # --- ADDED CHECK FOR GENERATION ERRORS --- 
            if raw_response is None or raw_response.startswith("[Run Error:") or raw_response.startswith("[No Response]") or raw_response.startswith("[Run completed, but no response found]"):
                error_message = f"LLM generation failed or produced known error string for {instance.instance_id}: {raw_response}"
                print(f"  [ERROR] {error_message}")
                raise SurveyGenerationFailedError(error_message)
            # --- END ADDED CHECK ---

        except SurveyGenerationFailedError: # Explicitly re-raise if it's our custom error
            raise
        except Exception as gen_err: # Catch other generation errors
            print(f"  [ERROR] Generation FAILED for {instance.instance_id}: {gen_err}") # DEBUG
            # Wrap other generation errors in our custom exception type if not already one
            # This ensures tenacity doesn't retry on these if not ValueError
            raise SurveyGenerationFailedError(f"Generation failed for {instance.instance_id}: {gen_err}") from gen_err

        try:
            # Parsing happens here, potentially raising ValueError
            parsed_response = self._parse_response(raw_response, instance.instance_id)
            return raw_response, parsed_response, prompt_string
        except ValueError as e: # Catch the specific parsing error
            print(f"Parsing failed for {instance.instance_id}, will retry if attempts remain... Error: {e}")
            raise e # Re-raise the ValueError to trigger tenacity retry
        except Exception as e: # Catch other unexpected errors during parsing/generation
            print(f"Unexpected error administering survey to {instance.instance_id}: {str(e)}\\nRaw response (if available):\\n{raw_response}")
            return raw_response, None, prompt_string # Return None for parsed if non-ValueError occurs
    
    async def administer_survey_to_group(self,
                                       instances: List[LLMInstance],
                                       context: Optional[str] = None,
                                       pre_responses: Optional[Dict[str, SurveyResponse]] = None,
                                       file_content_base64: Optional[str] = None,
                                       file_media_type: Optional[str] = None
                                       ) -> Dict[str, Dict[str, Any]]:
        """Administer the survey to a group of LLM instances. Returns a dict mapping instance_id to {'raw': raw_response, 'parsed': parsed_response}.
        Retries failed instances up to 5 times to ensure all instances answer correctly."""
        print(f"\n--- Starting administer_survey_to_group (Context Provided: {context is not None}) ---") # DEBUG
        results = {}
        if not instances:
            print("  [WARN] No instances provided to administer_survey_to_group.")
            return results

        # Track which instances need retrying
        failed_instances = []
        max_retries = 5
        
        # First pass: try all instances IN PARALLEL
        async def survey_instance(instance, individual_pre_response):
            instance_id = instance.instance_id
            try:
                raw_response, parsed_response, prompt = await self.administer_survey(
                    instance,
                    context,
                    individual_pre_response,
                    file_content_base64,
                    file_media_type
                )
                return instance_id, {'raw': raw_response, 'parsed': parsed_response, 'prompt': prompt, 'error': None}
            except Exception as e:
                print(f"[ERROR] FAILED to get survey response from {instance_id}: {e}")
                return instance_id, {'raw': None, 'parsed': None, 'prompt': None, 'error': e, 'instance': instance, 'pre_response': individual_pre_response}
        
        # Create tasks for all instances
        tasks = []
        for instance in instances:
            instance_id = instance.instance_id
            individual_pre_response = pre_responses.get(instance_id) if pre_responses else None
            tasks.append(survey_instance(instance, individual_pre_response))
        
        # Run all survey calls in parallel
        print(f"  Running {len(tasks)} survey calls in parallel...")
        survey_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in survey_results:
            if isinstance(result, Exception):
                print(f"[ERROR] Survey task raised exception: {result}")
                continue
            instance_id, result_data = result
            if result_data.get('error') is not None or result_data.get('parsed') is None:
                failed_instances.append((result_data.get('instance'), result_data.get('pre_response')))
            else:
                results[instance_id] = {
                    'raw': result_data['raw'],
                    'parsed': result_data['parsed'],
                    'prompt': result_data['prompt']
                }
        
        # Retry failed instances up to max_retries times IN PARALLEL
        for retry_attempt in range(1, max_retries + 1):
            if not failed_instances:
                break  # All instances succeeded
            
            print(f"\n--- Retry attempt {retry_attempt}/{max_retries} for {len(failed_instances)} failed instance(s) (in parallel) ---")
            retry_list = failed_instances.copy()
            failed_instances = []
            
            # Create parallel retry tasks
            retry_tasks = []
            for instance, individual_pre_response in retry_list:
                instance_id = instance.instance_id
                retry_tasks.append(survey_instance(instance, individual_pre_response))
            
            # Run all retries in parallel
            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
            
            # Process retry results
            for result in retry_results:
                if isinstance(result, Exception):
                    print(f"[ERROR] Retry task raised exception: {result}")
                    continue
                instance_id, result_data = result
                if result_data.get('error') is not None or result_data.get('parsed') is None:
                    if retry_attempt < max_retries:
                        failed_instances.append((result_data.get('instance'), result_data.get('pre_response')))
                    else:
                        # Last attempt failed, store None
                        try:
                            error_prompt = self._format_survey_prompt(context, result_data.get('pre_response'))
                        except:
                            error_prompt = None
                        results[instance_id] = {'raw': result_data.get('raw'), 'parsed': None, 'prompt': error_prompt}
                else:
                    print(f"    [SUCCESS] Got valid parsed response for {instance_id} on retry {retry_attempt}")
                    results[instance_id] = {
                        'raw': result_data['raw'],
                        'parsed': result_data['parsed'],
                        'prompt': result_data['prompt']
                    }
        
        # Final validation: check that all instances have valid parsed responses
        failed_count = sum(1 for r in results.values() if r.get('parsed') is None)
        if failed_count > 0:
            failed_ids = [instance_id for instance_id, r in results.items() if r.get('parsed') is None]
            print(f"\n[WARNING] {failed_count} instance(s) failed to provide valid survey responses after {max_retries} retries:")
            for instance_id in failed_ids:
                print(f"  - {instance_id}")
        else:
            print(f"\n[SUCCESS] All {len(instances)} instance(s) provided valid survey responses")
        
        print(f"--- Finished administer_survey_to_group ---") # DEBUG
        return results 