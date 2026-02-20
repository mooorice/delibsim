import asyncio
import json
import random
from typing import List, Dict, Tuple, Optional, Any

from ..llm.manager import LLMInstance
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class DRISurveyManager:
    """Manages the administration of DRI surveys to participants."""
    
    # ... existing code ...

    async def administer_survey(
        self, 
        instance: LLMInstance,
        items: List[dict], 
        participant_id: str,
        context: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Administers the DRI survey to a single participant using an LLM instance."""
        max_retries = 3
        retry_delay = 1 # seconds
        
        for attempt in range(max_retries):
            try:
                prompt_string = self._build_survey_prompt(items, participant_id, context)
                # print(f"--- Sending Survey Prompt to {participant_id} ---")
                # print(prompt_string)
                # print(f"--- End Survey Prompt ---")

                # Wrap the prompt string in the standard message format
                messages_payload = [{"role": "user", "content": prompt_string}]

                response_text = await instance.generate(
                    messages=messages_payload, # Use the wrapped payload
                    temperature=temperature
                )
                # print(f"--- Raw Survey Response from {participant_id} ---")
                # print(response_text)
                # print(f"--- End Raw Survey Response ---")
                
                parsed_response, reasoning = self._parse_survey_response(response_text, items)
                return parsed_response, reasoning

            except Exception as e:
                logger.error(f"Error administering survey to {participant_id} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt)) # Exponential backoff
                else:
                    logger.error(f"Failed to administer survey to {participant_id} after {max_retries} attempts.")
                    # Return empty dicts or raise exception?
                    return {}, {"error": f"Failed after {max_retries} attempts: {e}"} 

    async def administer_survey_to_group(
        self, 
        instances: Dict[str, LLMInstance], 
        items: List[dict], 
        participant_ids: List[str], 
        context: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Tuple[Dict[str, int], Dict[str, Any]]]:
        """Administers the DRI survey concurrently to a group of participants."""
        tasks = []
        for participant_id in participant_ids:
            if participant_id not in instances:
                logger.warning(f"No LLM instance found for participant {participant_id}. Skipping survey.")
                continue
            instance = instances[participant_id]
            tasks.append(
                self.administer_survey(
                    instance=instance,
                    items=items,
                    participant_id=participant_id,
                    context=context,
                    temperature=temperature
                )
            )
        
        results = await asyncio.gather(*tasks)
        
        # Map results back to participant IDs
        survey_results = {}
        result_idx = 0
        for participant_id in participant_ids:
            if participant_id in instances: # Only include participants who had an instance
                survey_results[participant_id] = results[result_idx]
                result_idx += 1
            else:
                survey_results[participant_id] = ({}, {"error": "Skipped due to missing LLM instance"}) # Indicate skipped
                
        return survey_results

    # ... existing code ... 