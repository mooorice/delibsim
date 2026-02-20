from typing import List, Dict, Any, Optional, Union, Literal
import random
import numpy as np
import asyncio
from ..llm.manager import LLMInstance
from ..utils.config import GroupAssignmentStrategy, DeliberationProtocol
from ..survey.dri_survey import DRISurvey

class ConversationTurn:
    def __init__(self, instance_id: str, turn_number: int, response: str):
        self.instance_id = instance_id
        self.turn_number = turn_number
        self.response = response

class ConversationHistory:
    def __init__(self):
        self.turns: List[ConversationTurn] = []
        
    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
        
    def get_transcript(self) -> str:
        return "\n".join([
            f"Turn {turn.turn_number} ({turn.instance_id}): {turn.response}"
            for turn in self.turns
        ])

class DeliberationGroup:
    def __init__(self, group_id: str, instances: List[LLMInstance]):
        self.group_id = group_id
        self.instances = instances
        self.conversation_history = ConversationHistory()

class DeliberationEngine:
    """Engine for running multi-agent deliberation sessions."""

    def __init__(self,
                 group_size: int,
                 assignment_strategy: GroupAssignmentStrategy,
                 deliberation_tool_choice: Union[Literal["auto", "none", "required"], Dict[str, Any]],
                 survey: DRISurvey,
                 deliberation_prompt_type: str = "control_1"):
        self.group_size = group_size
        self.assignment_strategy = assignment_strategy
        self.deliberation_tool_choice = deliberation_tool_choice
        self.survey = survey
        self.deliberation_prompt_type = deliberation_prompt_type
        
    def _assign_groups_random(self, instances: List[LLMInstance]) -> List[DeliberationGroup]:
        """Assign instances to groups randomly."""
        random.shuffle(instances)
        groups = []
        for i in range(0, len(instances), self.group_size):
            group_instances = instances[i:i + self.group_size]
            if len(group_instances) == self.group_size:  # Only create complete groups
                groups.append(DeliberationGroup(f"group-{i//self.group_size}", group_instances))
        return groups
    
    def _assign_groups_by_model(self, instances: List[LLMInstance]) -> List[DeliberationGroup]:
        """Assign instances to groups based on their model type, creating mixed groups."""
        # Group instances by model type
        model_groups: Dict[str, List[LLMInstance]] = {}
        for instance in instances:
            model_name = instance.setup.model_name
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(instance)
        # Randomize the order within each model bucket to avoid deterministic ordering
        for lst in model_groups.values():
            random.shuffle(lst)
            
        # Create mixed groups by cycling through models
        groups = []
        group_count = 0
        
        # Continue until we can't form complete groups
        while True:
            group_instances = []
            model_names = list(model_groups.keys())
            # Randomize the order of models for each group creation
            random.shuffle(model_names)
            
            # Try to add one instance from each model to the group
            for model_name in model_names:
                if model_groups[model_name] and len(group_instances) < self.group_size:
                    group_instances.append(model_groups[model_name].pop(0))
            
            # If we have a complete group, create it
            if len(group_instances) == self.group_size:
                # Randomize the final order of agents within the group to avoid fixed speaking starts
                random.shuffle(group_instances)
                groups.append(DeliberationGroup(f"group-{group_count}", group_instances))
                group_count += 1
            else:
                # If we can't form a complete group, break
                break
                
        return groups
    
    def _assign_groups_individual_model(self, instances: List[LLMInstance]) -> List[DeliberationGroup]:
        """Assign instances to groups where all agents in each group use the same model."""
        # Group instances by model type
        model_groups: Dict[str, List[LLMInstance]] = {}
        for instance in instances:
            model_name = instance.setup.model_name
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(instance)
        
        # Create groups where all agents use the same model
        groups = []
        group_count = 0
        
        for model_name, model_instances in model_groups.items():
            # Randomize the order of instances within the model before grouping
            random.shuffle(model_instances)
            # Create groups of the specified size from this model
            for i in range(0, len(model_instances), self.group_size):
                group_instances = model_instances[i:i + self.group_size]
                if len(group_instances) == self.group_size:  # Only create complete groups
                    # Optional: randomize order inside group (keeps same-model composition)
                    random.shuffle(group_instances)
                    groups.append(DeliberationGroup(f"group-{group_count}-{model_name}", group_instances))
                    group_count += 1
        
        return groups
    
    def assign_groups(self, instances: List[LLMInstance]) -> List[DeliberationGroup]:
        """Assign instances to deliberation groups based on the configured strategy."""
        groups: List[DeliberationGroup] = [] # Local variable
        if self.assignment_strategy == GroupAssignmentStrategy.RANDOM:
            groups = self._assign_groups_random(instances)
        elif self.assignment_strategy == GroupAssignmentStrategy.BY_MODEL:
            groups = self._assign_groups_by_model(instances)
        elif self.assignment_strategy == GroupAssignmentStrategy.MIXED:
            groups = self._assign_groups_by_model(instances)  # Same logic as BY_MODEL
        elif self.assignment_strategy == GroupAssignmentStrategy.INDIVIDUAL_MODEL:
            groups = self._assign_groups_individual_model(instances)
        else:
            raise ValueError(f"Unsupported group assignment strategy: {self.assignment_strategy}")
        # Return the created groups instead of storing them in self.groups
        return groups
    
    def _get_item_text(self, item_id: str) -> Optional[str]:
        """Helper to get the text of a consideration or preference by ID."""
        for item in self.survey.considerations:
            if item.id == item_id:
                return item.text
        for item in self.survey.preferences:
             if item.id == item_id:
                return item.text
        return None

    def _format_guided_prompt_section(self,
                                       consideration_variances: Optional[Dict[str, float]] = None,
                                       preference_variances: Optional[Dict[str, float]] = None,
                                       top_n_considerations: int = 10,
                                       top_n_preferences: int = 3
                                       ) -> str:
        """Formats the guidance text based on high-variance items."""
        if not consideration_variances and not preference_variances:
            return ""

        guidance_text = "\n\n--- Focus Points (High Variance Items) ---\n"
        guidance_text += "Based on initial responses, there was notable variance on the following items. Please pay special attention to resolving disagreements or understanding different perspectives on these during your discussion:\n"
        top_c = []
        top_p = []
        # Process Considerations
        if consideration_variances:
            valid_c_vars = {k: v for k, v in consideration_variances.items() if v is not None and not np.isnan(v)}
            sorted_c = sorted(valid_c_vars.items(), key=lambda item: item[1], reverse=True)
            top_c = sorted_c[:top_n_considerations]
            if top_c:
                guidance_text += "\nConsiderations (Top {} by Variance):\n".format(len(top_c))
                for item_id, var in top_c:
                    text = self._get_item_text(item_id) or f"ID: {item_id}"
                    guidance_text += f"- {item_id}: {text} (Variance: {var:.3f})\n"

        # Process Preferences
        if preference_variances:
            valid_p_vars = {k: v for k, v in preference_variances.items() if v is not None and not np.isnan(v)}
            sorted_p = sorted(valid_p_vars.items(), key=lambda item: item[1], reverse=True)
            top_p = sorted_p[:top_n_preferences]
            if top_p:
                guidance_text += "\nPreferences (Top {} by Variance):\n".format(len(top_p))
                for item_id, var in top_p:
                    text = self._get_item_text(item_id) or f"ID: {item_id}"
                    guidance_text += f"- {item_id}: {text} (Variance: {var:.3f})\n"

        guidance_text += "------------------------------------------\n\n"
        # Only return guidance if we actually found items
        if len(top_c) > 0 or len(top_p) > 0:
            return guidance_text
        else:
            return "" # Return empty if no valid variances found

    async def _run_deliberation_loop(
        self,
        group: DeliberationGroup,
        topic: str,
        num_turns: int,
        display_conversation: bool = False,
        protocol: DeliberationProtocol = DeliberationProtocol.FREE_FLOW,
        consideration_variances: Optional[Dict[str, float]] = None,
        preference_variances: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run turn-based deliberation for a group.

        Uses round-robin speaking order where all agents speak equally often.
        Each round has a randomized order of all agents for fair participation.

        Returns dict with "main", "with_reasoning", and "turn_logs" keys.
        """
        # Generate guidance text if using GUIDED protocol
        guidance_section = ""
        if protocol == DeliberationProtocol.GUIDED:
            guidance_section = self._format_guided_prompt_section(
                consideration_variances, preference_variances
            )

        transcript_main: List[str] = []
        transcript_with_reasoning: List[str] = []
        full_turn_logs: List[Dict[str, Any]] = []

        # Create round-robin speaking order with randomized rounds
        group_size = len(group.instances)
        num_rounds = num_turns // group_size
        remaining_turns = num_turns % group_size

        speaking_order = []
        for _ in range(num_rounds):
            round_order = group.instances.copy()
            random.shuffle(round_order)
            speaking_order.extend(round_order)

        for _ in range(remaining_turns):
            speaking_order.append(random.choice(group.instances))

        # Initial turn
        starter_instance = speaking_order[0]
        initial_prompt = self._format_deliberation_prompt(topic, None, is_initial=True)
        if guidance_section:
            initial_prompt += guidance_section

        initial_response = await starter_instance.generate(
            messages=[{"role": "user", "content": initial_prompt}],
            temperature=None,
            tool_choice=self.deliberation_tool_choice
        )

        group.conversation_history.add_turn(
            ConversationTurn(starter_instance.instance_id, 0, initial_response)
        )
        transcript_main.append(f"Turn 0 ({starter_instance.instance_id}): {initial_response}")
        transcript_with_reasoning.append(f"Turn 0 ({starter_instance.instance_id}): {initial_response}")
        full_turn_logs.append({
            "turn": 0,
            "instance_id": starter_instance.instance_id,
            "prompt": initial_prompt,
            "response": initial_response
        })

        if display_conversation:
            print(f"\nGroup {group.group_id} - Turn 0:")
            print(f"{starter_instance.instance_id}: {initial_response}\n")

        # Subsequent turns
        for turn in range(1, num_turns):
            current_transcript = "\n".join(transcript_main)
            next_instance = speaking_order[turn]

            prompt = self._format_deliberation_prompt(topic, current_transcript, is_initial=False)
            if guidance_section:
                prompt += guidance_section

            response = await next_instance.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=None,
                tool_choice=self.deliberation_tool_choice
            )

            group.conversation_history.add_turn(
                ConversationTurn(next_instance.instance_id, turn, response)
            )
            transcript_main.append(f"Turn {turn} ({next_instance.instance_id}): {response}")
            transcript_with_reasoning.append(f"Turn {turn} ({next_instance.instance_id}): {response}")
            full_turn_logs.append({
                "turn": turn,
                "instance_id": next_instance.instance_id,
                "prompt": prompt,
                "response": response
            })

            if display_conversation:
                print(f"\nGroup {group.group_id} - Turn {turn}:")
                print(f"{next_instance.instance_id}: {response}\n")

        return {
            "main": "\n".join(transcript_main),
            "with_reasoning": "\n".join(transcript_with_reasoning),
            "turn_logs": full_turn_logs
        }

    def _format_deliberation_prompt(self, topic: str, transcript: Optional[str], is_initial: bool) -> str:
        """Format the deliberation prompt based on prompt type.

        Args:
            topic: The deliberation topic
            transcript: Current conversation transcript (None for initial turn)
            is_initial: Whether this is the first turn

        Returns:
            Formatted prompt string
        """
        if self.deliberation_prompt_type == "treatment":
            # Normative treatment prompt with deliberative norms
            base = (
                f"You're taking part in a citizen's assembly on the topic: {topic}\n"
                f"Deliberate with the other participants.\n"
                f"Be respectful, reasoned, and authentic. Do not use force in your language.\n"
                f"Express your viewpoints and give reasons. Orient your arguments and viewpoints towards the common good.\n"
                f"Your goal is to develop a shared understanding of the topic and to find the best solutions with the other participants.\n"
                f"Consider others' perspectives and engage with others' arguments. Do not seek premature consensus or try to average positions.\n"
            )
            if not is_initial and transcript:
                base += f"Current Conversation:\n{transcript}\n\n"
            return base
        else:
            # Basic/control prompt - minimal instructions
            base = (
                f"You're taking part in a citizen's assembly on the topic: {topic}\n"
                f"Deliberate with the other participants."
            )
            if not is_initial and transcript:
                base += f"\nCurrent Conversation:\n{transcript}\n\n"
            return base

    async def run_deliberation(
        self,
        groups: List[DeliberationGroup],
        topic: str,
        num_turns: int,
        protocol: DeliberationProtocol,
        display_conversation: bool = False,
        consideration_variances: Optional[Dict[str, Optional[float]]] = None,
        preference_variances: Optional[Dict[str, Optional[float]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run deliberation sessions for all groups using the specified protocol.

        Args:
            groups: List of deliberation groups
            topic: The deliberation topic
            num_turns: Number of turns per deliberation
            protocol: FREE_FLOW or GUIDED
            display_conversation: Whether to print conversation to console
            consideration_variances: Pre-deliberation consideration variances (for GUIDED)
            preference_variances: Pre-deliberation preference variances (for GUIDED)

        Returns:
            Dict mapping group_id to transcript dict with "main", "with_reasoning", "turn_logs"
        """
        transcripts: Dict[str, Dict[str, Any]] = {}
        print(f"\nStarting deliberation on topic: {topic} (Protocol: {protocol.value})")

        MAX_RETRIES = 3
        RETRY_BASE_DELAY = 30

        async def run_group_deliberation(group: DeliberationGroup):
            for attempt in range(MAX_RETRIES):
                try:
                    transcript_data = await self._run_deliberation_loop(
                        group, topic, num_turns, display_conversation,
                        protocol, consideration_variances, preference_variances
                    )
                    return group.group_id, transcript_data, None
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = any(x in error_str for x in ['ratelimit', 'rate_limit', 'rate limit', 'retryerror'])

                    if is_rate_limit and attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        print(f"  [RETRY] Rate limit for {group.group_id}, waiting {delay}s...")
                        await asyncio.sleep(delay)
                        group.conversation_history = ConversationHistory()
                        continue
                    else:
                        print(f"  [ERROR] Deliberation failed for {group.group_id}: {e}")
                        return group.group_id, None, e

            return group.group_id, None, Exception("Max retries exceeded")

        tasks = [run_group_deliberation(group) for group in groups]
        print(f"  Running {len(tasks)} deliberation groups in parallel...")

        group_results = await asyncio.gather(*tasks, return_exceptions=True)

        failed_groups = []
        for result in group_results:
            if isinstance(result, Exception):
                failed_groups.append(str(result))
                continue
            group_id, transcript_data, error = result
            if transcript_data is None:
                failed_groups.append(group_id)
                continue
            transcripts[group_id] = transcript_data

        if failed_groups:
            print(f"  [WARN] {len(failed_groups)} group(s) failed: {failed_groups}")

        return transcripts