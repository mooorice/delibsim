"""DRI Calculator based on pairwise correlations."""
# IC means intersubjective consistency

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
from collections import defaultdict

try:
    from ..survey.dri_survey import SurveyResponse, Consideration, Preference
except ImportError:
    print("Warning: Could not import SurveyResponse relative path.")
    class SurveyResponse:
        pass
    class Consideration: id: str
    class Preference: id: str


class DRICalculator:
    """Calculates DRI based on intermediate pairwise correlations."""

    def __init__(self):
        pass

    def _dri_calc(self, data: pd.DataFrame, v1_col: str, v2_col: str) -> float:
        """Core DRI calculation using pairwise correlation data."""
        if v1_col not in data.columns or v2_col not in data.columns or data.empty:
            return np.nan
        v1_data = data[v1_col]
        v2_data = data[v2_col]
        valid_idx = v1_data.notna() & v2_data.notna()
        if not valid_idx.any():
            return np.nan
        v1_data = v1_data[valid_idx]
        v2_data = v2_data[valid_idx]
        lambda_val = 1 - (np.sqrt(2) / 2)
        mean_abs_diff = np.mean(np.abs((v1_data - v2_data) / np.sqrt(2)))
        dri = 2 * (((1 - mean_abs_diff) - lambda_val) / (1 - lambda_val)) - 1
        return dri

    def _generate_ic_dataframe(self, parsed_responses: Dict[str, SurveyResponse]) -> pd.DataFrame:
        """Generates a DataFrame of pairwise correlations (intersubjective consistency)."""

        instance_ids = list(parsed_responses.keys())
        if len(instance_ids) < 2:
            print("Warning: Need at least 2 valid responses to generate IC DataFrame.")
            return pd.DataFrame()

        # Prepare DataFrames for Q (Considerations) and R (Preferences)
        q_data = {}
        r_data = {}
        valid_instance_ids = []
        num_q = 0
        num_r = 0

        for inst_id, resp in parsed_responses.items():
            if resp:
                q_ratings = resp.consideration_ratings
                r_rankings = resp.preference_rankings
                if q_ratings and r_rankings:
                    q_data[inst_id] = list(q_ratings.values())
                    r_data[inst_id] = list(r_rankings.values())
                    valid_instance_ids.append(inst_id)
                    if num_q == 0:
                        num_q = len(q_data[inst_id])
                    if num_r == 0:
                        num_r = len(r_data[inst_id])
                else:
                    print(f"Warning: Missing Q or R data for instance {inst_id}")
            else:
                print(f"Warning: No response object for instance {inst_id}")

        if len(valid_instance_ids) < 2:
            print("Warning: Less than 2 instances with valid Q and R data.")
            return pd.DataFrame()

        # Convert to DataFrames (Items as rows, Instances as columns)
        try:
            Q_df = pd.DataFrame(q_data, index=[f"C{i+1}" for i in range(num_q)])
            R_df = pd.DataFrame(r_data, index=[f"P{i+1}" for i in range(num_r)])
            Q_df = Q_df[valid_instance_ids]
            R_df = R_df[valid_instance_ids]
        except ValueError as e:
            print(f"Error creating Q/R DataFrames (likely inconsistent lengths): {e}")
            print("Q Data lengths:", {k: len(v) for k, v in q_data.items()})
            print("R Data lengths:", {k: len(v) for k, v in r_data.items()})
            return pd.DataFrame()

        # Calculate Spearman correlation matrices
        q_corr_matrix = Q_df.corr(method='spearman')
        r_corr_matrix = R_df.corr(method='spearman')

        # Generate unique pairs and extract correlations
        pairs = list(combinations(valid_instance_ids, 2))
        if not pairs:
            return pd.DataFrame()

        try:
            q_corrs = [q_corr_matrix.loc[p1, p2] for p1, p2 in pairs]
            r_corrs = [r_corr_matrix.loc[p1, p2] for p1, p2 in pairs]
        except KeyError as e:
            print(f"Error accessing correlations (KeyError: {e}). Check instance IDs.")
            print("Q Corr Matrix Columns:", q_corr_matrix.columns)
            print("R Corr Matrix Columns:", r_corr_matrix.columns)
            print("Pairs:", pairs)
            return pd.DataFrame()

        # Create the final IC DataFrame
        IC_df = pd.DataFrame({
            'P1': [p[0] for p in pairs],
            'P2': [p[1] for p in pairs],
            'Q_corr': q_corrs,
            'R_corr': r_corrs
        })
        IC_df['P_P'] = IC_df['P1'].astype(str) + '-' + IC_df['P2'].astype(str)

        return IC_df[['P_P', 'P1', 'P2', 'Q_corr', 'R_corr']]

    def calculate_dri_scores(
        self,
        parsed_responses: Dict[str, Optional[SurveyResponse]]
    ) -> Tuple[Dict[str, float], float, pd.DataFrame]:
        """Calculate individual and group DRI scores."""

        print("--- Calculating DRI Scores ---")

        # Filter out None responses before generating IC dataframe
        valid_parsed_responses = {k: v for k, v in parsed_responses.items() if v is not None}

        if len(valid_parsed_responses) < 2:
            print("Not enough valid parsed responses (< 2) to calculate DRI.")
            return {}, 0.0, pd.DataFrame()

        # Generate the IC DataFrame from valid responses
        ic_df = self._generate_ic_dataframe(valid_parsed_responses)

        if ic_df.empty:
            print("IC DataFrame generation failed or is empty. Cannot calculate DRI.")
            return {}, 0.0, pd.DataFrame()

        # Calculate Individual DRI Scores
        individual_scores = {}
        all_participants = list(valid_parsed_responses.keys())

        for p_id in all_participants:
            # Filter IC data for the current participant's pairs
            participant_ic_df = ic_df[(ic_df['P1'] == p_id) | (ic_df['P2'] == p_id)].copy()

            if participant_ic_df.empty:
                print(f"Warning: No correlation pairs found for participant {p_id} in IC DataFrame.")
                score = np.nan
            else:
                score = self._dri_calc(participant_ic_df, 'R_corr', 'Q_corr')

            individual_scores[p_id] = score if not np.isnan(score) else 0.0
            print(f"  Individual score for {p_id}: {individual_scores[p_id]:.3f}")

        # Calculate Group DRI Score using all pairwise correlations
        group_score = self._dri_calc(ic_df, 'R_corr', 'Q_corr')
        group_score = group_score if not np.isnan(group_score) else 0.0
        print(f"  Group score: {group_score:.3f}")

        print("--- Finished Calculating DRI Scores ---")
        return individual_scores, group_score, ic_df

    def analyze_dri_changes(
        self,
        pre_scores: Dict[str, float],
        post_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze changes in DRI scores between pre and post deliberation."""
        changes = {}
        for instance_id in pre_scores:
            if instance_id in post_scores:
                pre = pre_scores[instance_id]
                post = post_scores[instance_id]
                if pd.notna(pre) and pd.notna(post):
                    changes[instance_id] = post - pre
                else:
                    changes[instance_id] = np.nan

        valid_changes = {k: v for k, v in changes.items() if pd.notna(v)}

        if not valid_changes:
            return {
                'mean_change': 0.0,
                'std_change': 0.0,
                'individual_changes': changes
            }

        return {
            'mean_change': np.mean(list(valid_changes.values())),
            'std_change': np.std(list(valid_changes.values())),
            'individual_changes': changes
        }

    def _calculate_variances(self, responses_by_item: Dict[str, List[int]]) -> Dict[str, Optional[float]]:
        """Helper to calculate variance for each item's list of responses."""
        variances = {}
        for item_id, responses in responses_by_item.items():
            if len(responses) >= 2:
                variances[item_id] = np.var(responses)
            else:
                variances[item_id] = None
        return variances

    def analyze_response_variance(
        self,
        survey_responses: Dict[str, Optional[SurveyResponse]],
        survey_items: List[Any],
        response_type: str
    ) -> Dict[str, Optional[float]]:
        """Restructures responses and calculates variance for considerations or preferences."""
        print(f"--- Analyzing Response Variance ({response_type}) ---")
        responses_by_item: Dict[str, List[int]] = defaultdict(list)

        item_attr = f"{response_type}_ratings" if response_type == 'consideration' else f"{response_type}_rankings"

        valid_responses = 0
        for instance_id, response in survey_responses.items():
            if response and hasattr(response, item_attr):
                response_dict = getattr(response, item_attr)
                if response_dict:
                    valid_responses += 1
                    for item in survey_items:
                        item_id = item.id
                        value = response_dict.get(item_id)
                        if value is not None:
                            responses_by_item[item_id].append(value)

        print(f"  Found {valid_responses} valid response sets for variance analysis.")
        if not responses_by_item:
            print("  No responses found to group by item.")
            return {}

        variances = self._calculate_variances(responses_by_item)
        print(f"--- Finished Analyzing Response Variance ({response_type}) ---")
        return variances

    def find_highest_variance(self, variances: Dict[str, Optional[float]]) -> Tuple[Optional[str], Optional[float]]:
        """Finds the item ID and value with the highest variance."""
        valid_variances = {k: v for k, v in variances.items() if v is not None}

        if not valid_variances:
            return None, None

        highest_id = max(valid_variances, key=valid_variances.get)
        highest_var = valid_variances[highest_id]

        return highest_id, highest_var
