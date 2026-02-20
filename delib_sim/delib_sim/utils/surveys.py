import pandas as pd
from functools import lru_cache
from pathlib import Path
import csv

SURVEYS_PATH = "data/dri_surveys/surveys.xlsx" # Adjusted path
SURVEY_TOPICS_CSV = Path(__file__).resolve().parents[2] / "data" / "dri_surveys" / "survey_topics.csv"


def get_survey_names(file_path=SURVEYS_PATH, no_template=False):
    xls = pd.ExcelFile(file_path)
    survey_names = xls.sheet_names
    if no_template:
        try:
            survey_names.remove("template")
        except ValueError:
            pass # Ignore if template sheet doesn't exist
    return survey_names


def get_surveys_data(file_path=SURVEYS_PATH):
    xls = pd.ExcelFile(file_path)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = pd.read_excel(xls, sheet_name)
    return sheets


def sort_statements(statements, order):

    if len(statements) != len(order):
        raise ValueError("The lengths of 'statements' and 'order' must be equal.")

    # Combine the statements and order lists into a list of tuples
    statement_order_pairs = list(zip(statements, order))

    # Sort the pairs based on the order values
    sorted_statement_order_pairs = sorted(statement_order_pairs, key=lambda x: x[1])

    # Separate the sorted statements from their corresponding order values
    sorted_statements = [x[0] for x in sorted_statement_order_pairs]

    return sorted_statements


@lru_cache(maxsize=1)
def get_survey_topics() -> dict:
    """Load the mapping of survey_id -> deliberation_topic from survey_topics.csv."""
    if not SURVEY_TOPICS_CSV.exists():
        raise FileNotFoundError(
            f"Survey topics file not found at {SURVEY_TOPICS_CSV}. "
            "Please generate it (e.g., via analysis/surveys_topic_builder.ipynb)."
        )

    topics: dict[str, str] = {}
    with SURVEY_TOPICS_CSV.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            survey_id = (row.get("survey_id") or "").strip()
            topic = (row.get("deliberation_topic") or "").strip()
            if not survey_id or not topic:
                continue
            topics[survey_id] = topic

    if not topics:
        raise ValueError(
            f"No topics could be read from {SURVEY_TOPICS_CSV}. "
            "Ensure the file contains 'survey_id' and 'deliberation_topic' columns."
        )
    return topics


def get_survey_topic(survey_name: str) -> str:
    topics = get_survey_topics()
    try:
        return topics[survey_name]
    except KeyError as exc:
        available = ", ".join(sorted(topics.keys()))
        raise ValueError(
            f"No deliberation topic configured for survey '{survey_name}'. "
            f"Available surveys with topics: {available}"
        ) from exc


def get_policies_and_considerations(sheet):

    policies = sheet["policies"].dropna().tolist()
    considerations = sheet["considerations"].dropna().tolist()

    policies_order = sheet["policies_order"].dropna().tolist()
    policies_order = [int(i) for i in policies_order]
    considerations_order = sheet["considerations_order"].dropna().tolist()
    considerations_order = [int(i) for i in considerations_order]

    # sort policies and considerations based on
    policies = sort_statements(policies, policies_order)
    considerations = sort_statements(considerations, considerations_order)

    # read optional params
    scale_max = sheet["scale_max"].dropna().tolist()
    q_method = sheet["q-method"].dropna().tolist()

    # print(scale_max)
    # print(q_)

    # convert to scale_max to value -- default: 10
    if len(scale_max) > 0:
        scale_max = int(scale_max[0])
    else:
        scale_max = 10

    # convert to q_method to value -- default: False
    if len(q_method) > 0:
        try:
            # Handle potential boolean values or string representations
            q_method_val = sheet["q-method"].dropna().iloc[0]
            if isinstance(q_method_val, str):
                q_method = q_method_val.strip().upper() == 'TRUE'
            else:
                q_method = bool(q_method_val)
        except:
            q_method = False # Default if parsing fails
    else:
        q_method = False

    return policies, considerations, scale_max, q_method


def load_survey_config(survey_name: str, file_path=SURVEYS_PATH) -> dict:
    """Loads and processes data for a specific survey sheet."""
    try:
        xls = pd.ExcelFile(file_path)
        if survey_name not in xls.sheet_names:
            raise ValueError(f"Survey '{survey_name}' not found in {file_path}. Available surveys: {xls.sheet_names}")
        
        sheet_data = pd.read_excel(xls, survey_name)
        policies, considerations, scale_max, q_method = get_policies_and_considerations(sheet_data)
        topic = get_survey_topic(survey_name)

        # Convert to the format expected by DRISurvey Pydantic model
        survey_dict = {
            "issue_topic": topic,
            "considerations": [{"id": f"C{i+1}", "text": text} for i, text in enumerate(considerations)],
            "preferences": [{"id": f"P{i+1}", "text": text} for i, text in enumerate(policies)],
            "response_format": {
                "consideration_scale": {"min": 1, "max": scale_max},
                "preference_ranking": {"description": f"Rank from 1 (highest priority) to {len(policies)} (lowest priority)"},
                "q_method": q_method # Store q_method flag
            }
        }
        return survey_dict
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Survey file not found at {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading survey '{survey_name}' from {file_path}: {e}") 