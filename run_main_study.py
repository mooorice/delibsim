"""Main study runner for DelibSim multi-agent deliberation experiments.

Replicates the experimental design for the DelibSim paper:
- 11 model configurations (9 single-model + 2 mixed-model)
- 12 deliberation topics
- 3 conditions (no_treatment, basic, normative)
- 5 repetitions per configuration
"""

from __future__ import annotations

import asyncio
import copy
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from dotenv import load_dotenv

# Ensure the project root is on the import path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Load environment variables (API keys) from .env if present
load_dotenv(ROOT_DIR / ".env")

from delib_sim.delib_sim.run import SimulationRunner
from delib_sim.delib_sim.survey.dri_survey import DRISurvey
from delib_sim.delib_sim.utils.config import SimulationConfig
from delib_sim.delib_sim.utils.surveys import get_survey_topic, load_survey_config

# ---------------------------------------------------------------------------
# Study Parameters
# ---------------------------------------------------------------------------

BASE_CONFIG_PATH = ROOT_DIR / "delib_sim" / "config" / "simulation_config.yaml"
OUTPUT_ROOT = ROOT_DIR / "delib_sim" / "data" / "simulation_results" / "test_study" # Remove test_study for final run

RUNS_PER_CONDITION = 5
GROUP_SIZE = 5
TOTAL_TURNS = 10  # 2 rounds per agent (5 agents × 2 = 10 turns)
GLOBAL_TEMPERATURE = 0.0
MAX_PARALLEL_RUNS = 1 # Increase this for faster inference

# All 12 deliberation topics from surveys.xlsx
TOPICS: List[str] = [
    "acp",
    "auscj",
    "bep",
    "biobanking_wa",
    "ccps",
    "energy_futures",
    "fnqcj",
    "forestera",
    "fremantle",
    "swiss_health",
    "uppsala_speaks",
    "zukunft",
]

# Three experimental conditions:
# - control_1 (No Treatment): Pre/post surveys only, no deliberation
# - control_2 (Basic Treatment): Deliberation with minimal prompt
# - treatment (Normative Treatment): Deliberation with deliberative norms prompt
CONDITIONS: List[str] = ["treatment", "control_1", "control_2"]

# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------
# The study uses 11 model configurations:
# - 9 single-model groups (all 5 agents use the same model)
# - 2 mixed-model groups (agents from different model families)

# Single-model configurations
SINGLE_MODEL_CONFIGS = [
    # GPT-5.1 (Reasoning and No-Reasoning variants)
    {"key": "openai/gpt-5.1-reasoning", "label": "GPT-5.1-R", "type": "single"},
    {"key": "openai/gpt-5.1-no-reasoning", "label": "GPT-5.1-NR", "type": "single"},
    # Gemini-3-Pro-Preview (Reasoning and No-Reasoning variants)
    {"key": "google/gemini-3-pro-preview-03-25-reasoning", "label": "Gemini-3-Pro-R", "type": "single"},
    {"key": "google/gemini-3-pro-preview-03-25-no-reasoning", "label": "Gemini-3-Pro-NR", "type": "single"},
    # DeepSeek-V3.2-Exp (Reasoning and No-Reasoning variants)
    {"key": "deepseek/deepseek-v3.2-exp-reasoning", "label": "DeepSeek-V3.2-R", "type": "single"},
    {"key": "deepseek/deepseek-v3.2-exp-no-reasoning", "label": "DeepSeek-V3.2-NR", "type": "single"},
    # Kimi-K2-Thinking (Reasoning and No-Reasoning variants)
    {"key": "moonshotai/kimi-k2-thinking-reasoning", "label": "Kimi-K2-R", "type": "single"},
    {"key": "moonshotai/kimi-k2-thinking-no-reasoning", "label": "Kimi-K2-NR", "type": "single"},
    # Claude Opus 4.5 (No-Reasoning only)
    {"key": "anthropic/claude-opus-4.5", "label": "Claude-Opus-4.5", "type": "single"},
]

# Mixed-model configurations (one agent from each model family)
MIXED_MODEL_CONFIGS = [
    {
        "key": "mixed-reasoning",
        "label": "Mixed-R",
        "type": "mixed",
        "models": [
            "openai/gpt-5.1-reasoning",
            "google/gemini-3-pro-preview-03-25-reasoning",
            "deepseek/deepseek-v3.2-exp-reasoning",
            "moonshotai/kimi-k2-thinking-reasoning",
            "anthropic/claude-opus-4.5",  
        ],
    },
    {
        "key": "mixed-no-reasoning",
        "label": "Mixed-NR",
        "type": "mixed",
        "models": [
            "openai/gpt-5.1-no-reasoning",
            "google/gemini-3-pro-preview-03-25-no-reasoning",
            "deepseek/deepseek-v3.2-exp-no-reasoning",
            "moonshotai/kimi-k2-thinking-no-reasoning",
            "anthropic/claude-opus-4.5",
        ],
    },
]

# Combined list of all model configurations
MODEL_CONFIG = SINGLE_MODEL_CONFIGS + MIXED_MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_base_config() -> Dict[str, Any]:
    """Load the base configuration from YAML file."""
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Base configuration not found at {BASE_CONFIG_PATH}. "
            "Create it before running the study."
        )
    with BASE_CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def configure_llm_setup(
    base_llm_setup: Iterable[Dict[str, Any]],
    model_config: Dict[str, Any],
    group_size: int,
) -> List[Dict[str, Any]]:
    """Configure LLM setup for a single-model or mixed-model scenario."""
    llm_setup = copy.deepcopy(list(base_llm_setup))
    if not llm_setup:
        raise ValueError("The base configuration must define at least one LLM.")

    if model_config["type"] == "mixed":
        # Mixed model: enable one agent from each specified model
        target_models = model_config["models"]
        configured_entries = []

        for entry in llm_setup:
            if entry["model_name"] in target_models:
                entry["enabled"] = True
                entry["count"] = 1
                configured_entries.append(entry)
            else:
                entry["enabled"] = False
                entry["count"] = 0

        if len(configured_entries) != group_size:
            available = [e["model_name"] for e in configured_entries]
            raise ValueError(
                f"Mixed model config requires {group_size} models, "
                f"but found {len(configured_entries)}: {available}"
            )
        return configured_entries

    # Single model: enable only the specified model with full group size
    scenario_key = model_config["key"]
    matched = False
    configured_entries = []

    for entry in llm_setup:
        if entry["model_name"] == scenario_key:
            entry["enabled"] = True
            entry["count"] = group_size
            matched = True
            configured_entries.append(entry)
        else:
            entry["enabled"] = False
            entry["count"] = 0

    if not matched:
        available = ", ".join(entry["model_name"] for entry in llm_setup)
        raise ValueError(
            f"Model '{scenario_key}' not found in base llm_setup. "
            f"Available models: {available}"
        )
    return configured_entries


def build_run_config(
    base_config: Dict[str, Any],
    llm_setup: List[Dict[str, Any]],
    *,
    model_config: Dict[str, Any],
    condition: str,
    survey_name: str,
    random_seed: int,
) -> Dict[str, Any]:
    """Build configuration for a single simulation run."""
    config = copy.deepcopy(base_config)
    config["llm_setup"] = llm_setup
    config["group_size"] = GROUP_SIZE
    config["num_deliberation_turns"] = TOTAL_TURNS
    config["num_repetitions"] = RUNS_PER_CONDITION
    config["global_temperature"] = GLOBAL_TEMPERATURE
    config["survey_name"] = survey_name

    # Configure condition-specific settings:
    # - control_1 (No Treatment): No deliberation, surveys only
    # - control_2 (Basic Treatment): Deliberation with minimal prompt
    # - treatment (Normative Treatment): Deliberation with normative prompt
    config["dri_conversation_enabled"] = condition in ["treatment", "control_2"]
    config["assess_dqi"] = condition == "treatment"
    config["deliberation_prompt_type"] = condition

    # Group assignment: mixed for multi-model, individual_model for single-model
    config["group_assignment_strategy"] = (
        "mixed" if model_config["type"] == "mixed" else "individual_model"
    )

    config["deliberation_topic"] = None  # Inferred from survey metadata
    config["random_seed"] = random_seed
    config["parallelize_repetitions"] = True
    config["max_parallel_repetitions"] = None
    config["display_conversation"] = True

    return config


# ---------------------------------------------------------------------------
# Study Runner
# ---------------------------------------------------------------------------

async def run_single_configuration(
    run_index: int,
    total_configurations: int,
    model_config: Dict[str, Any],
    condition: str,
    topic_name: str,
    topic_question: str,
    base_config: Dict[str, Any],
    base_llm_setup: List[Dict[str, Any]],
    survey_data: Dict[str, Dict[str, Any]],
    output_root: Path,
    base_seed: int,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """Run a single configuration (model × condition × topic)."""
    async with semaphore:
        run_label = f"{run_index}/{total_configurations}"
        print(
            f"\n--- Configuration {run_label} (STARTING) ---\n"
            f"Model: {model_config['label']}\n"
            f"Condition: {condition}\n"
            f"Survey: {topic_name} ({topic_question})"
        )

        try:
            llm_setup = configure_llm_setup(base_llm_setup, model_config, GROUP_SIZE)
            run_config_dict = build_run_config(
                base_config,
                llm_setup,
                model_config=model_config,
                condition=condition,
                survey_name=topic_name,
                random_seed=base_seed + run_index,
            )

            run_config = SimulationConfig(**run_config_dict)
            survey = DRISurvey(**copy.deepcopy(survey_data[topic_name]))

            # Create output directory with sanitized model key
            model_key_safe = model_config["key"].replace("/", "_")
            run_output_dir = (
                output_root
                / f"model_{model_key_safe}_condition_{condition}_topic_{topic_name}"
            )
            run_output_dir.mkdir(parents=True, exist_ok=True)

            # Save configuration used for this run
            config_log_path = run_output_dir / "config_used.yaml"
            with config_log_path.open("w") as f_cfg:
                config_dict = run_config.model_dump(mode="json")
                yaml.safe_dump(config_dict, f_cfg, default_flow_style=False)

            runner = SimulationRunner(run_config, survey)
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            run_results = await runner.run(str(run_output_dir))

            if not run_results:
                print(f"✗ Configuration {run_label} returned no results.")
                return None

            result = {
                "model_key": model_config["key"],
                "model_label": model_config["label"],
                "model_type": model_config["type"],
                "condition": condition,
                "topic": topic_name,
                "deliberation_topic": topic_question,
                "config": run_config.model_dump(mode="json"),
                "results": run_results,
                "output_dir": str(run_output_dir),
                "run_timestamp": run_timestamp,
            }
            print(f"✓ Configuration {run_label} completed successfully.")
            return result

        except Exception as exc:
            print(f"✗ Configuration {run_label} failed with error: {exc}")
            import traceback
            print(traceback.format_exc())
            return None


async def run_main_study() -> List[Dict[str, Any]]:
    """Run the full main study experiment."""
    base_config = load_base_config()
    base_llm_setup = base_config.get("llm_setup", [])

    study_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = OUTPUT_ROOT / study_run_id
    output_root.mkdir(parents=True, exist_ok=True)

    total_configurations = len(MODEL_CONFIG) * len(CONDITIONS) * len(TOPICS)
    total_iterations = total_configurations * RUNS_PER_CONDITION

    print("=" * 80)
    print("DelibSim Main Study")
    print("=" * 80)
    print(f"Models: {[m['label'] for m in MODEL_CONFIG]}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Topics: {TOPICS}")
    print(f"Configurations: {total_configurations}")
    print(f"Total simulation iterations: {total_iterations}")
    print(f"Output directory: {output_root}")
    print("=" * 80)

    # Load survey data for all topics
    survey_data: Dict[str, Dict[str, Any]] = {}
    survey_path = ROOT_DIR / "delib_sim" / "data" / "dri_surveys" / "surveys.xlsx"
    for survey_name in TOPICS:
        survey_data[survey_name] = load_survey_config(survey_name, file_path=str(survey_path))

    all_results: List[Dict[str, Any]] = []
    base_seed = base_config.get("random_seed", 42) or 42

    # Create semaphore to limit concurrent runs
    semaphore = asyncio.Semaphore(MAX_PARALLEL_RUNS)
    print(f"\nParallelization: Max {MAX_PARALLEL_RUNS} concurrent configurations")

    # Build list of all configuration tasks
    tasks = []
    run_index = 0
    for topic_name in TOPICS:
        topic_question = get_survey_topic(topic_name)
        for model_config in MODEL_CONFIG:
            for condition in CONDITIONS:
                run_index += 1
                tasks.append(
                    run_single_configuration(
                        run_index=run_index,
                        total_configurations=total_configurations,
                        model_config=model_config,
                        condition=condition,
                        topic_name=topic_name,
                        topic_question=topic_question,
                        base_config=base_config,
                        base_llm_setup=base_llm_setup,
                        survey_data=survey_data,
                        output_root=output_root,
                        base_seed=base_seed,
                        semaphore=semaphore,
                    )
                )

    # Run all configurations in parallel
    print(f"\nStarting {len(tasks)} configurations in parallel...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Track errors and failures
    failed_configs = []
    error_configs = []
    incomplete_configs = []

    # Process results
    for i, result in enumerate(results):
        config_idx = i
        topic_idx = config_idx // (len(MODEL_CONFIG) * len(CONDITIONS))
        remainder = config_idx % (len(MODEL_CONFIG) * len(CONDITIONS))
        model_idx = remainder // len(CONDITIONS)
        condition_idx = remainder % len(CONDITIONS)

        config_info = {
            "index": i + 1,
            "topic": TOPICS[topic_idx] if topic_idx < len(TOPICS) else "unknown",
            "model": MODEL_CONFIG[model_idx]["label"] if model_idx < len(MODEL_CONFIG) else "unknown",
            "condition": CONDITIONS[condition_idx] if condition_idx < len(CONDITIONS) else "unknown",
        }

        if isinstance(result, Exception):
            print(f"✗ Configuration {i+1} raised exception: {result}")
            error_configs.append({**config_info, "error": str(result)})
        elif result is None:
            print(f"✗ Configuration {i+1} returned None")
            failed_configs.append({**config_info, "error": "Returned None"})
        elif isinstance(result, dict):
            if result.get("results", {}).get("error"):
                error_msg = result["results"]["error"]
                print(f"✗ Configuration {i+1} reported error: {error_msg}")
                error_configs.append({**config_info, "error": error_msg})
            else:
                detailed_results = result.get("results", {}).get("detailed_results", [])
                expected_runs = RUNS_PER_CONDITION
                actual_runs = len(detailed_results)

                if actual_runs < expected_runs:
                    incomplete_configs.append({
                        **config_info,
                        "expected": expected_runs,
                        "actual": actual_runs,
                        "model_key": result.get("model_key", "unknown"),
                    })
                    print(f"⚠ Configuration {i+1} incomplete: {actual_runs}/{expected_runs} runs")

                all_results.append(result)
        else:
            print(f"✗ Configuration {i+1} returned unexpected type: {type(result)}")
            failed_configs.append({**config_info, "error": f"Unexpected type: {type(result)}"})

    # Print final summary
    print("\n" + "=" * 80)
    print("MAIN STUDY COMPLETED")
    print("=" * 80)
    print(f"\nSuccessful configurations: {len(all_results)}/{total_configurations}")

    if error_configs:
        print(f"\n❌ ERRORS ({len(error_configs)} configurations):")
        print("-" * 60)
        for cfg in error_configs:
            print(f"  • [{cfg['index']}] {cfg['model']} / {cfg['condition']} / {cfg['topic']}")
            print(f"    Error: {cfg['error'][:100]}...")

    if failed_configs:
        print(f"\n❌ FAILED ({len(failed_configs)} configurations):")
        print("-" * 60)
        for cfg in failed_configs:
            print(f"  • [{cfg['index']}] {cfg['model']} / {cfg['condition']} / {cfg['topic']}")
            print(f"    Reason: {cfg['error']}")

    if incomplete_configs:
        print(f"\n⚠️  INCOMPLETE ({len(incomplete_configs)} configurations):")
        print("-" * 60)
        for cfg in incomplete_configs:
            print(f"  • [{cfg['index']}] {cfg['model']} / {cfg['condition']} / {cfg['topic']}")
            print(f"    Runs: {cfg['actual']}/{cfg['expected']} completed")

    total_errors = len(error_configs) + len(failed_configs)
    total_incomplete = len(incomplete_configs)
    total_successful = len(all_results)

    print("\n" + "-" * 60)
    print("SUMMARY:")
    print(f"  ✓ Successful: {total_successful}")
    print(f"  ⚠ Incomplete: {total_incomplete}")
    print(f"  ✗ Failed/Errors: {total_errors}")
    print("-" * 60)

    # Save error report if needed
    if error_configs or failed_configs or incomplete_configs:
        error_report_path = output_root / "error_report.txt"
        with open(error_report_path, "w") as f:
            f.write(f"Main Study Error Report\n")
            f.write(f"Run ID: {study_run_id}\n")
            f.write(f"=" * 60 + "\n\n")

            if error_configs:
                f.write(f"ERRORS ({len(error_configs)}):\n")
                for cfg in error_configs:
                    f.write(f"  [{cfg['index']}] {cfg['model']} / {cfg['condition']} / {cfg['topic']}\n")
                    f.write(f"    Error: {cfg['error']}\n\n")

            if failed_configs:
                f.write(f"FAILED ({len(failed_configs)}):\n")
                for cfg in failed_configs:
                    f.write(f"  [{cfg['index']}] {cfg['model']} / {cfg['condition']} / {cfg['topic']}\n")
                    f.write(f"    Reason: {cfg['error']}\n\n")

            if incomplete_configs:
                f.write(f"INCOMPLETE ({len(incomplete_configs)}):\n")
                for cfg in incomplete_configs:
                    f.write(f"  [{cfg['index']}] {cfg['model']} / {cfg['condition']} / {cfg['topic']}\n")
                    f.write(f"    Runs: {cfg['actual']}/{cfg['expected']}\n\n")

        print(f"\n📄 Error report saved to: {error_report_path}")

    export_combined_csv(all_results, output_root, study_run_id)
    return all_results


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_combined_csv(
    all_results: List[Dict[str, Any]], destination_dir: Path, study_run_id: str
) -> None:
    """Export combined results to CSV for analysis."""
    if not all_results:
        print("No successful runs to export.")
        return

    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for entry in all_results:
        detailed_results = entry["results"].get("detailed_results", [])
        for iteration_result in detailed_results:
            pre_block = iteration_result.get("pre_deliberation", {})
            post_block = iteration_result.get("post_deliberation", {})
            pre_dri = pre_block.get("group_score")
            post_dri = post_block.get("group_score")
            dri_change = (
                post_dri - pre_dri if pre_dri is not None and post_dri is not None else None
            )

            rows.append({
                "model_key": entry["model_key"],
                "model": entry["model_label"],
                "model_type": entry.get("model_type", "single"),
                "condition": entry["condition"],
                "topic": entry["topic"],
                "deliberation_topic": entry["deliberation_topic"],
                "run_timestamp": entry["run_timestamp"],
                "pre_dri": pre_dri,
                "post_dri": post_dri,
                "dri_change": dri_change,
                "temperature": entry["config"].get("global_temperature"),
                "group_size": entry["config"].get("group_size"),
                "num_turns": entry["config"].get("num_deliberation_turns"),
                "protocol": entry["config"].get("deliberation_protocol"),
            })

    if not rows:
        print("No iteration-level rows were produced; skipping CSV export.")
        return

    df = pd.DataFrame(rows)
    csv_path = destination_dir / f"main_study_results_{study_run_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCombined CSV exported to: {csv_path}")
    print(f"Total rows: {len(df)}")

    # Print summary statistics
    summary = df.groupby(["model", "condition"])["dri_change"].agg(["count", "mean", "std"])
    print("\nSummary by model and condition (dri_change):")
    print(summary.round(3))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_main_study())
