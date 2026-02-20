# DelibSim: Multi-Agent LLM Deliberation Simulator

A framework for simulating deliberation processes using multiple LLM agents. DelibSim measures discourse quality during deliberation (AQuA) and deliberative outcomes via pre/post surveys (DRI).

## Overview

DelibSim orchestrates multi-agent deliberation sessions where LLM instances participate in structured discussions on policy topics. The framework:

- Administers pre-deliberation surveys to measure initial positions
- Runs turn-based deliberation with configurable prompts
- Administers post-deliberation surveys to measure opinion changes
- Calculates the Deliberative Reason Index (DRI) to quantify intersubjective consistency
- Optionally assesses discourse quality using AQuA (Automated Quality Assessment)

## Experimental Design

The main study replicates experiments with:
- **11 model configurations**: 9 single-model groups + 2 mixed-model groups
- **12 deliberation topics**: Policy issues from citizen assemblies worldwide
- **3 conditions**: No Treatment (survey-only), Basic Treatment, Normative Treatment
- **5 repetitions** per configuration for statistical robustness

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd deliberation_simulator_final

# Clone AQuA-Repo into delib_sim/external:
gh repo clone mabehrendt/AQuA

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your OpenRouter API key:

```
OPENROUTER_API_KEY=your_key_here
```

Optional metadata for OpenRouter rankings:
```
OPENROUTER_SITE_URL=https://your-site.example
OPENROUTER_SITE_NAME=DelibSim
```

## Usage

### Run the Main Study

```bash
python run_main_study.py
```

This runs the full experimental design with all model configurations, topics, and conditions.

### Custom Simulations

Edit `delib_sim/config/simulation_config.yaml` to customize:
- `llm_setup`: Model configurations and counts
- `deliberation_protocol`: "free_flow" or "guided"
- `num_deliberation_turns`: Turns per deliberation session
- `group_size`: Agents per deliberation group
- `survey_name`: Topic from surveys.xlsx
- `deliberation_prompt_type`: "treatment", "control_1", or "control_2"

Then run:
```bash
python run_main_study.py
```

## Project Structure

```
deliberation_simulator_final/
├── run_main_study.py              # Main experiment runner
├── requirements.txt               # Python dependencies
├── delib_sim/
│   ├── config/
│   │   └── simulation_config.yaml # Simulation parameters
│   ├── data/
│   │   ├── dri_surveys/
│   │   │   └── surveys.xlsx       # Survey definitions (12 topics)
│   │   └── simulation_results/    # Output directory
│   └── delib_sim/
│       ├── run.py                 # SimulationRunner orchestrator
│       ├── llm/
│       │   └── manager.py         # LLM instance management
│       ├── deliberation/
│       │   └── engine.py          # Multi-agent conversation logic
│       ├── survey/
│       │   └── dri_survey.py      # Survey administration
│       ├── analysis/
│       │   ├── dri_calculator.py     # DRI score computation
│       │   └── dqi_assessor.py    # AQuA discourse quality
│       ├── utils/
│       │   ├── config.py          # Pydantic configuration models
│       │   └── surveys.py         # Survey loading utilities
│       └── visualization/
│           └── dri_plots.py       # Result visualization
├── scripts/                       # Analysis scripts
├── paper/                         # Research paper drafts
└── external/                      # External dependencies (AQuA)
```

## Output

Results are saved to `delib_sim/data/simulation_results/main_study/<timestamp>/`:

- `main_study_results_<timestamp>.csv`: Combined results for analysis
- `model_*/`: Per-configuration directories with:
  - `config_used.yaml`: Configuration snapshot
  - `simulation_results_*.json`: Detailed results with DRI scores
  - `deliberation_transcript_*.txt`: Conversation logs

## Metrics

### DRI (Deliberative Reason Index)
Measures intersubjective consistency among group members based on survey responses. Higher DRI indicates greater agreement on the relative importance of considerations.

### AQuA (Automated Quality Assessment)
Adapted from the Discourse Quality Index (DQI), AQuA uses transformer models to assess deliberation quality dimensions including respect, justification, and constructive politics.

## License

MIT License
