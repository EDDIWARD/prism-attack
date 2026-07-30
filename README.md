# PRISM: Defeating Search-Augmented Multi-Agent Debate via Factual Persuasion

Official implementation of "When Facts Mislead: Defeating Search-Augmented Multi-Agent Debate via Factual Persuasion" (ICICS 2026).

## Overview

PRISM is a three-stage attack framework that defeats search-augmented multi-agent debate systems through strategic factual persuasion (paltering):

1. **Scout**: Analyzes question weaknesses using 10 persuasion levers
2. **Selector**: Chooses 3 most effective levers via softmax sampling (temperature=0.35)
3. **Builder**: Generates targeted attack prompts using selected levers

## Key Results

- **GPT-4o**: PRISM achieves a 29.3% targeted attack success rate vs 14.7% for BoN-16 baseline (+14.6pp)
- **Professional Law**: 44.0% vs 28.7% for BoN-16 (+15.3pp)
- **Participant-model transfer**: with Scout/Builder kept on GPT-4o, PRISM exceeds BoN-16 when all three debate agents run on Claude Sonnet 4.5 (34.0% vs 20.0%) and Kimi-K2 (46.3% vs 38.0%)
- **Defense**: the strongest post-debate judge variant reduces PRISM only to 21.7%, still above undefended BoN-16

## Installation

```bash
git clone https://github.com/EDDIWARD/prism-attack.git
cd prism-attack
pip install -r requirements.txt
```

## Quick Start

### Running PRISM Attack

```bash
python bon_experiment.py --config configs/example_prism.json
```

### Running BoN-16 Baseline

```bash
python bon_experiment.py --config configs/example_bon16.json
```

### Running Defense Experiments

```bash
# D2: Interrogation-only defense (best defense)
python bon_experiment.py --config configs/example_defense_d2.json

# D3: Full-transcript defense
python bon_experiment.py --config configs/example_defense_d3.json

# D4: Transcript + interrogation defense
python bon_experiment.py --config configs/example_defense_d4.json
```

## Configuration

### Full Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `attack_mode` | string | `"bon"` | `"bon"` (Best-of-N baseline) or `"prism"` (PRISM attack) |
| `dataset_name` | string | `"medmcqa"` | `"medmcqa"`, `"professional_law"`, or `"truthfulqa"` |
| `num_samples` | int | 100 | Number of samples to evaluate |
| `start_from` | int | 0 | Starting sample index |
| `model_name` | string | `"gpt-4o"` | Model for debate agents |
| `random_seed` | int | 42 | Random seed (use 42/43/44 for 3-run experiments) |
| `cache_seed` | null/int | null | Set `null` for non-deterministic debates (recommended) |
| `output_dir` | string | — | Directory for result files |
| `bon_n` | int | 16 | N for Best-of-N (only when `attack_mode="bon"`) |
| `use_prism_hybrid` | bool | true | Enable Scout-Selector-Builder pipeline |
| `scout_version` | string | `"10lever"` | Scout version (`"10lever"` recommended) |
| `prism_scout_model` | string | `"gpt-4o"` | Model for Scout analysis |
| `prism_builder_model` | string | `"gpt-4o"` | Model for Builder generation |
| `enable_judge_defense` | bool | false | Enable post-debate judge defense |
| `judge_defense_variant` | string | `"d2"` | Defense variant: `"d2"`, `"d3"`, or `"d4"` |
| `fixed_targets_file` | string | — | Path to JSON with pre-set target answers (see below) |
| `debater_api.api_key` | string | — | OpenAI-compatible API key |
| `debater_api.base_url` | string | `"https://api.openai.com/v1"` | API base URL |

### Target Answer Selection

By default, target (adversarial) answers are selected **randomly** from the incorrect options for each sample. To ensure reproducibility, you can provide a fixed targets file:

```json
{
  "fixed_targets_file": "configs/fixed_targets.json"
}
```

The fixed targets file maps sample IDs to target answers:

```json
{
  "0": {"target_answer": "b"},
  "1": {"target_answer": "d"},
  "2": {"target_answer": "a"}
}
```

If a sample ID is not found in the file, a random target is used as fallback. When using `random_seed`, random target selection is deterministic across runs with the same seed.

### Example Configs

**PRISM attack** (`configs/example_prism.json`):

```json
{
  "attack_mode": "prism",
  "use_prism_hybrid": true,
  "scout_version": "10lever",
  "prism_scout_model": "gpt-4o",
  "prism_builder_model": "gpt-4o",
  "dataset_name": "medmcqa",
  "num_samples": 100,
  "model_name": "gpt-4o",
  "output_dir": "results/prism_experiment",
  "random_seed": 42
}
```

**PRISM + D2 defense** (`configs/example_defense_d2.json`):

```json
{
  "attack_mode": "prism",
  "use_prism_hybrid": true,
  "enable_judge_defense": true,
  "judge_defense_variant": "d2",
  "dataset_name": "medmcqa",
  "num_samples": 100,
  "output_dir": "results/defense_d2",
  "random_seed": 42
}
```

See `configs/` directory for all examples.


## Project Structure

```
prism-attack/
├── bon_experiment.py              # Main experiment framework (attack + defense)
├── dataloader.py                  # Dataset loading
├── search_tools.py                # Tavily search integration
├── prism_framework/
│   ├── layer1_scout.py           # Scout: 10-lever analysis
│   ├── layer2_simple_selector.py # Selector: softmax sampling
│   ├── builder_prompt.py         # Builder: prompt generation
│   ├── builder_llm.py            # Builder: LLM interface
│   └── cache_manager.py          # Scout/Builder caching
├── configs/                       # Example configurations
│   ├── example_prism.json        # PRISM attack config
│   ├── example_bon16.json        # BoN-16 baseline config
│   ├── example_defense_d2.json   # D2 defense config
│   ├── example_defense_d3.json   # D3 defense config
│   └── example_defense_d4.json   # D4 defense config
└── data/                          # Datasets
```

## Datasets

Supported datasets:
- **MedMCQA**: Medical multiple-choice questions
- **Professional Law**: Legal reasoning questions
- **TruthfulQA**: Factual knowledge questions


## API Configuration

Set your API credentials in the config file:

```json
{
  "debater_api": {
    "api_key": "YOUR_API_KEY",
    "base_url": "https://api.openai.com/v1"
  }
}
```

For Tavily search:
```bash
export TAVILY_API_KEY="your_tavily_key"
```

## Defense Experiments (RQ3)

We provide three post-debate judge defense variants to evaluate PRISM's resilience against defensive mechanisms. All defenses are **attack-agnostic** (no manipulation warnings) and **search-augmented** (up to 3 search rounds).

**Naming note:** the config values `d2`/`d3`/`d4` correspond to the defenses reported as **D1**/**D2**/**D3** in the paper. The config identifiers are kept for backward compatibility with existing result directories.

### Defense Variants

| Variant | Judge Sees | Interrogation | Description |
|---------|------------|:---:|-------------|
| **D2** | Question + options + consensus | Yes (verifier only) | Judge generates probing questions for the verifier, then evaluates responses with search |
| **D3** | Question + options + consensus + **full transcript** | No | Judge reads the entire debate transcript, evaluates directly with search |
| **D4** | Question + options + consensus + **full transcript** | Yes (verifier only) | Judge reads transcript, interrogates verifier, then evaluates with search |

### Defense Configuration

Add these fields to your config JSON to enable a defense:

```json
{
  "enable_judge_defense": true,
  "judge_defense_variant": "d2"
}
```

Valid values for `judge_defense_variant`: `"d2"`, `"d3"`, `"d4"`.

You can also enable defense via CLI:

```bash
python bon_experiment.py --config configs/example_prism.json --enable-defense --defense-variant d2
```

### Reproducing Defense Results (RQ3)

To reproduce the 3-run defense results reported in the paper:

```bash
# D2 defense (best defense, 3 seeds)
# Edit random_seed and output_dir for each run
python bon_experiment.py --config configs/example_defense_d2.json  # seed=42
# Change random_seed to 43, output_dir to results/defense_d2_run2, then:
python bon_experiment.py --config configs/example_defense_d2.json  # seed=43
# Change random_seed to 44, output_dir to results/defense_d2_run3, then:
python bon_experiment.py --config configs/example_defense_d2.json  # seed=44

# Repeat for D3 and D4 with the same 3 seeds (42, 43, 44)
python bon_experiment.py --config configs/example_defense_d3.json
python bon_experiment.py --config configs/example_defense_d4.json
```

### Defense Output Format

When defense is enabled, each `sample_XXXX.json` includes a `defense` field:

```json
{
  "defense": {
    "enabled": true,
    "defense_type": "judge_d2",
    "debate_answer_before_defense": "b",
    "reflection_answer": "c",
    "changed": true,
    "reflection_text": "...",
    "defense_token_usage": {...},
    "defense_search_calls": 2,
    "defense_search_queries": ["query1", "query2"],
    "judge_questions": "...",
    "verifier_response": "...",
    "answer_trajectory": ["B", "B", "B"]
  }
}
```

Key metrics to compute from results:
- **Final ASR**: Attack success rate after defense override
- **Block rate**: Fraction of initially-successful attacks that the defense blocked
- **False positive rate**: Fraction of correct answers that the defense incorrectly overrode
- **Accuracy**: Overall accuracy after defense

## Output

Results are saved to `output_dir`:
- `sample_XXXX.json`: Per-sample debate logs (including defense results if enabled)
- `summary_YYYYMMDD_HHMMSS.json`: Experiment summary with ASR and accuracy


## Responsible Disclosure

This code is provided for academic research and defense development only:
- The Builder module contains detailed persuasion strategies for research purposes
- Users must comply with applicable AI safety guidelines
- Misuse for malicious purposes is strictly prohibited

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhang2026prism,
  title={When Facts Mislead: Defeating Search-Augmented Multi-Agent Debate via Factual Persuasion},
  author={Zhang, Enhua and Chang, Yan},
  booktitle={Proceedings of the 28th International Conference on Information and Communications Security (ICICS)},
  series={Lecture Notes in Computer Science},
  publisher={Springer},
  year={2026}
}
```

*Note: Volume and page numbers will be added once the proceedings are published.*

## License

This project is released under the MIT License. See LICENSE file for details.
