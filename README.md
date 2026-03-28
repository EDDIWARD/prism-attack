# PRISM: Defeating Search-Augmented Multi-Agent Debate via Factual Persuasion

Official implementation of "When Facts Mislead: Defeating Search-Augmented Multi-Agent Debate via Factual Persuasion"

## Overview

PRISM is a three-stage attack framework that defeats search-augmented multi-agent debate systems through strategic factual persuasion (paltering):

1. **Scout**: Analyzes question weaknesses using 10 persuasion levers
2. **Selector**: Chooses 3 most effective levers via softmax sampling (temperature=0.35)
3. **Builder**: Generates targeted attack prompts using selected levers

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/prism-attack.git
cd prism-attack

# Install dependencies
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

## Configuration

Example PRISM configuration:

```json
{
  "attack_mode": "authority_trap",
  "use_v8_hybrid": true,
  "scout_version": "10lever",
  "v8_scout_model": "gpt-4o",
  "v8_builder_model": "gpt-4o",
  "dataset_name": "medmcqa",
  "num_samples": 100,
  "model_name": "gpt-4o",
  "output_dir": "results/prism_experiment",
  "random_seed": 42
}
```

See `configs/` directory for more examples.

## Project Structure

```
prism-attack/
├── bon_experiment.py              # Main experiment framework
├── dataloader.py                  # Dataset loading (MedMCQA, etc.)
├── search_tools.py                # Tavily search integration
├── prism_framework/
│   ├── layer1_scout.py           # Scout: 10-lever analysis
│   ├── layer2_simple_selector.py # Selector: softmax sampling
│   ├── builder_prompt.py         # Builder: prompt generation
│   ├── builder_llm.py            # Builder: LLM interface
│   ├── v8_phase2_hybrid_generator.py  # Hybrid generator
│   └── cache_manager.py          # Scout/Builder caching
├── configs/
│   ├── example_prism.json        # PRISM configuration
│   └── example_bon16.json        # BoN-16 configuration
└── data/
    └── medmcqa/
        └── dev.json              # MedMCQA dataset

```

## Datasets

The framework supports:
- **MedMCQA**: Medical multiple-choice questions (default)
- **Professional Law**: Legal reasoning questions
- Custom datasets (see `dataloader.py`)

Place dataset files in `data/<dataset_name>/` directory.

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

For Tavily search, set environment variable:
```bash
export TAVILY_API_KEY="your_tavily_key"
```

## Output

Results are saved to the specified `output_dir`:
- `sample_XXXX.json`: Per-sample debate logs and results
- `summary_YYYYMMDD_HHMMSS.json`: Experiment summary with ASR and accuracy

## Responsible Disclosure

In accordance with responsible AI research practices:
- The Builder module contains detailed persuasion strategies for research purposes
- This code is provided for academic research and defense development only
- Users must comply with applicable AI safety guidelines and regulations
- Misuse for malicious purposes is strictly prohibited

## CitationIf you use this code in your research, please cite:```bibtex@article{prism2026,  title={When Facts Mislead: Defeating Search-Augmented Multi-Agent Debate via Factual Persuasion},  author={Anonymous Authors},  journal={Under Review},  year={2026}}```*Note: This is a preprint. Citation information will be updated upon publication.*## License

MIT License

