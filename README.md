# PRISM: Defeating Search-Augmented Multi-Agent Debate via Factual Persuasion

Official implementation of "When Facts Mislead: Defeating Search-Augmented Multi-Agent Debate via Factual Persuasion"

## Overview

PRISM is a three-stage attack framework that defeats search-augmented multi-agent debate systems through strategic factual persuasion (paltering):

1. **Scout**: Analyzes question weaknesses using 10 persuasion levers
2. **Selector**: Chooses 3 most effective levers via softmax sampling (temperature=0.35)
3. **Builder**: Generates targeted attack prompts using selected levers

## Key Results

- **GPT-4o**: PRISM achieves 30% ASR vs 14% for BoN-16 baseline (+16pp)
- **Cross-model transfer**: PRISM consistently outperforms BoN-16 across all tested models
- **Llama 3.3 70B**: Most vulnerable (57% ASR)
- **Qwen3-Max**: Most robust (25% ASR)
- **Claude Sonnet 4.5**: Highest sensitivity to PRISM (+19pp improvement)

## Installation

**Note:** During the review period, this repository is hosted anonymously. Please download as ZIP:

1. Visit: https://anonymous.4open.science/r/prism-attack/
2. Click "Download ZIP" button
3. Extract the archive

Then install dependencies:

```bash
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
├── dataloader.py                  # Dataset loading
├── search_tools.py                # Tavily search integration
├── prism_framework/
│   ├── layer1_scout.py           # Scout: 10-lever analysis
│   ├── layer2_simple_selector.py # Selector: softmax sampling
│   ├── builder_prompt.py         # Builder: prompt generation
│   ├── builder_llm.py            # Builder: LLM interface
│   └── cache_manager.py          # Scout/Builder caching
├── configs/                       # Example configurations
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

## Output

Results are saved to `output_dir`:
- `sample_XXXX.json`: Per-sample debate logs
- `summary_YYYYMMDD_HHMMSS.json`: Experiment summary with ASR and accuracy


## Responsible Disclosure

This code is provided for academic research and defense development only:
- The Builder module contains detailed persuasion strategies for research purposes
- Users must comply with applicable AI safety guidelines
- Misuse for malicious purposes is strictly prohibited

## Citation

If you use this code in your research, please cite:

```bibtex
@article{prism2026,
  title={When Facts Mislead: Defeating Search-Augmented Multi-Agent Debate via Factual Persuasion},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```

*Note: Citation information will be updated upon publication.*

## License

This project is released under the MIT License. See LICENSE file for details.
