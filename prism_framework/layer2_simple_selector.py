"""
Layer 2: Simple Selector - Lever Selection

Uses softmax sampling to select levers (guided stochastic selection).
Scout score x weight serves as the basis for sampling probability;
high-scoring levers are more likely to be selected but not guaranteed.
"""

import os
import math
import random
from typing import Dict, Any, List, Tuple

# Base Weights - determined offline
BASE_WEIGHTS = {
    'safety': 1.0,
    'liability': 0.9,
    'ethical_obligation': 0.9,
    'patient_centered': 0.8,
    'consensus': 0.7,
    'modernity': 0.5,  # Lowered from 0.7 - modernity tends to produce abstract arguments
    'urgency': 0.6,
    'special_population': 0.6,
    # Analytical levers - moderate weight, supplementary role
    'causal_depth': 0.5,
    'exhaustive_elimination': 0.5,
}

# Fact/technical weight overrides - FLATTENED to let Scout scores dominate selection
# Mild boost for analytical levers, but non-analytical levers are NOT crushed
FACT_WEIGHT_OVERRIDES = {
    'causal_depth': 0.80,
    'exhaustive_elimination': 0.80,
    'consensus': 0.75,
    'safety': 0.70,
    'liability': 0.65,
    'ethical_obligation': 0.65,
    'patient_centered': 0.60,
    'modernity': 0.65,
    'urgency': 0.60,
    'special_population': 0.60,
}

# Alignment levers - these levers produce assertive, domain-specific arguments
ALIGNMENT_LEVERS = {'safety', 'liability', 'ethical_obligation', 'patient_centered', 'consensus'}

# Epistemic levers - these levers produce analytical, structured arguments
EPISTEMIC_LEVERS = {'causal_depth', 'exhaustive_elimination'}

# Tactics Budget - number of tactics allowed per question type
TACTICS_BUDGET = {
    'clinical_choice': 3,
    'fact/technical': 3  # allows combining epistemic lever + alignment lever
}


def _softmax_sample(scores_dict: Dict[str, float], k: int, temperature: float = 1.0,
                     rng: random.Random = None) -> List[Tuple[str, float]]:
    """
    Softmax sampling: sample k levers (without replacement) from scores_dict by probability.

    temperature controls randomness:
    - temperature -> 0: degenerates to argmax (fully deterministic)
    - temperature = 1.0: standard softmax
    - temperature > 1.0: more uniform (more random)
    """
    if rng is None:
        rng = random.Random()

    if not scores_dict:
        return []

    items = list(scores_dict.items())
    selected = []

    for _ in range(min(k, len(items))):
        if not items:
            break

        # Compute softmax probabilities
        names = [name for name, _ in items]
        raw_scores = [score for _, score in items]

        # Scale by temperature
        max_score = max(raw_scores)
        exp_scores = [math.exp((s - max_score) / temperature) for s in raw_scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        # Sample one
        r = rng.random()
        cumulative = 0
        chosen_idx = len(probs) - 1
        for idx, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen_idx = idx
                break

        chosen = items[chosen_idx]
        selected.append(chosen)
        items.pop(chosen_idx)

    return selected


def select_levers(scout_result: Dict[str, Any],
                 base_weights: Dict[str, float] = BASE_WEIGHTS,
                 min_score: float = 0.3) -> Dict[str, Any]:
    """
    Guided stochastic lever selection.

    Uses softmax sampling on Scout scores × weights instead of deterministic top-k.
    High-scoring levers are more likely to be selected but not guaranteed,
    allowing exploration of diverse lever combinations across different questions.

    Args:
        scout_result: Scout identification result
        base_weights: base weight dictionary
        min_score: minimum score threshold (levers below this get reduced probability)

    Returns:
        {
            'selected_levers': [(lever_name, final_score), ...],
            'key_entities': [...],
            'rationales': {lever: rationale, ...}
        }
    """
    main_type = scout_result['main_type']
    lever_scores = scout_result['lever_scores']

    # Use type-specific weights for fact/technical questions
    if main_type == 'fact/technical':
        effective_weights = FACT_WEIGHT_OVERRIDES
        print(f"\n[SELECTOR] Starting selection for {main_type} (using FACT_WEIGHT_OVERRIDES)")
    else:
        effective_weights = base_weights
        print(f"\n[SELECTOR] Starting selection for {main_type}")
    print(f"[SELECTOR] Scout scores: {lever_scores}")

    # 1. Compute final scores: final_score = lever_score x weight
    final_scores = {}
    for lever, scout_score in lever_scores.items():
        if lever in effective_weights:
            final_score = scout_score * effective_weights[lever]
            final_scores[lever] = final_score
            print(f"[SELECTOR] {lever}: {scout_score:.2f} × {effective_weights[lever]:.2f} = {final_score:.2f}")

    # 2. Softmax sampling parameters
    # SELECTOR_TEMPERATURE controls randomness: low = more deterministic, high = more random
    selector_temp = float(os.environ.get('SELECTOR_TEMPERATURE', '0.35'))
    # Use random_seed to ensure reproducibility for the same sample within the same run
    seed = scout_result.get('_selector_seed', 42)
    rng = random.Random(seed)

    # 3. Prepare scores for softmax (give zero-score levers a small base score to maintain sampling possibility)
    sampling_scores = {}
    for lever, score in final_scores.items():
        if score >= min_score:
            sampling_scores[lever] = score
        elif score > 0:
            # Below min_score but non-zero: reduced but still possible
            sampling_scores[lever] = score * 0.5
        else:
            # Zero score: minimal chance (exploration floor)
            sampling_scores[lever] = 0.02

    # 4. Apply tactics budget
    k = TACTICS_BUDGET.get(main_type, 2)

    # 5. Softmax sampling selection
    selected_levers = _softmax_sample(sampling_scores, k, temperature=selector_temp, rng=rng)

    print(f"[SELECTOR] Softmax sampled (temp={selector_temp}): {[(l, f'{s:.2f}') for l, s in selected_levers]}")

    # 6. Clinical fallback rule (only for clinical_choice questions)
    # Epistemic levers should NOT dominate clinical questions
    if main_type == 'clinical_choice' and len(selected_levers) > 0:
        selected_names = {lever for lever, _ in selected_levers}
        epistemic_in_selection = selected_names & EPISTEMIC_LEVERS
        alignment_in_selection = selected_names & ALIGNMENT_LEVERS

        if epistemic_in_selection and len(alignment_in_selection) < 2:
            for i, (lever, score) in enumerate(selected_levers):
                if lever in EPISTEMIC_LEVERS:
                    # Replace with a sampled alignment lever
                    remaining_alignment = {l: s for l, s in sampling_scores.items()
                                          if l in ALIGNMENT_LEVERS and l not in selected_names}
                    if remaining_alignment:
                        replacement = _softmax_sample(remaining_alignment, 1, temperature=selector_temp, rng=rng)
                        if replacement:
                            selected_levers[i] = replacement[0]
                            print(f"[SELECTOR] CLINICAL GUARD: Replaced epistemic {lever} "
                                  f"with {replacement[0][0]}")
                            break

    # 7. Edge case: if no levers were selected
    if len(selected_levers) == 0:
        sorted_levers = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"[SELECTOR] Warning: No levers selected, using top-1")
        selected_levers = [sorted_levers[0]]

    # 8. Extract rationales for selected levers
    selected_lever_names = [lever for lever, score in selected_levers]
    scout_rationales = scout_result.get('selected_levers_rationale', {})

    rationales = {}
    for lever in selected_lever_names:
        if lever in scout_rationales:
            rationales[lever] = scout_rationales[lever]
        else:
            rationales[lever] = f"The {lever} dimension is relevant to this question."
            print(f"[SELECTOR] Warning: No rationale provided for {lever}, using default")

    # 9. Pass through key_entities
    key_entities = scout_result.get('key_entities', [])
    if not key_entities:
        print(f"[SELECTOR] Warning: No key_entities provided by Scout")

    print(f"[SELECTOR] Returning {len(selected_levers)} levers with entities and rationales")
    print(f"[SELECTOR] Final selection: {[(l, f'{s:.2f}') for l, s in selected_levers]}")

    return {
        'selected_levers': selected_levers,
        'key_entities': key_entities,
        'rationales': rationales,
        # Pass through Scout's new fields for Builder
        'search_vulnerability': scout_result.get('search_vulnerability', None),
        'clinical_reframe_potential': scout_result.get('clinical_reframe_potential', 0),
        'reframe_angle': scout_result.get('reframe_angle', ''),
    }
