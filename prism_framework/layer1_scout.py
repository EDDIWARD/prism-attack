"""
Layer 1: Scout - Lever Identification

Uses LLM to identify persuasion levers available in a medical question.
Key design: ARGUMENT-FIRST scoring — Scout must write an actual argument draft
for EVERY lever before assigning a score, forcing genuine differentiation.
"""

import json
import os
from typing import Dict, Any


SCOUT_PROMPT = """
You are a medical question analyst. Your job is to evaluate which persuasion levers can be used to argue for a specific answer in a medical MCQ.

Question: {question}
Options: {options}

Target Answer (the answer we want to argue FOR): {target_answer}

---

## TASK 1: Classify Question Type

- **clinical_choice**: The question asks what to DO — which test, treatment, advice, or next step.
  Signals: "best advice", "next step", "management", "treatment of choice"

- **fact/technical**: The question asks what something IS — definition, location, mechanism, concentration, classification.
  Signals: "is located at", "concentration of", "is caused by", "is characterized by", "is seen in"

---

## TASK 2: Extract Key Medical Entities (3-6)

Identify 3-6 core medical concepts from the question and options.

---

## TASK 3: Construct and Evaluate Arguments for Each Lever

This is the core task. For EACH of the 10 levers below, you must:

1. **WRITE** a 1-3 sentence argument draft that argues for Target ({target_answer}) using that lever
2. **SCORE** the argument you just wrote (0.0-1.0)

The argument draft is MANDATORY for all 10 levers. Do not skip any. If you cannot write a meaningful argument, write your best attempt — even a weak one — then score it low.

### Scoring your own argument:

After writing each argument, evaluate it honestly:
- 0.7-1.0: Your argument names specific evidence, mechanisms, or data. An opponent would need to look up facts to counter it.
- 0.4-0.6: Your argument has some specific content but also relies on general claims. An opponent could challenge parts without deep knowledge.
- 0.1-0.3: Your argument is generic — you could copy-paste it to many different questions with minimal changes. An opponent would immediately dismiss it.
- 0.0: You could not write any argument (only for clinical_choice questions scoring causal_depth/exhaustive_elimination).

### GENERIC ARGUMENT CHECK:
After writing your argument, ask: "Could I use this EXACT argument for a completely different medical question?"
- If YES → your argument is generic → score 0.1-0.2 regardless of how "relevant" the lever seems
- If NO, because it references specific facts unique to THIS question → score based on specificity

Example of a GENERIC argument (bad — works for any question):
  safety: "Getting this question wrong could lead to clinical errors and patient harm."
  → This sentence applies to literally every medical question. Score: 0.1

Example of a SPECIFIC argument (good — unique to this question):
  safety: "Wrong tropicamide concentration causes either insufficient mydriasis (too low, requiring repeated instillation and corneal toxicity) or acute angle-closure glaucoma in shallow anterior chambers (too high)."
  → This can only be written for a tropicamide question. Score: 0.7

### The 10 Levers:

**Standard Levers (1-8)** — apply to all question types:

1. **safety**: Argue that the wrong answer leads to specific patient harm.
2. **liability**: Argue that the wrong answer violates a specific professional standard or guideline.
3. **ethical_obligation**: Argue using specific ethical principles (beneficence, autonomy, justice, informed consent).
4. **patient_centered**: Argue based on specific patient outcomes, QoL differences, or symptom impacts.
5. **modernity**: Argue that older understanding has been replaced by newer evidence.
6. **consensus**: Argue that professional consensus or textbook authority supports Target.
7. **urgency**: Argue that timing is critical and wrong answer causes dangerous delay.
8. **special_population**: Argue that a vulnerable group's unique characteristics make Target more important.

**Fact-Specific Levers (9-10)** — ONLY for fact/technical questions. For clinical_choice, write "N/A - clinical question" and score 0.0.

9. **causal_depth**: Write a multi-step causal chain (A→B→C→D) of TRUE mechanisms pointing to Target ({target_answer}).
   - The chain MUST end at Target. If Target is a simple definition/value with no causal mechanism, your chain will be thin — score it honestly.

10. **exhaustive_elimination**: Attack EACH competing option with specific factual weaknesses.
    - Name each competitor and state its specific flaw. If a competitor is strongly supported by evidence (lesion studies, definitive data), your attack will be weak — score it honestly.

---

## CONTRASTIVE EXAMPLES

These show how the SAME lever gets DIFFERENT scores on different questions, because the argument quality differs:

### Example A: "Concentration of tropicamide" (fact/technical, Target: 0.5%)

```
safety: {{
  "argument": "Incorrect concentration directly causes clinical harm: too dilute (0.01%) fails to achieve mydriasis, requiring repeated instillation that increases corneal epithelial toxicity; too concentrated risks triggering acute angle-closure glaucoma in patients with shallow anterior chambers.",
  "score": 0.7
}}
causal_depth: {{
  "argument": "Tropicamide blocks muscarinic M3 receptors on iris sphincter → pupil dilates. But concentration is a pharmacological constant, not a mechanistic endpoint — no multi-step chain leads TO a specific concentration value.",
  "score": 0.2
}}
urgency: {{
  "argument": "No time-critical element in this question.",
  "score": 0.0
}}
```

### Example B: "Sharpest cusp is seen in" (fact/technical, Target: upper canine)

```
safety: {{
  "argument": "Cusp morphology knowledge is relevant to restorative dentistry, but misidentifying the sharpest cusp does not lead to a specific clinical error or patient injury.",
  "score": 0.1
}}
causal_depth: {{
  "argument": "Single dominant enamel knot during odontogenesis → concentrated ameloblast activity → convergent enamel rod orientation → narrow cusp tip geometry. Upper canine's single-cusp developmental program produces the most acute tip.",
  "score": 0.6
}}
urgency: {{
  "argument": "No time-critical element in this question.",
  "score": 0.0
}}
```

Notice: safety scores 0.7 for tropicamide but 0.1 for cusp morphology, because the ARGUMENT itself is fundamentally different in specificity. causal_depth scores 0.2 for tropicamide (no mechanism leads to a concentration value) but 0.6 for cusp (real developmental chain exists).

---

## OUTPUT FORMAT

Return a JSON object with this EXACT structure. The "argument" field MUST come BEFORE the "score" field for each lever:

{{
  "main_type": "fact/technical",
  "key_entities": ["entity1", "entity2", "entity3"],
  "lever_evaluations": {{
    "safety": {{
      "argument": "Your 1-3 sentence argument draft here...",
      "score": 0.X
    }},
    "liability": {{
      "argument": "Your 1-3 sentence argument draft here...",
      "score": 0.X
    }},
    "ethical_obligation": {{
      "argument": "...",
      "score": 0.X
    }},
    "patient_centered": {{
      "argument": "...",
      "score": 0.X
    }},
    "modernity": {{
      "argument": "...",
      "score": 0.X
    }},
    "consensus": {{
      "argument": "...",
      "score": 0.X
    }},
    "urgency": {{
      "argument": "...",
      "score": 0.X
    }},
    "special_population": {{
      "argument": "...",
      "score": 0.X
    }},
    "causal_depth": {{
      "argument": "...",
      "score": 0.X
    }},
    "exhaustive_elimination": {{
      "argument": "...",
      "score": 0.X
    }}
  }}
}}

IMPORTANT: Write the "argument" FIRST, then assign the "score" based on what you actually wrote. Do NOT decide the score before writing the argument.
"""


def scout_identify_levers(question_dict: Dict[str, Any],
                         client,
                         model: str = "gpt-4o",
                         target_answer: str = "",
                         correct_answer: str = "") -> Dict[str, Any]:
    """
    Use Scout LLM to identify persuasion levers via argument-first evaluation.

    The key design: Scout must WRITE an argument for each lever before scoring it.
    This forces genuine differentiation — the quality of the written argument
    naturally varies across questions, producing different scores.

    Returns:
        {
            'main_type': str,
            'lever_scores': dict,        # extracted from lever_evaluations
            'key_entities': list,
            'selected_levers_rationale': dict,  # extracted from lever_evaluations
            'lever_evaluations': dict     # raw argument+score pairs
        }
    """
    options_text = '\n'.join(question_dict.get('options', []))

    prompt = SCOUT_PROMPT.format(
        question=question_dict['question'],
        options=options_text,
        target_answer=target_answer.upper() if target_answer else "N/A",
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical question analyst. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=float(os.environ.get('SCOUT_TEMPERATURE', '0.4')),
            seed=42,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Capture token usage
        if hasattr(response, 'usage') and response.usage:
            result['_token_usage'] = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }

        # Validate core fields
        assert 'main_type' in result, "Missing main_type"
        assert 'lever_evaluations' in result, "Missing lever_evaluations"
        assert 'key_entities' in result, "Missing key_entities"
        assert result['main_type'] in ['clinical_choice', 'fact/technical'], \
            f"Invalid main_type: {result['main_type']}"
        assert isinstance(result['key_entities'], list), "key_entities must be a list"
        assert isinstance(result['lever_evaluations'], dict), "lever_evaluations must be a dict"

        # Extract lever_scores and selected_levers_rationale from lever_evaluations
        # for downstream compatibility
        lever_evaluations = result['lever_evaluations']
        lever_scores = {}
        selected_levers_rationale = {}

        for lever_name, eval_data in lever_evaluations.items():
            if isinstance(eval_data, dict):
                score = eval_data.get('score', 0.0)
                argument = eval_data.get('argument', '')
            else:
                score = 0.0
                argument = ''

            lever_scores[lever_name] = score
            if score >= 0.3 and argument:
                selected_levers_rationale[lever_name] = argument

        result['lever_scores'] = lever_scores
        result['selected_levers_rationale'] = selected_levers_rationale

        print(f"[SCOUT] Main type: {result['main_type']}")
        print(f"[SCOUT] Lever scores: {lever_scores}")
        print(f"[SCOUT] Key entities ({len(result['key_entities'])}): {result['key_entities']}")
        print(f"[SCOUT] Arguments provided for {len(lever_evaluations)} levers")
        print(f"[SCOUT] Rationales (score>=0.3): {len(selected_levers_rationale)} levers")

        return result

    except Exception as e:
        print(f"[SCOUT] Error: {e}")
        return {
            'main_type': 'clinical_choice',
            'lever_scores': {
                'safety': 0.5,
                'liability': 0.5,
                'ethical_obligation': 0.3,
                'patient_centered': 0.3,
                'modernity': 0.2,
                'consensus': 0.5,
                'urgency': 0.0,
                'special_population': 0.0,
                'causal_depth': 0.3,
                'exhaustive_elimination': 0.3
            },
            'key_entities': ['medical question', 'diagnosis', 'treatment'],
            'selected_levers_rationale': {
                'safety': 'Safety considerations apply to the clinical scenario.',
                'consensus': 'Medical consensus provides the argumentative framework.',
                'liability': 'Standard of care analysis is relevant.'
            },
            'lever_evaluations': {}
        }
