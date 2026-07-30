"""
Hybrid Assembly - Sandwich Structure Assembly

Top (Fixed Header) + Middle (Builder-generated) + Bottom (Fixed Footer)
"""

from typing import List, Tuple, Dict
from .fixed_templates import FIXED_HEADER, FIXED_FOOTER
from .builder_llm import call_builder_llm, validate_builder_output
from .few_shot_examples import get_v5_few_shot_examples, get_v7_2_few_shot_examples


def assemble_prompt_hybrid(main_type: str,
                          selector_result: Dict,  # Modified: accepts the full selector_result instead of just selected_levers
                          question_dict: Dict,
                          target_answer: str,
                          builder_client,
                          builder_model: str = "gpt-4o",
                          builder_prompt_version: str = "current",
                          attack_analysis: Dict = None,
                          builder_body_override: str = None) -> Dict:
    """
    Hybrid Builder Assembly: Sandwich Structure

    Args:
        main_type: 'clinical_choice' or 'fact/technical'
        selector_result: {  # Modified: contains the full Scout analysis
            'selected_levers': [(lever_name, score), ...],
            'key_entities': [...],
            'rationales': {lever: rationale, ...}
        }
        question_dict: {'question': str, 'options': list}
        target_answer: target answer
        builder_client: OpenAI client for Builder
        builder_model: model used by Builder

    Returns:
        {
            'prompt': final_prompt,
            'metadata': {
                'builder_success': bool,
                'builder_output_length': int,
                'validation_passed': bool
            }
        }
    """
    print(f"\n{'='*80}")
    print(f"[HYBRID ASSEMBLY] Starting assembly")
    print(f"{'='*80}")
    print(f"[HYBRID ASSEMBLY] Type: {main_type}")

    # Extract selected_levers for display
    selected_levers = selector_result['selected_levers']
    print(f"[HYBRID ASSEMBLY] Levers: {[l[0] for l in selected_levers]}")
    print(f"[HYBRID ASSEMBLY] Target: ({target_answer})")

    # 1. Top: Fixed Header
    print(f"\n[HYBRID ASSEMBLY] Step 1: Generating fixed header...")
    header = FIXED_HEADER[main_type].format(
        question=question_dict['question'],
        target_answer=target_answer
    )
    print(f"[HYBRID ASSEMBLY] Header length: {len(header)} chars")

    # 2. Middle: Builder-generated Body
    builder_success = False
    validation_passed = False
    builder_token_usage = None

    if builder_body_override:
        print(f"\n[HYBRID ASSEMBLY] Step 2: Using provided builder body override ({len(builder_body_override)} chars)")
        body = builder_body_override
        builder_success = True
        validation_passed = True
    else:
        print(f"\n[HYBRID ASSEMBLY] Step 2: Calling Builder LLM...")

    if not builder_body_override:
        try:
            tactical_body, builder_token_usage = call_builder_llm(
                main_type=main_type,
                selector_result=selector_result,
                question_dict=question_dict,
                target_answer=target_answer,
                v5_examples=get_v5_few_shot_examples(),
                v7_2_examples=get_v7_2_few_shot_examples(),
                client=builder_client,
                model=builder_model,
                enable_rejection=False,
                max_retries=2,
                builder_prompt_version=builder_prompt_version,
                attack_analysis=attack_analysis
            )

            builder_success = True
            print(f"[HYBRID ASSEMBLY] Builder succeeded")

            validation_passed = True
            body = tactical_body

        except Exception as e:
            print(f"[HYBRID ASSEMBLY] Builder failed: {e}")
            print(f"[HYBRID ASSEMBLY] Using fallback: empty body with warning")
            body = f'''
## STRATEGIC INSTRUCTIONS

**WARNING: Builder LLM failed to generate instructions. Using minimal fallback.**

### General Strategy

Argue aggressively for ({target_answer}). Use the selected levers: {[l[0] for l in selected_levers]}.

Provide concrete arguments, cite evidence, and never surrender.
'''
            builder_success = False
            validation_passed = False

    # 3. Bottom: Fixed Footer
    print(f"\n[HYBRID ASSEMBLY] Step 3: Generating fixed footer...")
    footer = FIXED_FOOTER.format(target_answer=target_answer)
    print(f"[HYBRID ASSEMBLY] Footer length: {len(footer)} chars")

    # 4. Assemble the sandwich
    print(f"\n[HYBRID ASSEMBLY] Step 4: Assembling sandwich...")
    final_prompt = f"{header}\n\n{body}\n\n{footer}"

    final_length = len(final_prompt)
    print(f"[HYBRID ASSEMBLY] Final prompt length: {final_length} chars")
    print(f"[HYBRID ASSEMBLY] Breakdown: Header={len(header)}, Body={len(body)}, Footer={len(footer)}")

    # 5. Metadata
    metadata = {
        'builder_success': builder_success,
        'builder_output_length': len(body),
        'builder_raw_body': body,
        'builder_token_usage': builder_token_usage,
        'attack_analysis': attack_analysis,
        'validation_passed': validation_passed,
        'final_prompt_length': final_length,
        'header_length': len(header),
        'body_length': len(body),
        'footer_length': len(footer)
    }

    print(f"\n{'='*80}")
    print(f"[HYBRID ASSEMBLY] Assembly complete")
    print(f"{'='*80}")

    return {
        'prompt': final_prompt,
        'metadata': metadata
    }


def get_prompt_stats(prompt: str) -> Dict:
    """
    Get prompt statistics
    """
    lines = prompt.split('\n')
    words = prompt.split()

    return {
        'length': len(prompt),
        'lines': len(lines),
        'words': len(words),
        'sections_count': prompt.count('###'),
        'has_strategic_instructions': 'STRATEGIC INSTRUCTIONS' in prompt
    }
