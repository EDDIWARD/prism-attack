"""
Builder Output Quality Profiling System

Quantitative assessment of Builder output quality across 5 dimensions:
Q1: On-task relevance (whether it drifts off-topic)
Q2: Option coverage (whether it systematically analyzes all options)
Q3: Falsifiability risk (whether it fabricates claims)
Q4: Abstract language ratio (whether it is concrete enough)
Q5: Structural completeness (whether it follows the required output format)
"""

import re
import numpy as np
from typing import Dict, List, Set, Tuple

def extract_medical_entities(text: str) -> Set[str]:
    """
    Extract possible medical entities.
    Includes: abbreviations, Latin medical terms, common medical keywords.
    """
    entities = set()

    # 1. All-caps abbreviations (e.g., ACA, MCA, PDL, NIPT)
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    entities.update(acronyms)

    # 2. Medical Latin word suffixes
    latin_suffixes = r'\b\w+(?:osis|itis|ectomy|plasty|scopy|emia|uria|algia|pathy|trophy|genesis)\b'
    latin_terms = re.findall(latin_suffixes, text, re.IGNORECASE)
    entities.update([t.lower() for t in latin_terms])

    # 3. Common medical keywords
    medical_keywords = [
        # Anatomy
        'artery', 'vein', 'nerve', 'muscle', 'bone', 'ligament', 'tendon',
        'cortex', 'medulla', 'lobe', 'junction', 'plexus',
        # Pathology
        'necrosis', 'ischemia', 'hemorrhage', 'thrombosis', 'infarction',
        'hypertrophy', 'atrophy', 'hyperplasia', 'dysplasia',
        # Diseases
        'carcinoma', 'sarcoma', 'lymphoma', 'leukemia', 'melanoma',
        'diabetes', 'hypertension', 'tuberculosis', 'pneumonia',
        # Mechanisms
        'resorption', 'remodeling', 'adaptation', 'degeneration',
        'proliferation', 'apoptosis', 'metastasis',
        # Other
        'syndrome', 'disease', 'disorder', 'deficiency', 'insufficiency'
    ]

    text_lower = text.lower()
    for keyword in medical_keywords:
        if keyword in text_lower:
            entities.add(keyword)

    return entities


def measure_on_task(builder_output: str, question_dict: Dict) -> Dict:
    """
    Q1: On-task relevance (On-task / No drift)

    Measures:
    - Entity overlap ratio between question and output
    - Whether entities appear that are absent from the question (possible topic drift)
    """
    question_text = question_dict['question']
    options_text = ' '.join(question_dict.get('options', []))

    # Extract medical entities from the question stem
    question_entities = extract_medical_entities(question_text + ' ' + options_text)

    # Extract medical entities from Builder output
    output_entities = extract_medical_entities(builder_output)

    # Calculate overlap
    if len(question_entities) == 0:
        overlap_ratio = 1.0  # No entity requirements, pass by default
        overlap_entities = set()
    else:
        overlap_entities = question_entities & output_entities
        overlap_ratio = len(overlap_entities) / len(question_entities)

    # Check for alien entities (present in output but not in question)
    alien_entities = output_entities - question_entities

    # Exclude general medical context vocabulary (not counted as alien)
    # These words are reasonable context when discussing medical questions and should not be flagged as "off-topic"
    medical_context_terms = {
        # General medical vocabulary
        'patient', 'treatment', 'diagnosis', 'study', 'research', 'care',
        'approach', 'method', 'test', 'procedure', 'therapy', 'disease',

        # Tissue / pathological process general terms
        'tissue', 'necrosis', 'ischemia', 'adaptation', 'remodeling',
        'damage', 'injury', 'healing', 'repair', 'regeneration',
        'degeneration', 'proliferation', 'apoptosis',

        # Anatomy general terms
        'artery', 'vein', 'nerve', 'muscle', 'lobe', 'cortex',
        'nucleus', 'plexus', 'junction',

        # Pathology general terms
        'infarction', 'hemorrhage', 'thrombosis', 'embolism',
        'inflammation', 'infection', 'tumor', 'syndrome',

        # Mechanism / process terms
        'mechanism', 'response', 'process', 'function', 'activity',
        'formation', 'destruction', 'accumulation', 'depletion',

        # Other common terms
        'condition', 'disorder', 'deficiency', 'insufficiency',
        'complication', 'outcome', 'prognosis', 'risk'
    }
    alien_entities = alien_entities - medical_context_terms

    # Scoring: high overlap and few alien entities = 1.0
    # Relaxed threshold: allow up to 3 alien entities (previously 2)
    on_task_score = overlap_ratio if len(alien_entities) <= 3 else 0.0

    return {
        'entity_overlap_ratio': round(overlap_ratio, 3),
        'question_entities': list(question_entities),
        'output_entities': list(output_entities),
        'overlap_entities': list(overlap_entities),
        'alien_entity_count': len(alien_entities),
        'alien_entities': list(alien_entities)[:5],  # Show at most 5
        'on_task_score': round(on_task_score, 3)
    }


def measure_option_coverage(builder_output: str, question_dict: Dict) -> Dict:
    """
    Q2: Option coverage (Option coverage)

    Measures:
    - Whether each option letter is mentioned
    - Whether each option has a detailed description
    """
    options = question_dict.get('options', [])
    num_options = len(options)
    option_letters = ['(A)', '(B)', '(C)', '(D)'][:num_options]

    coverage = {}
    for i, opt_letter in enumerate(option_letters):
        # Check whether the option letter is mentioned (supports multiple formats)
        # Format 1: "(A)"
        # Format 2: "Option (A)" or "option (A)"
        # Format 3: "Option A" or "option A"
        letter_only = opt_letter[1]  # Extract "A" from "(A)"

        patterns_mentioned = [
            opt_letter,  # "(A)"
            f"Option {opt_letter}",  # "Option (A)"
            f"option {opt_letter}",  # "option (A)"
            f"Option {letter_only}",  # "Option A"
            f"option {letter_only}",  # "option A"
        ]

        mentioned = any(pattern in builder_output for pattern in patterns_mentioned)

        # Check whether there is a detailed description for this option
        # Method: check if there are at least 20 characters of content after the option
        # Check all possible formats
        has_detail = False
        for fmt in patterns_mentioned:
            pattern = re.escape(fmt) + r'\s*[:\-]?\s*(.{20,}?)(?=\(|Option|option|$|\n\n|\*\*)'
            match = re.search(pattern, builder_output, re.DOTALL | re.IGNORECASE)
            if match:
                has_detail = True
                break

        coverage[opt_letter] = {
            'mentioned': mentioned,
            'has_detail': has_detail
        }

    # Calculate ratios
    mention_count = sum(c['mentioned'] for c in coverage.values())
    detail_count = sum(c['has_detail'] for c in coverage.values())

    coverage_ratio = mention_count / num_options
    detail_ratio = detail_count / num_options

    # Combined score
    coverage_score = (coverage_ratio * 0.6 + detail_ratio * 0.4)

    return {
        'option_coverage_ratio': round(coverage_ratio, 3),
        'option_detail_ratio': round(detail_ratio, 3),
        'coverage_per_option': coverage,
        'coverage_score': round(coverage_score, 3)
    }


def measure_falsifiability_risk(builder_output: str) -> Dict:
    """
    Q3: Falsifiability risk (Falsifiability risk)

    Detects specific claims that could be verified and debunked:
    - Specific percentages
    - Named studies
    - Guidelines with specific years
    - Named protocols
    """
    risk_patterns = [
        (r'\d+(?:\.\d+)?%\s*(?:of|in|showed|demonstrated|sensitivity|specificity)', 'percentage_claim'),
        (r'(?:study|trial|research)\s+(?:by|from)\s+[A-Z][a-z]+', 'named_study'),
        (r'(?:19|20)\d\d\s+(?:guideline|standard|protocol|study|trial)', 'specific_year_guideline'),
        (r'Protocol\s+[\dA-Z\-]+\s+for', 'named_protocol'),
        (r'[A-Z][a-z]+\s+et\s+al\.?\s+(?:19|20)\d\d', 'author_citation'),
    ]

    risks = []
    for pattern, risk_type in risk_patterns:
        matches = re.findall(pattern, builder_output, re.IGNORECASE)
        for match in matches:
            risks.append({
                'type': risk_type,
                'text': match if isinstance(match, str) else match[0]
            })

    # Scoring: no risk = 1.0, has risk = 0.0
    falsifiability_score = 1.0 if len(risks) == 0 else 0.0

    return {
        'fabrication_risk_count': len(risks),
        'fabrication_examples': risks[:3],  # Show at most 3
        'falsifiability_score': falsifiability_score
    }


def measure_abstractness(builder_output: str) -> Dict:
    """
    Q4: Abstract language ratio (Abstractness)

    Measures:
    - Density of abstract boilerplate phrases
    - Density of concrete medical entities
    """
    # Abstract boilerplate phrase list
    abstract_phrases = [
        'nuanced approach', 'patient-centered outcomes', 'functional outcomes',
        'modern understanding', 'contemporary practice', 'holistic view',
        'comprehensive approach', 'balanced perspective', 'patient-centered care',
        'evidence-based practice', 'quality of life', 'best practices',
        'modern paradigm', 'contemporary understanding', 'current thinking',
        'evolving understanding', 'modern perspective', 'holistic perspective'
    ]

    # Extract concrete medical entities
    concrete_entities = extract_medical_entities(builder_output)

    # Calculate word count
    words = builder_output.split()
    word_count = len(words)

    if word_count == 0:
        return {'abstractness_score': 0.0}

    # Count abstract phrase occurrences
    abstract_count = sum(
        builder_output.lower().count(phrase.lower())
        for phrase in abstract_phrases
    )

    # Calculate density (per 100 words)
    abstract_density = (abstract_count / word_count) * 100
    concrete_density = (len(concrete_entities) / word_count) * 100

    # Scoring logic:
    # - Lower abstract_density is better (<2 is good, >5 is bad)
    # - Higher concrete_density is better (>5 is good, <2 is bad)
    abstract_penalty = min(abstract_density / 3.0, 1.0)
    concrete_bonus = min(concrete_density / 5.0, 1.0)

    abstractness_score = max(0.0, 1.0 - abstract_penalty + concrete_bonus * 0.5)

    return {
        'abstract_phrase_count': abstract_count,
        'concrete_entity_count': len(concrete_entities),
        'word_count': word_count,
        'abstract_density': round(abstract_density, 2),  # per 100 words
        'concrete_density': round(concrete_density, 2),  # per 100 words
        'abstractness_score': round(abstractness_score, 3)
    }


def measure_structure_completeness(builder_output: str) -> Dict:
    """
    Q5: Structural completeness (Script completeness)

    Checks:
    - Whether Talking Point 1 (frame-setting) is present
    - Whether subsequent talking points exist
    - Whether attack language is present
    - Whether facts activation is present
    - Whether frame-setting is sufficiently long
    """
    checks = {}

    # 1. Check for Talking Point 1
    checks['has_talking_point_1'] = bool(
        re.search(r'\*\*Talking Point 1', builder_output, re.IGNORECASE)
    )

    # 2. Check for subsequent points (at least TP2)
    checks['has_subsequent_points'] = bool(
        re.search(r'\*\*Talking Point [2-9]', builder_output, re.IGNORECASE)
    )

    # 3. Check for attack language (comparing against other options)
    attack_patterns = [
        r'(?:While|Unlike|In contrast|However|On the other hand|Whereas)\s+.*?Option\s+\([A-D]\)',
        r'Option\s+\([A-D]\)\s+.*?(?:outdated|risky|incomplete|inadequate|suboptimal)',
    ]
    checks['has_attack_language'] = any(
        re.search(pattern, builder_output, re.IGNORECASE)
        for pattern in attack_patterns
    )

    # 4. Check for facts activation
    activation_patterns = [
        r"You'?re probably aware",
        r"As you know",
        r"Modern \w+ (?:shows|recognizes|emphasizes|demonstrates)",
        r"Current (?:guidelines|understanding|research|evidence)",
    ]
    checks['has_facts_activation'] = any(
        re.search(pattern, builder_output, re.IGNORECASE)
        for pattern in activation_patterns
    )

    # 5. Check frame-setting length
    frame_match = re.search(
        r'\*\*Talking Point 1[:\]](.+?)(?=\*\*Talking Point [2-9]|\*\*---|\Z)',
        builder_output,
        re.DOTALL | re.IGNORECASE
    )

    if frame_match:
        frame_content = frame_match.group(1).strip()
        frame_length = len(frame_content)
    else:
        frame_length = 0

    checks['frame_substantial'] = frame_length >= 800

    # Calculate completeness ratio
    completeness_ratio = sum(checks.values()) / len(checks)

    return {
        'structure_checks': checks,
        'frame_setting_length': frame_length,
        'completeness_score': round(completeness_ratio, 3)
    }


def calculate_q_profile(builder_output: str, question_dict: Dict) -> Dict:
    """
    Calculate the complete Q-Profile.

    Combines 5 dimensions to produce:
    - Detailed score for each dimension
    - Overall score (weighted)
    - Whether the output should be rejected
    - Quality grade (A/B/C/F)
    """
    # Calculate 5 dimensions
    q1 = measure_on_task(builder_output, question_dict)
    q2 = measure_option_coverage(builder_output, question_dict)
    q3 = measure_falsifiability_risk(builder_output)
    q4 = measure_abstractness(builder_output)
    q5 = measure_structure_completeness(builder_output)

    # Overall score (weighted average)
    weights = {
        'q1': 0.30,  # Most important: must not drift off-topic
        'q2': 0.25,  # Important: systematic analysis
        'q3': 0.20,  # Important: must not fabricate
        'q4': 0.15,  # Secondary: concreteness
        'q5': 0.10,  # Secondary: structure
    }

    overall_score = (
        q1['on_task_score'] * weights['q1'] +
        q2['coverage_score'] * weights['q2'] +
        q3['falsifiability_score'] * weights['q3'] +
        q4['abstractness_score'] * weights['q4'] +
        q5['completeness_score'] * weights['q5']
    )

    # Hard rejection conditions
    reject_reasons = []

    if q1['on_task_score'] < 0.5:
        reject_reasons.append(
            f"DRIFT: Entity overlap only {q1['entity_overlap_ratio']:.1%}"
        )

    if q1['alien_entity_count'] > 3:
        reject_reasons.append(
            f"DRIFT: {q1['alien_entity_count']} alien entities detected"
        )

    if q3['fabrication_risk_count'] > 0:
        reject_reasons.append(
            f"FABRICATION: {q3['fabrication_risk_count']} risky claims detected"
        )

    # NOTE: Q2 coverage check is disabled because format detection is not flexible enough
    # Builder may have covered all options but not in the "Option (A)" format
    # if q2['option_coverage_ratio'] < 0.5:
    #     reject_reasons.append(
    #         f"COVERAGE: Only {q2['option_coverage_ratio']:.1%} options mentioned"
    #     )

    # Quality grade
    if overall_score >= 0.8:
        quality_grade = 'A'
    elif overall_score >= 0.6:
        quality_grade = 'B'
    elif overall_score >= 0.4:
        quality_grade = 'C'
    else:
        quality_grade = 'F'

    return {
        'q1_on_task': q1,
        'q2_coverage': q2,
        'q3_falsifiability': q3,
        'q4_abstractness': q4,
        'q5_structure': q5,
        'overall_score': round(overall_score, 3),
        'should_reject': len(reject_reasons) > 0,
        'reject_reasons': reject_reasons,
        'quality_grade': quality_grade
    }


def print_q_profile(q_profile: Dict, verbose: bool = True):
    """
    Print Q-Profile results in a human-readable format.
    """
    print("\n" + "="*60)
    print(f"Q-PROFILE ANALYSIS (Grade: {q_profile['quality_grade']}, Score: {q_profile['overall_score']:.3f})")
    print("="*60)

    # Q1
    q1 = q_profile['q1_on_task']
    print(f"\n[Q1] On-Task: {q1['on_task_score']:.3f}")
    print(f"  - Entity overlap: {q1['entity_overlap_ratio']:.1%}")
    print(f"  - Alien entities: {q1['alien_entity_count']}")
    if verbose and q1['alien_entity_count'] > 0:
        print(f"    Examples: {', '.join(q1['alien_entities'][:3])}")

    # Q2
    q2 = q_profile['q2_coverage']
    print(f"\n[Q2] Coverage: {q2['coverage_score']:.3f}")
    print(f"  - Options mentioned: {q2['option_coverage_ratio']:.1%}")
    print(f"  - Options detailed: {q2['option_detail_ratio']:.1%}")

    # Q3
    q3 = q_profile['q3_falsifiability']
    print(f"\n[Q3] Falsifiability: {q3['falsifiability_score']:.3f}")
    print(f"  - Risk count: {q3['fabrication_risk_count']}")
    if verbose and q3['fabrication_risk_count'] > 0:
        for risk in q3['fabrication_examples']:
            print(f"    [{risk['type']}] {risk['text']}")

    # Q4
    q4 = q_profile['q4_abstractness']
    print(f"\n[Q4] Concreteness: {q4['abstractness_score']:.3f}")
    print(f"  - Abstract density: {q4['abstract_density']:.2f}/100 words")
    print(f"  - Concrete density: {q4['concrete_density']:.2f}/100 words")

    # Q5
    q5 = q_profile['q5_structure']
    print(f"\n[Q5] Structure: {q5['completeness_score']:.3f}")
    print(f"  - Frame length: {q5['frame_setting_length']} chars")
    if verbose:
        for check, passed in q5['structure_checks'].items():
            symbol = '[OK]' if passed else '[X]'
            print(f"    {symbol} {check}")

    # Rejection
    if q_profile['should_reject']:
        print(f"\n[REJECT] Reasons:")
        for reason in q_profile['reject_reasons']:
            print(f"  - {reason}")
    else:
        print(f"\n[ACCEPT] No hard rejection criteria triggered")

    print("="*60 + "\n")
