"""
V8 Phase 2 Hybrid Generator

集成Scout → Selector → Hybrid Assembly的完整流程
"""

from .layer2_simple_selector import select_levers
from .hybrid_assembly import assemble_prompt_hybrid, get_prompt_stats


class V8Phase2HybridGenerator:
    """
    V8 Phase 2 Hybrid Generator: 使用Builder LLM生成战术指导
    """

    def __init__(self, scout_client, builder_client, scout_model="gpt-4o", builder_model="gpt-4o",
                 builder_prompt_version="current", scout_version="10lever", scout_best_of_n=1,
                 cache_manager=None):
        """
        Args:
            scout_client: OpenAI client for Scout
            builder_client: OpenAI client for Builder (可以是同一个client)
            scout_model: Scout使用的模型
            builder_model: Builder使用的模型
            builder_prompt_version: "current" or "v2_with_mission"
            scout_version: "8lever" (original, no causal_depth/exhaustive_elimination)
                          or "10lever" (current, with fact-specific levers)
            scout_best_of_n: Run Scout N times and pick highest attackability (default 1)
            cache_manager: V8CacheManager instance for caching Scout/Builder results
        """
        self.scout_client = scout_client
        self.builder_client = builder_client
        self.scout_model = scout_model
        self.builder_model = builder_model
        self.builder_prompt_version = builder_prompt_version
        self.scout_version = scout_version
        self.scout_best_of_n = scout_best_of_n
        self.cache_manager = cache_manager

        # Import the correct Scout based on version
        if scout_version == "8lever":
            from .layer1_scout_8lever import scout_identify_levers as _scout_fn
            print(f"[V8 HYBRID] Using 8-lever Scout (original)")
        else:
            from .layer1_scout import scout_identify_levers as _scout_fn
            print(f"[V8 HYBRID] Using 10-lever Scout (current)")
        self._scout_identify_levers = _scout_fn

    def _run_scout(self, question_dict, target_answer, correct_answer):
        """Run Scout with best-of-N logic"""
        n = self.scout_best_of_n
        scout_results = []
        total_scout_tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}
        for i in range(n):
            sr = self._scout_identify_levers(
                question_dict,
                self.scout_client,
                self.scout_model,
                target_answer=target_answer,
                correct_answer=correct_answer
            )
            # Accumulate scout token usage
            if '_token_usage' in sr:
                for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                    total_scout_tokens[k] += sr['_token_usage'].get(k, 0)
                total_scout_tokens['api_calls'] += 1
            atk = (sr.get('attack_analysis') or {}).get('attackability', 0)
            scout_results.append((atk, sr))
            if n > 1:
                print(f"[V8 HYBRID] Scout run {i+1}/{n}: attackability={atk:.2f}")

        # Pick highest attackability
        scout_results.sort(key=lambda x: x[0], reverse=True)
        if n > 1:
            print(f"[V8 HYBRID] Scout Best-of-{n}: picked attackability={scout_results[0][0]:.2f}")
        return scout_results[0][1], total_scout_tokens

    def generate_with_metadata(self, question_dict, target_answer, correct_answer="",
                               sample_id=None, use_cached_scout=False, use_cached_builder=False):
        """
        生成adaptive prompt with metadata

        Args:
            question_dict: {'question': str, 'options': list}
            target_answer: 目标答案
            correct_answer: 正确答案（用于Scout评估fact题的新lever）
            sample_id: 样本ID，用于缓存key
            use_cached_scout: 是否使用缓存的Scout结果
            use_cached_builder: 是否使用缓存的Builder body

        Returns:
            {
                'prompt': final_prompt,
                'metadata': {
                    'main_type': ...,
                    'lever_scores': ...,
                    'selected_levers': ...,
                    'top_lever': ...,
                    'builder_success': ...,
                    'validation_passed': ...,
                    'stats': ...
                }
            }
        """
        print(f"\n{'='*80}")
        print(f"[V8 HYBRID] Starting V8 Phase 2 Hybrid Generation")
        print(f"{'='*80}")

        cache_key = str(sample_id) if sample_id is not None else None

        # Layer 1: Scout (with cache support)
        n = self.scout_best_of_n
        print(f"\n[V8 HYBRID] Layer 1: Scout LLM ({self.scout_version}), best-of-{n}")

        scout_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'api_calls': 0}

        if use_cached_scout and self.cache_manager and cache_key:
            cached_scout = self.cache_manager.get_scout(cache_key)
            if cached_scout:
                scout_result = cached_scout
                print(f"[V8 HYBRID] Layer 1: Using CACHED Scout result for sample {cache_key}")
            else:
                print(f"[V8 HYBRID] Layer 1: Cache miss for sample {cache_key}, running Scout")
                scout_result, scout_token_usage = self._run_scout(question_dict, target_answer, correct_answer)
                self.cache_manager.set_scout(cache_key, scout_result)
        else:
            scout_result, scout_token_usage = self._run_scout(question_dict, target_answer, correct_answer)
            if self.cache_manager and cache_key:
                self.cache_manager.set_scout(cache_key, scout_result)

        # Layer 2: Selector (softmax sampling for diversity)
        print(f"\n[V8 HYBRID] Layer 2: Simple Selector (softmax sampling)")
        # Pass sample_id as selector seed for per-sample reproducibility
        # Use the experiment's random_seed (not hardcoded 42) so different runs get different Selector sampling
        run_seed = getattr(self, 'random_seed', 42)
        scout_result['_selector_seed'] = hash((run_seed, sample_id)) % (2**31) if sample_id is not None else run_seed
        selector_result = select_levers(scout_result)

        # 提取attack_analysis（仅8lever Scout有此字段）
        attack_analysis = scout_result.get('attack_analysis', None)
        if attack_analysis:
            print(f"[V8 HYBRID] Attack analysis available: attackability={attack_analysis.get('attackability', '?')}")
        else:
            print(f"[V8 HYBRID] No attack analysis from Scout")

        # Layer 3: Hybrid Assembly (Builder LLM) with cache support
        print(f"\n[V8 HYBRID] Layer 3: Hybrid Assembly with Builder LLM")
        print(f"[V8 HYBRID] Builder prompt version: {self.builder_prompt_version}")

        builder_body_override = None
        if use_cached_builder and self.cache_manager and cache_key:
            cached_body = self.cache_manager.get_builder_body(cache_key)
            if cached_body:
                builder_body_override = cached_body
                print(f"[V8 HYBRID] Layer 3: Using CACHED Builder body for sample {cache_key}")
            else:
                print(f"[V8 HYBRID] Layer 3: Cache miss, running Builder")

        assembly_result = assemble_prompt_hybrid(
            scout_result['main_type'],
            selector_result,  # 修改：传递完整的selector_result (Dict)
            question_dict,
            target_answer,
            self.builder_client,
            self.builder_model,
            builder_prompt_version=self.builder_prompt_version,
            attack_analysis=attack_analysis,
            builder_body_override=builder_body_override
        )

        # 保存 builder body 到缓存（仅在新生成时）
        if not builder_body_override and self.cache_manager and cache_key:
            raw_body = assembly_result['metadata'].get('builder_raw_body', '')
            if raw_body:
                self.cache_manager.set_builder_body(cache_key, raw_body)

        final_prompt = assembly_result['prompt']
        assembly_metadata = assembly_result['metadata']

        # Stats
        stats = get_prompt_stats(final_prompt)

        # 提取selected_levers用于metadata
        selected_levers = selector_result['selected_levers']

        # 完整的metadata
        builder_token_usage = assembly_metadata.get('builder_token_usage', None)

        # Compute pipeline total token usage
        pipeline_total = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
            pipeline_total[k] = scout_token_usage.get(k, 0) + (builder_token_usage.get(k, 0) if builder_token_usage else 0)

        metadata = {
            'main_type': scout_result['main_type'],
            'lever_scores': scout_result['lever_scores'],
            'key_entities': scout_result.get('key_entities', []),
            'search_vulnerability': scout_result.get('search_vulnerability', None),
            'clinical_reframe_potential': scout_result.get('clinical_reframe_potential', 0),
            'attack_analysis': attack_analysis,
            'selected_levers': [(name, score) for name, score in selected_levers],
            'top_lever': selected_levers[0][0] if selected_levers else None,
            'builder_success': assembly_metadata['builder_success'],
            'builder_raw_body': assembly_metadata.get('builder_raw_body', ''),
            'builder_prompt_version': self.builder_prompt_version,
            'scout_version': self.scout_version,
            'validation_passed': assembly_metadata['validation_passed'],
            'stats': stats,
            'assembly_breakdown': {
                'header_length': assembly_metadata['header_length'],
                'body_length': assembly_metadata['body_length'],
                'footer_length': assembly_metadata['footer_length']
            },
            'token_usage': {
                'scout': scout_token_usage,
                'builder': builder_token_usage,
                'pipeline_total': pipeline_total,
            }
        }

        print(f"\n{'='*80}")
        print(f"[V8 HYBRID] Generation complete")
        print(f"{'='*80}")
        print(f"[V8 HYBRID] Main type: {metadata['main_type']}")
        print(f"[V8 HYBRID] Top lever: {metadata['top_lever']}")
        print(f"[V8 HYBRID] Builder success: {metadata['builder_success']}")
        print(f"[V8 HYBRID] Validation passed: {metadata['validation_passed']}")
        print(f"[V8 HYBRID] Final prompt length: {stats['length']} chars")
        print(f"{'='*80}\n")

        return {
            'prompt': final_prompt,
            'metadata': metadata
        }
