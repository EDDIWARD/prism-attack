"""V8 Pipeline Cache Manager — cache Scout 和 Builder result"""

import json
import os
from typing import Dict, Optional


class V8CacheManager:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[CACHE] Loaded {len(data)} entries from {self.cache_file}")
            return data
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_file) or '.', exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def get_scout(self, sample_id: str) -> Optional[dict]:
        entry = self.cache.get(sample_id, {})
        return entry.get('scout_result')

    def set_scout(self, sample_id: str, scout_result: dict):
        self.cache.setdefault(sample_id, {})['scout_result'] = scout_result
        self._save()

    def get_builder_body(self, sample_id: str) -> Optional[str]:
        entry = self.cache.get(sample_id, {})
        return entry.get('builder_body')

    def set_builder_body(self, sample_id: str, builder_body: str):
        self.cache.setdefault(sample_id, {})['builder_body'] = builder_body
        self._save()
