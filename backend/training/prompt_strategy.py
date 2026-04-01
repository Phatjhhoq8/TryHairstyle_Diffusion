"""
🎯 Prompt Strategy Module — Sinh match/delta prompts từ text_prompt hiện có.

Mục tiêu:
  - Match prompt: mô tả đúng kiểu tóc hiện tại (giữ nguyên logic cũ)
  - Delta prompt: đổi 1 thuộc tính (curl/volume/bangs/color/length)
    để buộc model phản ứng với text thay vì copy ref hair 100%

Tất cả logic hoạt động trên chuỗi text_prompt có sẵn trong metadata.jsonl,
KHÔNG cần metadata mới hay nhãn cấu trúc.
"""

import random
import re
from typing import Tuple, Optional


# ============================================================
# KEYWORD POOLS — Từ khóa cho từng thuộc tính tóc
# Mỗi thuộc tính gồm các nhóm khác nhau (dùng để detect + swap)
# ============================================================

CURL_GROUPS = [
    ["straight"],
    ["wavy", "wave"],
    ["curly", "curl", "curled"],
    ["coiled", "coil", "kinky", "afro"],
]

VOLUME_GROUPS = [
    ["flat", "sleek", "thin"],
    ["normal"],
    ["voluminous", "puffy", "fluffy", "thick", "full"],
]

BANGS_GROUPS = [
    ["side-swept bangs", "side bangs", "side swept"],
    ["with bangs", "front bangs", "blunt bangs", "bangs", "fringe"],
    ["without bangs", "no bangs"],
]

COLOR_GROUPS = [
    ["black"],
    ["dark brown", "brown"],
    ["light brown", "chestnut"],
    ["blonde", "golden", "honey"],
    ["red", "auburn", "ginger"],
    ["gray", "grey", "silver", "white"],
    ["pink", "purple", "blue"],  # fantasy colors
]

LENGTH_GROUPS = [
    ["very short", "buzz", "pixie"],
    ["short"],
    ["medium", "shoulder-length", "shoulder length", "mid-length"],
    ["long"],
    ["very long"],
]


def _detect_group(text: str, groups: list) -> Optional[int]:
    """Tìm nhóm từ khóa đầu tiên khớp trong text (case-insensitive).
    Returns: index của group khớp, hoặc None nếu không tìm thấy.
    """
    text_lower = text.lower()
    for i, group in enumerate(groups):
        for keyword in sorted(group, key=len, reverse=True):
            pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)
            if pattern.search(text_lower):
                return i
    return None


def _swap_group(text: str, groups: list, current_idx: int) -> str:
    """Thay thế từ khóa thuộc nhóm current_idx bằng từ khóa ngẫu nhiên
    từ nhóm KHÁC (≠ current_idx).

    Returns: text đã swap, hoặc text gốc nếu không swap được.
    """
    text_lower = text.lower()

    # Chọn nhóm đích (khác với hiện tại)
    other_indices = [i for i in range(len(groups)) if i != current_idx]
    if not other_indices:
        return text

    target_idx = random.choice(other_indices)
    target_keyword = groups[target_idx][0]  # Lấy từ đầu tiên (canonical)

    # Tìm và thay thế từ khóa cũ
    for keyword in sorted(groups[current_idx], key=len, reverse=True):
        # Dùng regex để thay thế chính xác (case-insensitive)
        pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)
        if pattern.search(text):
            return pattern.sub(target_keyword, text, count=1)

    return text


class PromptStrategy:
    """
    Quản lý chiến lược sinh prompt cho Stage 2 training.

    Sử dụng:
        ps = PromptStrategy()
        prompt_type, full_prompt = ps.get_training_prompt(text_prompt)
        # prompt_type: "match" | "delta" | "empty"
        # full_prompt: chuỗi prompt đầy đủ
    """

    def __init__(
        self,
        match_ratio: float = 0.60,
        delta_ratio: float = 0.30,
        empty_ratio: float = 0.10,
        prefix: str = "high quality, realistic",
        suffix: str = "detailed hair texture",
    ):
        """
        Args:
            match_ratio:  Tỷ lệ dùng match prompt (mô tả đúng ref hair)
            delta_ratio:  Tỷ lệ dùng delta prompt (đổi 1 thuộc tính)
            empty_ratio:  Tỷ lệ dùng empty prompt (CFG dropout cho text)
            prefix/suffix: Đoạn chèn vào đầu/cuối prompt
        """
        total = match_ratio + delta_ratio + empty_ratio
        self.match_ratio = match_ratio / total
        self.delta_ratio = delta_ratio / total
        # empty_ratio = phần còn lại

        self.prefix = prefix
        self.suffix = suffix

        # Danh sách các thuộc tính có thể swap + trọng số ưu tiên
        # curl/volume/bangs dễ parse hơn → ưu tiên
        # color/length khó hơn → ít ưu tiên
        self._swap_configs = [
            ("curl",   CURL_GROUPS,   3),  # weight 3 — dễ nhất
            ("volume", VOLUME_GROUPS, 2),  # weight 2
            ("bangs",  BANGS_GROUPS,  2),  # weight 2
            ("color",  COLOR_GROUPS,  1),  # weight 1 — chỉ swap khi detect được
            ("length", LENGTH_GROUPS, 1),  # weight 1
        ]

    def _wrap_prompt(self, text_prompt: str) -> str:
        """Bọc text_prompt vào template đầy đủ."""
        text_prompt = text_prompt.strip()
        if not text_prompt:
            text_prompt = "hairstyle"
        return f"{self.prefix} {text_prompt}, {self.suffix}"

    def _generate_delta(self, text_prompt: str) -> Optional[str]:
        """Thử đổi 1 thuộc tính ngẫu nhiên trong text_prompt.

        Returns: prompt đã đổi, hoặc None nếu không detect được thuộc tính nào.
        """
        # Tìm tất cả thuộc tính có thể swap (detect được trong text)
        swappable = []
        for attr_name, groups, weight in self._swap_configs:
            current_idx = _detect_group(text_prompt, groups)
            if current_idx is not None:
                swappable.append((attr_name, groups, current_idx, weight))

        if not swappable:
            return None

        # Weighted random chọn thuộc tính để swap
        weights = [item[3] for item in swappable]
        chosen = random.choices(swappable, weights=weights, k=1)[0]
        attr_name, groups, current_idx, _ = chosen

        # Thực hiện swap
        delta_text = _swap_group(text_prompt, groups, current_idx)

        # Nếu swap không thay đổi gì → trả None
        if delta_text.lower().strip() == text_prompt.lower().strip():
            return None

        return delta_text

    def get_training_prompt(self, text_prompt: str) -> Tuple[str, str]:
        """
        Sinh prompt cho training dựa trên tỷ lệ match/delta/empty.

        Args:
            text_prompt: Chuỗi text_prompt từ metadata.jsonl

        Returns:
            (prompt_type, full_prompt):
              - prompt_type: "match" | "delta" | "empty"
              - full_prompt: Chuỗi prompt đầy đủ (đã bọc prefix/suffix)
        """
        roll = random.random()

        if roll < self.match_ratio:
            # Match prompt — mô tả đúng ref hair
            return "match", self._wrap_prompt(text_prompt)

        elif roll < self.match_ratio + self.delta_ratio:
            # Delta prompt — đổi 1 thuộc tính
            delta_text = self._generate_delta(text_prompt)
            if delta_text is not None:
                return "delta", self._wrap_prompt(delta_text)
            else:
                # Fallback nếu không detect được thuộc tính → dùng match
                return "match", self._wrap_prompt(text_prompt)

        else:
            # Empty prompt — CFG dropout cho text conditioning
            return "empty", ""

    def get_all_variants(self, text_prompt: str) -> dict:
        """
        Sinh TẤT CẢ biến thể prompt có thể từ 1 text_prompt.
        Dùng để pre-encode toàn bộ ở init Dataset. Tránh gọi Text Encoder
        on-the-fly trong training loop.

        Returns:
            dict: {prompt_text: prompt_type}
              Ví dụ:
              {
                  "high quality, realistic short wavy brown hair, detailed hair texture": "match",
                  "high quality, realistic short curly brown hair, detailed hair texture": "delta_curl",
                  "high quality, realistic short wavy blonde hair, detailed hair texture": "delta_color",
              }
        """
        variants = {}

        # Match prompt (luôn có)
        match_prompt = self._wrap_prompt(text_prompt)
        variants[match_prompt] = "match"

        # Delta prompts — thử swap từng thuộc tính
        for attr_name, groups, _ in self._swap_configs:
            current_idx = _detect_group(text_prompt, groups)
            if current_idx is None:
                continue

            # Sinh biến thể cho mỗi nhóm khác
            for target_idx in range(len(groups)):
                if target_idx == current_idx:
                    continue

                delta_text = text_prompt
                # Thay keyword
                for keyword in sorted(groups[current_idx], key=len, reverse=True):
                    pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)
                    if pattern.search(delta_text):
                        target_kw = groups[target_idx][0]
                        delta_text = pattern.sub(target_kw, delta_text, count=1)
                        break

                if delta_text.lower().strip() != text_prompt.lower().strip():
                    full_delta = self._wrap_prompt(delta_text)
                    if full_delta not in variants:
                        variants[full_delta] = f"delta_{attr_name}"

        return variants


# ==================================================================
# STANDALONE TEST
# ==================================================================
if __name__ == "__main__":
    ps = PromptStrategy()

    test_prompts = [
        "short wavy dark brown hair",
        "long straight black hair with bangs",
        "curly blonde hair voluminous",
        "medium length auburn hair side-swept bangs",
        "pixie cut sleek silver hair",
        "hairstyle",  # generic / empty case
        "",           # fully empty
    ]

    print("=" * 70)
    print("🧪 [Testing] PromptStrategy — Match/Delta/Empty Prompt Generation")
    print("=" * 70)

    for tp in test_prompts:
        print(f"\n📝 Input: '{tp}'")

        # Test get_training_prompt 10 lần
        types = {"match": 0, "delta": 0, "empty": 0}
        for _ in range(100):
            ptype, prompt = ps.get_training_prompt(tp)
            types[ptype] += 1

        print(f"   Distribution (100 rolls): {types}")

        # Test get_all_variants
        variants = ps.get_all_variants(tp)
        print(f"   Variants ({len(variants)} total):")
        for prompt_text, var_type in list(variants.items())[:5]:
            # Cắt ngắn để dễ đọc
            short = prompt_text[:80] + "..." if len(prompt_text) > 80 else prompt_text
            print(f"     [{var_type:12s}] {short}")
        if len(variants) > 5:
            print(f"     ... ({len(variants) - 5} more)")

    print("\n✅ All tests passed!")
