from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "skills" / "local-inference-lab-evaluating-prompts"
BENCH = ROOT / "skills" / "local-inference-lab-running-benchmarks"

TETRIS = "Make me an amazing mind blowing tetris clone with cool music and awesome visuals"
PLATFORMER = (
    "Build me an incredible, high fidelity clone of the original SMB, Level 1-1 and 1-2, "
    "as a single page web app. It needs to be faithful to the original, and feature music, "
    "good animations, sprites, correct physics, etc. etc, but not to the extent that you try "
    "to use original assets."
)


class PromptAndBenchmarkTests(unittest.TestCase):
    def test_exact_canonical_prompts_are_preserved(self) -> None:
        self.assertIn(TETRIS, (PROMPTS / "references" / "tetris.md").read_text(encoding="utf-8"))
        self.assertIn(PLATFORMER, (PROMPTS / "references" / "platformer.md").read_text(encoding="utf-8"))

    def test_flamingo_prompt_is_not_invented(self) -> None:
        text = (PROMPTS / "references" / "flamingo.md").read_text(encoding="utf-8")
        self.assertIn("does not yet contain a canonical Flamingo prompt", text)
        self.assertIn("Do not invent", text)
        self.assertIn("Not tested", text)

    def test_prompt_text_is_not_loaded_by_benchmark_skill(self) -> None:
        text = (BENCH / "SKILL.md").read_text(encoding="utf-8")
        self.assertNotIn(TETRIS, text)
        self.assertNotIn("Level 1-1 and 1-2", text)

    def test_benchmark_skill_requires_true_target_only_control(self) -> None:
        text = (BENCH / "references" / "llm-inference-bench.md").read_text(encoding="utf-8")
        self.assertIn("`MTP=0` is insufficient", text)
        for mode in ("MTP", "DFlash", "DSpark", "n-gram"):
            self.assertIn(mode, text)

    def test_standard_baseline_includes_estonia_and_lavd_with_optional_hotel(self) -> None:
        skill = (BENCH / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("Estonia and LAVD", skill)
        self.assertIn("Hotel Lights only", skill)


if __name__ == "__main__":
    unittest.main()
