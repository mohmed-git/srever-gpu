"""tests.test_two_tier_router - Comprehensive tests and sabotage verification for Two-Tier Smart Router.

Verifies:
1. resolve_tier: strictly adheres to <= 6 words -> 1.5b, > 6 words -> 7b, CJK handling.
2. TwoTierMtEngine: dispatch to Tier 1 vs Tier 2, batch reassembly in exact order.
3. Cache isolation: norm_hash across tiers never collides.
4. build_mt_engine integration: MT_TWO_TIER_ENABLED and MT_BACKEND=two_tier.
5. Sabotages: intentional mutations caught by test suite.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.config import Settings
from app.mt import (
    MtEngine,
    MtItem,
    MtResult,
    TwoTierMtEngine,
    build_mt_engine,
    resolve_tier,
)
from app.server import _resolve_item_tier, norm_hash


class MockSubEngine(MtEngine):
    """Mock sub-engine to trace calls and return predictable MtResults."""

    def __init__(self, settings: Settings, model_name: str, tier_name: str) -> None:
        super().__init__(settings)
        self.model_name = model_name
        self.tier_name = tier_name
        self.call_history: list[list[Any]] = []

    def load(self) -> None:
        pass

    @property
    def ready(self) -> bool:
        return True

    def translate_batch(self, items: list[Any]) -> list[MtResult]:
        self.call_history.append(items)
        results = []
        for it in items:
            text = it.text if isinstance(it, MtItem) else it[0]
            results.append(
                MtResult(
                    text=f"[{self.tier_name}] translated: {text}",
                    mt_ms=15.0 if self.tier_name == "1.5b" else 150.0,
                    backend="mock",
                    model=self.model_name,
                    input_tokens=10,
                    output_tokens=10,
                    batch_size=len(items),
                    hollow=False,
                    hollow_reason=None,
                    tier=self.tier_name,
                )
            )
        return results

    def info(self) -> dict[str, Any]:
        return {"backend": "mock", "model": self.model_name, "tier": self.tier_name}


class TestTwoTierRouter(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = Settings(mt_two_tier_enabled=True, mt_two_tier_threshold_words=6)
        self.mock_tier1 = MockSubEngine(self.settings, "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "1.5b")
        self.mock_tier2 = MockSubEngine(self.settings, "Qwen/Qwen2.5-7B-Instruct-AWQ", "7b")
        self.router = TwoTierMtEngine(
            self.settings,
            tier1_engine=self.mock_tier1,
            tier2_engine=self.mock_tier2,
            threshold_words=6,
        )

    # ---- 1. Length & Script Resolution Tests -------------------------------
    def test_short_utterance_routes_to_tier1(self) -> None:
        """Utterances <= 6 words of standard MSA/English strictly route to 1.5b."""
        self.assertEqual(resolve_tier("مرحبا", "ar", 6), "1.5b")
        self.assertEqual(resolve_tier("كيف حالك اليوم؟", "ar", 6), "1.5b")
        self.assertEqual(resolve_tier("أين تذهب الآن؟", "ar", 6), "1.5b")
        self.assertEqual(resolve_tier("Hello, how are you?", "en", 6), "1.5b")
        self.assertEqual(resolve_tier("Yes, definitely.", "en", 6), "1.5b")

    def test_dialect_short_utterance_routes_to_tier2(self) -> None:
        """Colloquial phrases and dialect markers route to 7B even when <= 6 words."""
        # Terse colloquial idioms (< 6 words)
        self.assertEqual(resolve_tier("وين طالع في هالقايلة", "ar", 6), "7b")
        self.assertEqual(resolve_tier("وين رايح الحين؟", "ar", 6), "7b")
        self.assertEqual(resolve_tier("ما تشيل هم ابدا", "ar", 6), "7b")
        self.assertEqual(resolve_tier("شو القصة يا زلمة", "ar", 6), "7b")
        self.assertEqual(resolve_tier("عامل ايه النهارده", "ar", 6), "7b")
        # With explicit dialect metadata
        self.assertEqual(resolve_tier("وين رايح؟", "ar", 6, source_variant="SA"), "7b")
        self.assertEqual(resolve_tier("شو بتعمل؟", "ar", 6, source_variant="JO"), "7b")

    def test_boundary_conditions(self) -> None:
        """Boundary: exactly 6 words -> 1.5b, exactly 7 words -> 7b."""
        six_words = "واحد اثنان ثلاثة أربعة خمسة ستة"
        self.assertEqual(len(six_words.split()), 6)
        self.assertEqual(resolve_tier(six_words, "ar", 6), "1.5b")

        seven_words = "واحد اثنان ثلاثة أربعة خمسة ستة سبعة"
        self.assertEqual(len(seven_words.split()), 7)
        self.assertEqual(resolve_tier(seven_words, "ar", 6), "7b")

    def test_long_complex_utterance_routes_to_tier2(self) -> None:
        """Utterances > 6 words route to 7b for deep dialect/reasoning."""
        long_ar = "أنا لا أحب الطعام الصحي ولا أحب الأكل في هذا المطعم بشكل عام"
        self.assertGreater(len(long_ar.split()), 6)
        self.assertEqual(resolve_tier(long_ar, "ar", 6), "7b")

        long_en = "We need to finish this implementation and verify all tests before proceeding to deployment"
        self.assertGreater(len(long_en.split()), 6)
        self.assertEqual(resolve_tier(long_en, "en", 6), "7b")

    def test_cjk_character_threshold(self) -> None:
        """CJK scripts without whitespace use 2x character threshold (12 chars)."""
        short_cjk = "你好世界"  # 4 chars
        self.assertEqual(resolve_tier(short_cjk, "zh", 6), "1.5b")

        exact_12_cjk = "一二三四五六七八九十百千"  # 12 chars
        self.assertEqual(resolve_tier(exact_12_cjk, "zh", 6), "1.5b")

        thirteen_cjk = "一二三四五六七八九十百千万"  # 13 chars
        self.assertEqual(resolve_tier(thirteen_cjk, "zh", 6), "7b")

    def test_empty_text_safe_fallback(self) -> None:
        """Empty or blank text safely returns 1.5b without crashing."""
        self.assertEqual(resolve_tier("", "ar", 6), "1.5b")
        self.assertEqual(resolve_tier("   ", "en", 6), "1.5b")

    # ---- 2. Engine Batch Dispatch & Reassembly Tests -----------------------
    def test_single_short_item_dispatch(self) -> None:
        items = [("مرحبا بك", "ar", "en", "", "", "")]
        results = self.router.translate_batch(items)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tier, "1.5b")
        self.assertEqual(len(self.mock_tier1.call_history), 1)
        self.assertEqual(len(self.mock_tier2.call_history), 0)

    def test_single_long_item_dispatch(self) -> None:
        items = [("أنا مسافر غدا إلى مدينة الرياض لحضور المؤتمر السنوي للذكاء الاصطناعي", "ar", "en", "", "", "")]
        results = self.router.translate_batch(items)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].tier, "7b")
        self.assertEqual(len(self.mock_tier1.call_history), 0)
        self.assertEqual(len(self.mock_tier2.call_history), 1)

    def test_mixed_batch_preserves_exact_order(self) -> None:
        """Mixed batch containing both short and long items must be reassembled in identical order."""
        items = [
            ("أهلاً وسهلاً", "ar", "en", "", "", ""),  # short: 0 -> tier 1
            ("هذه جملة طويلة جدا تتكون من أكثر من ست كلمات للتجربة", "ar", "en", "", "", ""),  # long: 1 -> tier 2
            ("صباح الخير", "ar", "en", "", "", ""),  # short: 2 -> tier 1
            ("نحن نحاول اختبار قدرة السيرفر على ترجمة العبارات المعقدة والمتشعبة", "ar", "en", "", "", ""),  # long: 3 -> tier 2
            ("شكراً جزيلاً", "ar", "en", "", "", ""),  # short: 4 -> tier 1
        ]
        results = self.router.translate_batch(items)
        self.assertEqual(len(results), 5)

        # Check that outputs correspond to the exact input index
        self.assertEqual(results[0].tier, "1.5b")
        self.assertIn("أهلاً وسهلاً", results[0].text)

        self.assertEqual(results[1].tier, "7b")
        self.assertIn("هذه جملة طويلة", results[1].text)

        self.assertEqual(results[2].tier, "1.5b")
        self.assertIn("صباح الخير", results[2].text)

        self.assertEqual(results[3].tier, "7b")
        self.assertIn("نحن نحاول اختبار", results[3].text)

        self.assertEqual(results[4].tier, "1.5b")
        self.assertIn("شكراً جزيلاً", results[4].text)

        # Verify that tier1 received exactly 3 items and tier2 received exactly 2 items
        self.assertEqual(len(self.mock_tier1.call_history[0]), 3)
        self.assertEqual(len(self.mock_tier2.call_history[0]), 2)

    def test_metrics_incrementation(self) -> None:
        """Router increments mt_tier1_routed and mt_tier2_routed correctly."""
        metrics_mock = MagicMock()
        self.router.metrics = metrics_mock
        items = [
            ("نعم", "ar", "en", "", "", ""),
            ("هذه جملة طويلة تتجاوز الحد المسموح به للمستوى الأول تماما", "ar", "en", "", "", ""),
        ]
        self.router.translate_batch(items)
        metrics_mock.incr.assert_any_call("mt_tier1_routed", 1)
        metrics_mock.incr.assert_any_call("mt_tier2_routed", 1)

    # ---- 3. Cache Tier Isolation Tests ------------------------------------
    def test_cache_key_tier_isolation(self) -> None:
        """norm_hash with tier='1.5b' must never collide with tier='7b'."""
        text = "مرحبا كيف حالك اليوم"
        h_tier1 = norm_hash(text, source="ar", target="en", tier="1.5b")
        h_tier2 = norm_hash(text, source="ar", target="en", tier="7b")
        self.assertNotEqual(h_tier1, h_tier2, "Hashes between 1.5b and 7b must be strictly isolated")

    def test_server_resolve_item_tier_helper(self) -> None:
        """_resolve_item_tier in server.py delegates cleanly to two-tier engine."""
        self.assertEqual(_resolve_item_tier(self.router, "مرحبا بك", "ar"), "1.5b")
        self.assertEqual(
            _resolve_item_tier(self.router, "هذه جملة طويلة جدا تتجاوز ست كلمات قطعا", "ar"),
            "7b",
        )

        # Legacy / single engine fallback
        legacy_mock = MagicMock(spec=MtEngine)
        legacy_mock.name = "qwen_vllm"
        del legacy_mock.resolve_tier
        self.assertEqual(_resolve_item_tier(legacy_mock, "أي نص", "ar"), "qwen_vllm")

    # ---- 4. Factory & Configuration Tests ---------------------------------
    def test_build_mt_engine_two_tier_auto(self) -> None:
        settings = Settings(mt_backend="auto", mt_two_tier_enabled=True)
        engine = build_mt_engine(settings)
        self.assertIsInstance(engine, TwoTierMtEngine)
        self.assertEqual(engine.name, "two_tier")

    def test_build_mt_engine_two_tier_explicit(self) -> None:
        settings = Settings(mt_backend="two_tier", mt_two_tier_enabled=True)
        engine = build_mt_engine(settings)
        self.assertIsInstance(engine, TwoTierMtEngine)

    def test_info_structure(self) -> None:
        info = self.router.info()
        self.assertEqual(info["backend"], "two_tier")
        self.assertTrue(info["ready"])
        self.assertEqual(info["threshold_words"], 6)
        self.assertEqual(info["tier1"]["tier"], "1.5b")
        self.assertEqual(info["tier2"]["tier"], "7b")

    def test_two_tier_load_instantiates_without_frozen_error(self) -> None:
        """Verify that TwoTierMtEngine.load does not fail with FrozenInstanceError on Settings."""
        from unittest.mock import patch, MagicMock
        settings = Settings(mt_backend="two_tier", mt_two_tier_enabled=True)
        engine = TwoTierMtEngine(settings)
        with patch("app.mt.QwenCt2Engine") as mock_ct2, patch("app.mt.QwenVllmEngine") as mock_vllm, patch("app.mt.QwenHfEngine") as mock_hf, patch("app.mt.M2M100Ct2Engine") as mock_m2m:
            mock_inst = MagicMock()
            mock_inst.ready = True
            mock_inst.metrics = None
            mock_ct2.return_value = mock_inst
            mock_vllm.return_value = mock_inst
            mock_hf.return_value = mock_inst
            mock_m2m.return_value = mock_inst
            engine.load()
            self.assertIsNotNone(engine.tier1_engine)
            self.assertIsNotNone(engine.tier2_engine)


class TestTwoTierSabotages(unittest.TestCase):
    """Sabotage tests: mutations must cause tests to fail."""

    def setUp(self) -> None:
        self.settings = Settings(mt_two_tier_enabled=True, mt_two_tier_threshold_words=6)
        self.mock_tier1 = MockSubEngine(self.settings, "1.5B", "1.5b")
        self.mock_tier2 = MockSubEngine(self.settings, "7B", "7b")
        self.router = TwoTierMtEngine(
            self.settings,
            tier1_engine=self.mock_tier1,
            tier2_engine=self.mock_tier2,
            threshold_words=6,
        )

    def test_sabotage_threshold_lowered(self) -> None:
        """Sabotage S1: If threshold is mistakenly lowered to 2, 4-word utterance goes to 7b."""
        phrase = "أنا ذاهب إلى العمل"  # 4 words
        # Under normal router (threshold 6) -> 1.5b
        self.assertEqual(self.router.resolve_tier(phrase, "ar"), "1.5b")
        # Under sabotaged threshold (2) -> 7b
        sabotaged_router = TwoTierMtEngine(
            self.settings,
            tier1_engine=self.mock_tier1,
            tier2_engine=self.mock_tier2,
            threshold_words=2,
        )
        self.assertEqual(sabotaged_router.resolve_tier(phrase, "ar"), "7b")

    def test_sabotage_tier_stamping_present(self) -> None:
        """Sabotage S2: Returned results must carry non-empty tier field."""
        res = self.router.translate_batch([("مرحبا", "ar", "en", "", "", "")])
        self.assertEqual(res[0].tier, "1.5b", "Tier must be stamped on MtResult")


class TestIncidentFixesIntegration(unittest.TestCase):
    """Integration checks for incident fixes: dedup, driver preflight, and pipeline decoupling."""

    def test_cuda_driver_preflight_executes_safely(self) -> None:
        from app.mt import check_cuda_driver_preflight
        ok, reason = check_cuda_driver_preflight()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(reason, str)

    def test_pending_result_deduplication(self) -> None:
        """Concurrent submissions of identical requests reuse the same in-flight future."""
        import asyncio
        import time
        from app.scheduler import BatchScheduler

        call_count = 0

        def slow_worker(items):
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)  # simulate GPU latency so concurrent requests overlap
            return [f"result:{it}" for it in items]

        sched = BatchScheduler(
            name="test_dedup",
            runner=slow_worker,
            max_batch=8,
            wait_ms=5.0,
        )

        async def run_concurrent():
            await sched.start()
            try:
                # Submit 3 identical items concurrently
                t1 = asyncio.create_task(sched.submit("identical_query"))
                t2 = asyncio.create_task(sched.submit("identical_query"))
                t3 = asyncio.create_task(sched.submit("identical_query"))
                res1, res2, res3 = await asyncio.gather(t1, t2, t3)
                return res1, res2, res3
            finally:
                await sched.stop()

        r1, r2, r3 = asyncio.run(run_concurrent())
        self.assertEqual(r1[0], "result:identical_query")
        self.assertEqual(r2[0], "result:identical_query")
        self.assertEqual(r3[0], "result:identical_query")
        # In-flight deduplication: exactly 1 request executes the worker, other 2 get dedup=True
        dedup_flags = [r1[1].get("dedup"), r2[1].get("dedup"), r3[1].get("dedup")]
        self.assertEqual(dedup_flags.count(False), 1)
        self.assertEqual(dedup_flags.count(True), 2)
        self.assertEqual(call_count, 1, "Runner must only be called once for in-flight duplicates")


if __name__ == "__main__":
    unittest.main()
