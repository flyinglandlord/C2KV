from unittest import TestCase

from c2kv.core.compression import CompressionSchedule


class CompressionScheduleTest(TestCase):
    def test_fixed_schedule_uses_first_ratio(self):
        schedule = CompressionSchedule((4, 8, 16), mode="fixed", seed=7)
        self.assertEqual(schedule.ratio_for("sample", epoch=99), 4)

    def test_deterministic_schedule_is_reproducible(self):
        schedule = CompressionSchedule((2, 4, 8), seed=11)
        first = schedule.ratio_for("paper-example", epoch=3)
        self.assertEqual(first, schedule.ratio_for("paper-example", epoch=3))
        self.assertIn(first, schedule.ratios)
