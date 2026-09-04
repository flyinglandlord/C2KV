from unittest import TestCase

from c2kv.core.positions import concatenate_memory_positions


class PositionTest(TestCase):
    def test_independent_documents_are_repositioned_for_concatenation(self):
        positions = concatenate_memory_positions(((3, 7), (1, 5)), (8, 6), prefix_length=2)
        self.assertEqual(positions, (5, 9, 11, 15))
