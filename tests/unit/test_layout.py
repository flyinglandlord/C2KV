from unittest import TestCase

from c2kv.core.layout import LayoutCompiler
from c2kv.core.masking import allowed_key_indices, build_allowed_attention, iter_allowed_key_ranges
from c2kv.core.types import ObjectiveKind, TokenRole, TokenizedExample
from c2kv.data.collator import build_teacher_layout


def build_plan(reconstruct=False):
    example = TokenizedExample(
        sample_id="paper-example",
        system_tokens=(10,),
        document_tokens=((11, 12, 13, 14), (21, 22)),
        query_tokens=(30,),
        response_tokens=(31,),
        reconstruction_document_ids=(0,) if reconstruct else (),
    )
    return LayoutCompiler(99, 98).compile(example, compression_ratio=2)


class LayoutCompilerTest(TestCase):
    def test_paper_names_and_positions_are_stable(self):
        plan = build_plan()
        self.assertEqual(plan.memory_indices, (3, 6, 9))
        self.assertEqual([plan.position_ids[i] for i in plan.memory_indices], [2, 4, 6])
        self.assertEqual(plan.token_roles[10], TokenRole.QUERY)
        self.assertEqual(plan.token_roles[11], TokenRole.RESPONSE)
        self.assertEqual(plan.objective_kinds[11], ObjectiveKind.LANGUAGE_MODEL)

    def test_documents_are_isolated(self):
        plan = build_plan()
        self.assertEqual(allowed_key_indices(plan, 7), (0, 7))
        self.assertNotIn(1, allowed_key_indices(plan, 7))

    def test_memory_slot_sees_sink_chunk_and_previous_memory(self):
        plan = build_plan()
        self.assertEqual(allowed_key_indices(plan, 6), (0, 1, 2, 3, 4, 5, 6))
        slot = plan.memory_slots[1]
        self.assertEqual(slot.compression_source_indices, (4, 5))
        self.assertEqual(slot.visible_document_indices, (1, 2, 4, 5))

    def test_query_sees_memory_but_not_raw_documents(self):
        plan = build_plan()
        visible = allowed_key_indices(plan, 10)
        self.assertEqual(visible, (0, 3, 6, 9, 10))
        self.assertNotIn(8, visible)

    def test_reconstruction_sees_only_its_document_memory(self):
        plan = build_plan(reconstruct=True)
        reconstruction = plan.reconstruction_spans[0]
        self.assertEqual(plan.token_roles[reconstruction.start], TokenRole.RECONSTRUCTION_MARKER)
        visible = allowed_key_indices(plan, reconstruction.start)
        self.assertIn(3, visible)
        self.assertIn(6, visible)
        self.assertNotIn(9, visible)

    def test_dense_and_sparse_reference_agree(self):
        plan = build_plan()
        dense = build_allowed_attention(plan)
        for query_index, row in enumerate(dense):
            from_ranges = set()
            for start, end in iter_allowed_key_ranges(plan, query_index):
                from_ranges.update(range(start, end))
            self.assertEqual(from_ranges, {index for index, allowed in enumerate(row) if allowed})

    def test_teacher_alignment_crosses_a_memory_boundary(self):
        example = TokenizedExample(
            sample_id="pretrain-example",
            system_tokens=(),
            document_tokens=((11, 12, 13, 14),),
            query_tokens=(),
            response_tokens=(21, 22),
        )
        plan = LayoutCompiler(99, 98).compile(example, compression_ratio=2)
        kept, teacher_index = build_teacher_layout(plan)
        first_response = plan.response_span.start
        student_predictor = first_response - 1
        teacher_target = kept.index(first_response)
        self.assertEqual(plan.token_roles[student_predictor], TokenRole.MEMORY)
        self.assertEqual(teacher_index[student_predictor], teacher_target - 1)
