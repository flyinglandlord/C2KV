import json
import tempfile
from pathlib import Path
from unittest import TestCase

from c2kv.data.adapters import JsonRecordStore


class JsonRecordStoreTest(TestCase):
    def test_jsonl_is_indexed_by_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(
                "\n".join(json.dumps({"id": index}) for index in range(3)) + "\n",
                encoding="utf-8",
            )
            records = JsonRecordStore(path)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[1], {"id": 1})
            self.assertIsInstance(records._entries[1], tuple)

    def test_json_object_and_list_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(json.dumps({"data": [{"id": "a"}, {"id": "b"}]}), encoding="utf-8")
            records = JsonRecordStore(path)
            self.assertEqual([records[index]["id"] for index in range(len(records))], ["a", "b"])
