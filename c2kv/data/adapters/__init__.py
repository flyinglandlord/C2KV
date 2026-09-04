"""Built-in C2KV dataset adapters."""

from .jsonl import JsonRecordStore, iter_records

__all__ = ["JsonRecordStore", "iter_records"]
