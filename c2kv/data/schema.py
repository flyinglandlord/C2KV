"""One normalized schema for every C2KV training mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class C2KVExample:
    sample_id: str
    system: str
    documents: Tuple[str, ...]
    query: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any], fallback_id: str) -> "C2KVExample":
        if "messages" in record:
            return cls._from_messages(record, fallback_id)
        documents = record.get("documents", record.get("context", ()))
        if isinstance(documents, str):
            documents = (documents,)
        query = str(record.get("query", record.get("question", record.get("instruction", ""))))
        if not documents and query:
            documents = (query,)
            query = ""
        response = record.get("response", record.get("answer", record.get("output", "")))
        if isinstance(response, (list, tuple)):
            response = response[0] if response else ""
        example = cls(
            sample_id=str(record.get("sample_id", record.get("id", fallback_id))),
            system=str(record.get("system", record.get("system_prompt", ""))),
            documents=tuple(str(value) for value in documents if str(value).strip()),
            query=query,
            response=str(response),
            metadata=dict(record.get("metadata", {})),
        )
        example.validate()
        return example

    @classmethod
    def _from_messages(cls, record: Mapping[str, Any], fallback_id: str) -> "C2KVExample":
        messages = list(record["messages"])
        system_parts = [str(item.get("content", "")) for item in messages if item.get("role") == "system"]
        last_assistant = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "assistant"),
            None,
        )
        if last_assistant is None:
            raise ValueError("messages example requires an assistant response")
        last_user = next(
            (
                index
                for index in range(last_assistant - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            None,
        )
        if last_user is None:
            raise ValueError("messages example requires a user query before the response")
        history = [
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in messages[:last_user]
            if item.get("role") != "system" and str(item.get("content", "")).strip()
        ]
        documents = record.get("documents", history)
        if isinstance(documents, str):
            documents = [documents]
        if not documents:
            documents = [str(messages[last_user].get("content", ""))]
        example = cls(
            sample_id=str(record.get("sample_id", record.get("id", fallback_id))),
            system="\n".join(system_parts),
            documents=tuple(str(value) for value in documents),
            query=str(messages[last_user].get("content", "")),
            response=str(messages[last_assistant].get("content", "")),
            metadata=dict(record.get("metadata", {})),
        )
        example.validate()
        return example

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.documents or any(not value.strip() for value in self.documents):
            raise ValueError(f"sample {self.sample_id!r} requires non-empty documents")
        if not self.query.strip() and not self.response.strip():
            raise ValueError(f"sample {self.sample_id!r} has neither query nor response")
