from typing import Any, Dict, List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


def chunk_document(
    document: Document, chunk_size: int = 900, chunk_overlap: int = 120
) -> List[Document]:
    """Chunk a single LangChain document into smaller documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents([document])
    return chunks


MAX_CHARS_PER_CHUNK = 6000


def documents_to_records(
    documents: Sequence[Document],
) -> List[Dict[str, Any]]:
    """Convert documents into Pinecone records with chunking applied.

    Each input document is expected to have at least the following metadata:
      - doc_id
      - source
      - title
      - url (optional)
      - published (optional)

    Output records follow the schema (logical representation):

    {
      "_id": "<doc_id>:<chunk_index>",
      "<text_field>": "<chunk>",  # PINECONE_TEXT_FIELD (default: 'chunk_text')
      "title": "...",
      "source": "...",
      "url": "...",
      "published": "...",
      "doc_id": "...",
      "chunk_id": <int>,
      ... additional metadata fields ...
    }
    """
    records: List[Dict[str, Any]] = []
    settings = get_settings()
    text_field = settings.PINECONE_TEXT_FIELD

    for document in documents:
        metadata = document.metadata or {}
        doc_id = metadata.get("doc_id")
        source = metadata.get("source")
        title = metadata.get("title", "")
        url = metadata.get("url", "")
        published = metadata.get("published", "")

        if not doc_id or not source:
            # Skip documents missing essential metadata
            continue

        base_metadata: Dict[str, Any] = {
            k: v
            for k, v in metadata.items()
            if k not in {"doc_id", "source", "title", "url", "published"}
        }

        chunks = chunk_document(document)
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.page_content or ""
            # Safety truncation for integrated embedding models like llama-text-embed-v2
            if len(chunk_text) > MAX_CHARS_PER_CHUNK:
                chunk_text = chunk_text[:MAX_CHARS_PER_CHUNK]

            record: Dict[str, Any] = {
                "_id": f"{doc_id}:{idx}",
                text_field: chunk_text,
                "title": title,
                "source": source,
                "url": url,
                "published": published,
                "doc_id": doc_id,
                "chunk_id": idx,
            }
            # Attach additional metadata fields
            record.update(base_metadata)
            records.append(record)

    return records