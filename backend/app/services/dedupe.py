from typing import Any, Dict, List, Set

from app.core.logging import get_logger

logger = get_logger(__name__)


def dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate records in-memory based on the `_id` field.

    If `_id` is missing on a record, it is always retained.
    """
    seen_ids: Set[str] = set()
    deduped: List[Dict[str, Any]] = []

    for record in records:
        record_id = record.get("_id")
        if record_id is None:
            deduped.append(record)
            continue

        if record_id in seen_ids:
            logger.debug("Skipping duplicate record id=%s in current batch", record_id)
            continue

        seen_ids.add(record_id)
        deduped.append(record)

    return deduped