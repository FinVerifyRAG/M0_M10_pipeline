"""
ingest/version_graph/query.py
------------------------------
Free-function API for the VersionGraph.

Phase 5: These functions now delegate to the VersionGraph instance methods
(history, current_version) added in build.py.  Kept for backward compatibility.
Prefer calling ``graph.history(section_id)`` and
``graph.current_version(section_id, date)`` directly.
"""
from typing import Optional, List
from ingest.version_graph.build import VersionGraph, VersionNode


def in_force(graph: VersionGraph, section_id: str, date: str) -> Optional[str]:
    """
    Return the doc_id version of section_id valid on `date`, else None.

    Phase 5: delegates to graph.current_version() which is the canonical
    implementation.  Kept as a free function for callers that already use it.
    """
    return graph.current_version(section_id, date)


def history(graph: VersionGraph, section_id: str) -> List[VersionNode]:
    """
    Return all historical versions of section_id sorted by effective_from
    (ascending).

    Phase 5: delegates to graph.history() which is the canonical implementation.
    """
    return graph.history(section_id)
