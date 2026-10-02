"""
ingest/version_graph/build.py
------------------------------
M1: Version graph for tracking supersession, amendment, and renumbering of
regulatory sections across document revisions.

Phase 5 additions
-----------------
- VersionGraph.history(section_id) — instance method, mirrors query.history().
- VersionGraph.current_version(section_id, date) — in-force doc_id on a date.
- VersionGraph.superseded_sections() — list of section_ids with any supersession.
- VersionGraph.amendment_events(since_date) — structured event list for drift.
"""
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field

logger = logging.getLogger("ingest.version_graph")


@dataclass
class VersionNode:
    """One version of a regulatory section / document."""
    doc_id:         str
    section_id:     str
    effective_from: str
    effective_to:   Optional[str] = None
    supersedes:     List[str] = field(default_factory=list)
    amends:         List[str] = field(default_factory=list)
    regulator:      Optional[str] = None   # Phase 5: for drift event filtering

    def __post_init__(self):
        # Guard: dataclass default_factory is fine but handle None from JSON
        if self.supersedes is None:
            self.supersedes = []
        if self.amends is None:
            self.amends = []

    @property
    def is_superseded(self) -> bool:
        return self.effective_to is not None

    @property
    def has_supersession(self) -> bool:
        return bool(self.supersedes)


class VersionGraph:
    """
    DAG of regulatory section versions.

    Nodes are keyed by ``section_id`` (e.g. ``SEBI_Chapter1_Reg52``).
    Each key maps to a list of VersionNode objects in chronological order.
    """

    def __init__(self):
        self.nodes: Dict[str, List[VersionNode]] = {}

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_node(self, node: VersionNode) -> None:
        self.nodes.setdefault(node.section_id, []).append(node)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        data = {k: [asdict(n) for n in v] for k, v in self.nodes.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved version graph: %d sections → %s", len(self.nodes), filepath)

    @classmethod
    def load(cls, filepath: str) -> "VersionGraph":
        graph = cls()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                graph.nodes[k] = [VersionNode(**nd) for nd in v]
            logger.info("Loaded version graph: %d sections from %s", len(graph.nodes), filepath)
        except FileNotFoundError:
            logger.warning("Version graph file not found: %s — starting empty.", filepath)
        return graph

    # ── Phase 5: Query methods (instance-level, mirrors query.py free-functions)

    def history(self, section_id: str) -> List[VersionNode]:
        """
        Return all versions of ``section_id`` in chronological order
        (ascending by effective_from).

        This is the instance-level equivalent of ``query.history(graph, section_id)``.
        Using the instance method is preferred; the free function in query.py
        is kept for backward compatibility.
        """
        versions = self.nodes.get(section_id, [])
        return sorted(versions, key=lambda v: v.effective_from)

    def current_version(self, section_id: str, date: str) -> Optional[str]:
        """
        Phase 5: Return the doc_id of the version of ``section_id`` that is
        in force on ``date`` (ISO YYYY-MM-DD), or None if none is found.

        This is the instance-level equivalent of ``query.in_force(graph, section_id, date)``.
        """
        versions = self.history(section_id)
        for v in reversed(versions):
            if v.effective_from <= date and (v.effective_to is None or date <= v.effective_to):
                return v.doc_id
        return None

    def superseded_sections(self) -> List[str]:
        """
        Phase 5: Return section_ids that have at least one superseded version
        (effective_to is set or supersedes list is non-empty).

        Used by the drift experiment to identify sections where value drift
        or supersession drift may have occurred.
        """
        result = []
        for sec_id, versions in self.nodes.items():
            if any(v.is_superseded or v.has_supersession for v in versions):
                result.append(sec_id)
        return result

    def amendment_events(self, since_date: Optional[str] = None) -> List[Dict]:
        """
        Phase 5: Produce a structured list of amendment/supersession events
        suitable for ``guarantee.drift.version_graph_trigger``.

        Each event dict has keys: date, regulator, sections, type.
        Type is one of: amends, supersedes, renumbers, adds.

        Parameters
        ----------
        since_date : optional ISO date — only include events on or after this date.

        Returns
        -------
        List of event dicts sorted by date ascending.
        """
        events: List[Dict] = []
        for sec_id, versions in self.nodes.items():
            for v in versions:
                # Supersession event: this version supersedes an older one
                if v.supersedes and (since_date is None or v.effective_from >= since_date):
                    events.append({
                        "date":      v.effective_from,
                        "regulator": v.regulator or sec_id.split("_")[0],
                        "sections":  [sec_id] + list(v.supersedes),
                        "type":      "supersedes",
                        "doc_id":    v.doc_id,
                    })

                # Amendment event: this version amends (but does not replace) another
                if v.amends and (since_date is None or v.effective_from >= since_date):
                    events.append({
                        "date":      v.effective_from,
                        "regulator": v.regulator or sec_id.split("_")[0],
                        "sections":  [sec_id] + list(v.amends),
                        "type":      "amends",
                        "doc_id":    v.doc_id,
                    })

        events.sort(key=lambda e: e["date"])
        logger.debug("Generated %d amendment events (since=%s)", len(events), since_date)
        return events

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"VersionGraph(sections={len(self.nodes)})"


# ── Factory function ──────────────────────────────────────────────────────────

def build_graph_from_chunks(chunks: List[Any]) -> VersionGraph:
    """
    Build a VersionGraph from a list of Chunk objects.

    Constructs one VersionNode per chunk using:
    - section_id : regulator_chapter_section
    - doc_id     : chunk.chunk_id
    - effective_from : chunk.effective_from or chunk.issue_date
    - supersedes / amends from chunk.metadata

    Phase 5: also stores chunk.regulator on each node for drift event filtering.
    """
    graph = VersionGraph()
    for chunk in chunks:
        meta       = getattr(chunk, "metadata", {}) or {}
        regulator  = getattr(chunk, "regulator", "UNKNOWN")
        chapter    = meta.get("chapter", "General")
        section    = meta.get("section", "General")
        sec_id     = f"{regulator}_{chapter}_{section}"
        doc_id     = chunk.chunk_id
        eff_from   = getattr(chunk, "effective_from", None) or getattr(chunk, "issue_date", "")
        eff_to     = getattr(chunk, "effective_to", None)
        supersedes = meta.get("supersedes", [])
        amends     = meta.get("amends", [])

        node = VersionNode(
            doc_id=doc_id,
            section_id=sec_id,
            effective_from=eff_from,
            effective_to=eff_to,
            supersedes=supersedes,
            amends=amends,
            regulator=regulator,
        )
        graph.add_node(node)

    logger.info("Built version graph: %d sections from %d chunks", len(graph), len(chunks))
    return graph
