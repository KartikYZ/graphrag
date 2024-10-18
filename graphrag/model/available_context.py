"""A package containing the 'AvailableContext' model."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AvailableContext:
    """A protocol for aggregated trimmed context information during indexing."""

    source_community_id: str = ""
    """Community ID this context information belongs to."""

    index: int | None = None
    """Index (inclusive) in the edge list beyond which the context is trimmed."""

    node_details: dict[str, dict] | None = None
    """Map of node identifier strings to node details dictionary"""

    edges: list[dict] | None = None
    """List of ALL edges in the context for this community"""

    attributes: dict[str, Any] | None = None
    """A dictionary of additional attributes associated with the trimmed context (optional)"""

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        source_community_id_key: str = "source_community_id",
        index_key: str = "index",
        node_details_key: str = "node_details",
        edges_key: str = "edges",
        attributes_key: str = "attributes",
    ) -> "AvailableContext":
        """Create a new available context from the dict data."""
        return AvailableContext(
            source_community_id=d[source_community_id_key],
            index=d.get(index_key),
            node_details=d.get(node_details_key),
            edges=d.get(edges_key),
            attributes=d.get(attributes_key),
        )
