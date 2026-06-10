from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorldObject:
    id: str
    name: str
    kind: Optional[str] = None
    properties: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        kind_str = f"[{self.kind}]" if self.kind else ""
        return f"WorldObject({self.name!r}{kind_str})"
