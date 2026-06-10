from dataclasses import dataclass, field


@dataclass
class Relation:
    source: str
    relation_type: str
    target: str
    evidence: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Relation({self.source!r} --{self.relation_type}--> {self.target!r})"
