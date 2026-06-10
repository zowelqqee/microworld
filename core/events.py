from dataclasses import dataclass


@dataclass
class Event:
    subject: str
    verb: str
    object: str
    raw_text: str

    def __repr__(self) -> str:
        return f"Event({self.subject!r} --{self.verb}--> {self.object!r})"
