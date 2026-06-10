from dataclasses import dataclass, field


# Default alias table: surface form -> canonical name.
# Extend via add_alias() or by passing a pre-populated normalizer to World.
_DEFAULT_ALIASES: dict[str, str] = {
    # яблоня
    "яблоню": "яблоня", "яблоне": "яблоня", "яблони": "яблоня",
    # яблоко
    "яблоки": "яблоко", "яблока": "яблоко", "яблоком": "яблоко",
    # семя
    "семени": "семя", "семенем": "семя", "семена": "семя", "семян": "семя",
    # курица
    "курицу": "курица", "курицы": "курица", "курице": "курица",
    # яйцо (neuter: nominative == accusative, but genitive/instrumental inflect)
    "яйца": "яйцо", "яйцом": "яйцо",
    # машина
    "машину": "машина", "машины": "машина", "машине": "машина",
    # прибыль
    "прибыли": "прибыль", "прибылью": "прибыль",
    # убыток
    "убытка": "убыток", "убытком": "убыток",
    # дым
    "дыма": "дым", "дымом": "дым",
    # лес
    "леса": "лес", "лесом": "лес",
    # вода
    "воду": "вода", "воды": "вода", "водой": "вода",
    # урожай
    "урожая": "урожай", "урожаю": "урожай", "урожаем": "урожай",
    # засуха
    "засухи": "засуха", "засухе": "засуха", "засухой": "засуха", "засуху": "засуха",
    # осадки
    "осадков": "осадки", "осадкам": "осадки", "осадками": "осадки",
    # инвестиции
    "инвестиций": "инвестиции", "инвестициям": "инвестиции",
    # производство
    "производства": "производство", "производству": "производство",
    # ресурсы
    "ресурсов": "ресурсы", "ресурсам": "ресурсы",
    # продукт
    "продукта": "продукт", "продукту": "продукт", "продуктом": "продукт",
    # прибыль already covered above
    # рост
    "росту": "рост", "роста": "рост", "ростом": "рост",
    # знание
    "знания": "знание", "знанию": "знание", "знанием": "знание",
    # книга
    "книгу": "книга", "книги": "книга", "книге": "книга",
}


class EntityNormalizer:
    """Maps inflected Russian surface forms to canonical (nominative) names."""

    def __init__(self) -> None:
        # surface -> canonical
        self._fwd: dict[str, str] = dict(_DEFAULT_ALIASES)
        # canonical -> list[surface]
        self._rev: dict[str, list[str]] = {}
        for surface, canonical in self._fwd.items():
            self._rev.setdefault(canonical, []).append(surface)

    def normalize(self, name: str) -> str:
        """Return canonical form; return *name* unchanged if not in table."""
        return self._fwd.get(name, name)

    def add_alias(self, surface: str, canonical: str) -> None:
        """Register a new surface -> canonical mapping."""
        self._fwd[surface] = canonical
        aliases = self._rev.setdefault(canonical, [])
        if surface not in aliases:
            aliases.append(surface)

    def get_aliases(self, canonical: str) -> list[str]:
        """Return all known surface forms for a canonical name."""
        return list(self._rev.get(canonical, []))
