# Архитектурный анализ MicroWorld перед экспериментом

Дата: 2026-07-09. Цель анализа — выделить минимальное переиспользуемое ядро
и отделить QA-специфичные допущения перед заменой источника знаний на корпус
русской поэзии.

## Контракт рантайма (production)

```text
Text -> Semantic Structures -> Semantic Reasoning -> Speech Plan -> Language Renderer -> Answer
```

Хранилище — JSON-артефакты (overlay-файлы, кэши, отчёты). Контракт
семантический, а не storage-специфичный (см. `docs/architecture.md`).

## Классификация модулей

### 1. Core engine — переносимые архитектурные механизмы

| Механизм | Где живёт в production | Что именно переносится |
|---|---|---|
| Типизированный семантический граф + распространение активации | `worldpgt/cognition/semantic_thought_graph.py` (`_GraphBuilder`, `_activate`: max-propagation, damping 0.35, 3 раунда) | построение графа концептов и выбор «ходов» по активации |
| Частотный граф фраз («крошечная локальная языковая модель») | `worldpgt/cognition/phrase_graph.py` (`PhraseGraph`: Counter-фрагменты, переходы, `_counter_best`) | обучение переходов из корпуса, детерминированный обход |
| Детерминированный seeded-выбор вариантов | `phrase_graph._seeded_weighted_pick`, `symbolic_text_generator._stable_index` (sha256 от seed) | один и тот же prompt всегда даёт один и тот же текст, разные — разные |
| Слоистый конвейер и явные трассы | `cognition/types.py` (ReasoningTrace), `verbalization_engine.py` | план генерации — явный инспектируемый артефакт, не скрытое состояние |
| JSON-артефакты как граница слоёв | `worldpgt/experiments/*.json`, `worldpgt/artifacts/` | ингест пишет артефакты; рассуждение и язык читают только их |

### 2. Knowledge ingestion — заменяется целиком

- `worldpgt/knowledge/` — wiki-ингест, нормализация фактов, overlay-провайдеры;
- `worldpgt/knowledge_pump/` — yield/precision gates, frontier;
- `worldpgt/wiki_snapshot_ingestion/`, `worldpgt/wiki_snapshots/`;
- `worldpgt/relation_extraction_v2/` — извлечение типизированных отношений
  (subject–predicate–object) из энциклопедического текста;
- `worldpgt/community_context/` — Reddit-паттерны (style-only);
- `worldpgt/web_search/` — волатильный live-поиск.

Все они предполагают, что знание = source-qualified факт-триплет. Для поэзии
это допущение неверно: знание = образы, сочетаемость, ритм, рифма.

### 3. Reasoning layer — сохраняется по принципу работы

- `worldpgt/cognition/semantic_thought_graph.py` — активация → выбор moves;
- `worldpgt/cognition/working_memory.py`, `thought_loop.py`, `reasoning_engine.py`;
- `worldpgt/reasoning/fact_graph.py`, `pattern_discovery.py` — поиск
  структурных регулярностей в собственном графе (детерминированно, без ML);
- `worldpgt/multihop_qa/relation_graph.py`, `path_planner.py` — многошаговый
  обход отношений.

Принцип: рассуждение — это детерминированные операции над явным графом
(активация, обход, комбинация), а не генерация. Именно это переносится.

### 4. Language realization — сохраняется по принципу работы

- `worldpgt/cognition/phrase_graph.py` — генерация связного текста обходом
  выученного частотного графа фрагментов/переходов;
- `worldpgt/cognition/verbalization_engine.py` — вербализация трасс;
- `worldpgt/entity_qa/symbolic_text_generator.py` — пошаговая генерация
  речевых единиц из явного состояния (buckets, priors, seeded-выбор);
- `worldpgt/entity_qa/semantic_speech_planner.py` — план речи до рендера.

Принцип: язык учится по корпусу (фрагменты + переходы + частоты), рендер —
детерминированный обход с seeded-вариативностью. Переносится напрямую.

### 5. QA-специфичная логика — удаляется из эксперимента

- Парсинг вопросов и интентов: `entity_qa/semantic_question_parser.py`,
  `entity_question_analyzer.py`, `multihop_qa/question_analyzer.py`,
  `cross_page_qa/*`, `query_engine/` (Find/Filter/Count/Compare);
- Роутинг и surface: `assistant_surface/` (router, orchestrator, styles);
- **Support gate / отказ «I don't know»**: `cognition/support_guard.py`,
  `entity_qa/entity_answer_validator.py`, `*_validator.py`, decision
  no/audit/answer — центральное QA-допущение «неподдержанное = запрещено».
  В творческом режиме заменяется на противоположное: неподдержанные
  *комбинации* поддержанных элементов разрешены;
- Флаги `factual_support_allowed`, карантин/промоушен памяти,
  `staleness_detector`, temporal policy;
- Диалог/кореференция: `worldpgt/dialogue/`;
- `worldpgt/api/`.

## Отображение слоёв на эксперимент (poetry_lab)

| Слой production | Эксперимент | Перенесённый механизм |
|---|---|---|
| wiki/reddit ингест → overlay JSON | `poemcore/ingest.py` → `artifacts/*.json` | корпус → семантические структуры, JSON-граница |
| accepted memory (факт-триплеты) | граф концептов: co-occurrence, эпитеты, рифмы + фразовая модель + профили стиля | те же типизированные узлы/рёбра, но отношения поэтические |
| `_activate` + выбор moves | `poemcore/concept_graph.py` (порт `_activate`) + `poemcore/planner.py` (поэтические moves: establish/develop/leap/turn/closure) | тот же алгоритм активации |
| speech plan → phrase graph traversal | `PoemPlan` (строка = семантическая цель + рифменная группа + слоговая цель) → `poemcore/generator.py` | план — явный артефакт, рендер — обход частотного графа |
| `_seeded_weighted_pick` | `poemcore/phrase_model.py` (порт) | детерминизм и вариативность |
| support guard («не знаю») | novelty guard (запрет воспроизводить корпусные 4-граммы) | инверсия гейта: вместо запрета нового — запрет заученного |

Research question: может ли то же ядро (граф + активация + частотный
языковой слой + детерминированные seeded-решения) производить связный
творческий текст после замены только источника знаний. Ответ — в README
после прогона eval-скриптов.
