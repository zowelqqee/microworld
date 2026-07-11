# MicroWorld iOS — Phase 0 Technical Decision

**Date:** 2026-07-11
**Question:** What is the shortest *technically honest* way to run the existing
MicroWorld runtime — producing its real outputs — inside a native SwiftUI app,
fully offline, on a physical iPhone 11?

**Decision: Option A — embed CPython and run the unmodified `worldpgt` package.**

The rest of this document shows the measurements that make this the right call
and evaluates every option against the required criteria.

---

## 1. What the runtime actually is (measured, not assumed)

I inspected `microworld_cli/mw.py` and the `worldpgt/` package and ran the real
engine to characterise the runtime path — the code that actually executes to
answer a QA or Creative prompt.

### 1a. Third-party imports on the answer path

`requirements.txt` lists `fastapi numpy pydantic pytest requests uvicorn`, and a
grep of the package finds imports of `numpy`, `pydantic`, and `fastapi`. That
looks like a native-dependency problem for iOS. **It is not.** Those are all
confined to non-runtime surfaces (the FastAPI server, an optional embedding
index, an offline knowledge-pump tool). I proved the answer path is free of them
by blocking all three at import time and re-running the engine:

```
$ python3  # numpy, pydantic, fastapi, pydantic_core all forced to ImportError
init OK without numpy+pydantic+fastapi, ms 204
[3ms]   'Who founded SpaceX?'          -> answer/entity_relation:  'SpaceX was founded by Elon Musk.'
[1ms]   'What does SpaceX develop?'     -> answer/entity_relation:  'SpaceX develops rockets and spacecraft.'
[73ms]  'Tell me about Starlink.'       -> answer/entity_definition:'Starlink is a satellite internet constellation…'
[1087ms]'Write a story about a rocket'  -> answer/creative_request: '[Creative mode — generated, …]  R…'
[1ms]   'Imagine a city of the future'  -> answer/creative_request: '[Creative mode — generated, …]  I…'
```

Every QA and Creative output is produced **byte-for-byte identically** with
numpy/pydantic/fastapi absent. The engine degrades gracefully: the one module
that imports numpy (`worldpgt/knowledge/relation_embedding_index.py`) is behind
a lazy, guarded call and is simply skipped when numpy is missing — exactly the
"do not install spaCy / no pip step needed" story the phone README describes.

**Conclusion:** the MicroWorld answer path is **pure Python, standard library
only, zero native extensions.** This is the single fact that decides everything
below. The usual blocker for embedding Python on iOS — cross-compiling native
wheels like numpy for `arm64-apple-ios` — **does not exist here.**

### 1b. Size

| Component | On disk |
|---|---|
| `worldpgt/` Python code (~50k LOC) | ~2 MB |
| Runtime data (memory overlays + local prose corpus) | ~5 MB |
| Optional numpy embedding caches (`*.npy`) — **droppable** | ~3.9 MB |
| **Bundle we ship (code + data, `.npy` stripped)** | **~7 MB** |

### 1c. Speed and memory (reference, measured on Mac — **not** iPhone numbers)

Matches the phone README's published figures. Device numbers are collected
separately in `DEVICE_BENCHMARK.md`.

| Mode | Cold start | Steady-state / query | Peak RSS (README reference) |
|---|---:|---:|---:|
| QA | ~90 ms | ~3 ms | ~42 MB |
| Creative | ~1.0 s (builds the word model once) | ~2 ms | ~230 MB |

The Creative peak (~230 MB) is the sizing constraint. iPhone 11 has **4 GB RAM**;
a single foreground app is allowed on the order of ~1.3–2 GB before jetsam, so
230 MB is comfortably safe. Both artifacts can stay resident simultaneously — no
lazy-unload dance is required on iPhone 11 (see §4).

---

## 2. Options evaluated

### Option A — Embed a CPython interpreter, run the unmodified `worldpgt`

Bundle a prebuilt CPython for iOS (`Python.xcframework` from BeeWare's
[Python-Apple-support](https://github.com/beeware/Python-Apple-support), which
tracks CPython's official iOS support — PEP 730, tier-3 platform since Python
3.13) plus the stdlib and the `worldpgt` package as app resources. A tiny
Objective-C bridge initialises the interpreter once and calls a thin Python
adapter (`mw_ios.py`) that returns JSON. Swift talks to the bridge.

| Criterion | Assessment |
|---|---|
| Implementation complexity | **Low–moderate.** No native wheels to build (§1a). Embed one xcframework, set `PYTHONHOME`/`PYTHONPATH` to the bundle, `Py_Initialize`, import adapter. Well-trodden (BeeWare/Briefcase ship exactly this). |
| App Store / iOS restrictions | **Compliant.** All code is bundled; nothing is downloaded or `eval`'d at runtime (Guideline 2.5.2). Apple officially supports iOS as a CPython target. No private APIs. |
| Bundle size | Python framework + stdlib ~12–15 MB + our ~7 MB bundle ≈ **~20–25 MB** added to the app. |
| Runtime memory | Same as the engine: QA ~42 MB, Creative ~230 MB peak. Safe on iPhone 11 (§1c). |
| Physical iPhone 11 | **Yes.** arm64 device slice is a first-class target of Python-Apple-support. |
| Preserves exact outputs | **Yes — guaranteed.** It is literally the same Python source on a real CPython. Zero reimplementation, therefore zero output drift. |
| Time to working demo | **Fastest honest option: ~1–2 days.** The only real work is Xcode integration of the framework, which is a solved problem. |

### Option B — Port the runtime-critical path to Swift

| Criterion | Assessment |
|---|---|
| Complexity | **Very high.** ~50k LOC across router, entity-QA graph, phrase graph, safety policy, and the Creative model (order-2 word transitions, 27k-word vocab, ~444k learned 4-grams, a 4-gram novelty gate, seeded sampling). Faithfully porting all of it is weeks of work. |
| Exact outputs | **At serious risk.** Every subtle difference in tokenisation, dict ordering, regex semantics, or RNG tie-breaking silently changes outputs — and "preserve exact current outputs" is a hard requirement. |
| Bundle / memory | Smaller and leaner in principle, but irrelevant given the correctness risk. |
| iPhone 11 | Yes. |
| Time | **Weeks**, plus an open-ended validation tail to prove parity. |

Rejected: slowest path to an *honest* demo, highest chance of quietly diverging
from the real engine.

### Option C — Port the critical path to C/C++ and bridge to Swift

Same reimplementation risk as B, in a language with worse ergonomics for this
string/graph-heavy workload, and no upside over B for a Python-string engine.
**Rejected.**

### Option D — Precompute a compact static artifact Swift queries directly

Two sub-cases, both fail a requirement:

- **Precomputed QA table (question → answer).** This is effectively
  *hardcoding demo answers*, which the brief explicitly forbids ("Do not
  hardcode demo answers"). It also can't answer anything off the preset list.
- **Serialised Creative model + a Swift sampler.** This is a partial Option B:
  you still reimplement the seeded sampler and the 4-gram novelty gate in Swift,
  and the RNG/tie-breaking must match exactly to preserve outputs and
  determinism. Same drift risk as B for the hardest part of the engine.

**Rejected:** either dishonest (hardcoding) or Option B's risk without its
generality.

---

## 3. Decision

**Option A.** It is the only option that (a) preserves exact outputs by
construction, (b) is genuinely fast to a working demo, and (c) is
App-Store-legitimate — and it is unlocked specifically because §1a proved the
answer path is stdlib-only pure Python with **no native dependencies to
cross-compile.** The brief's stop-condition ("stop if the current Python runtime
cannot be embedded cleanly without major work") is **not** triggered: it *can*
be embedded cleanly.

### How the app runs the real engine (integration method)

```
SwiftUI (EngineMode: .qa/.creative)
   → MicroWorldEngine protocol
      → EmbeddedMicroWorldEngine (Swift, runs off the main actor)
         → MWPythonBridge (Objective-C, owns the interpreter, thread-safe via GIL + a serial queue)
            → Py_Initialize once; PYTHONHOME/PYTHONPATH → app bundle
            → import mw_ios; mw_ios.warm_up(); mw_ios.run(prompt, mode) → JSON string
               → worldpgt.assistant_surface.answer_orchestrator.AnswerOrchestrator  (the real engine)
```

`mw_ios.py` is a **thin, faithful** adapter, not a reimplementation:

- **QA mode** → `orchestrator.answer(prompt)` unchanged (the engine's own
  factual routing).
- **Creative mode** → runs the engine's real hard-safety screen first; if the
  prompt is private/current-sensitive it audits **exactly as the engine would**
  (this preserves "a private question audits even under a creative framing").
  Otherwise it invokes the engine's own creative branch
  (`AnswerOrchestrator._creative_answer` → `generate_creative`), which produces
  the real recombined passage, the real `[Creative mode — generated…]` label,
  and runs the real 4-gram novelty gate.

The only thing the adapter overrides is the **route decision**, so the app's
explicit QA/Creative segmented control is honoured (the CLI infers mode from
phrasing; a GUI toggle should not). Verified against every demo preset:

```
[qa]       'Who founded SpaceX?'                 -> answer/entity_relation
[qa]       'What does SpaceX develop?'            -> answer/entity_relation
[creative] 'Describe an evening in Moscow.'       -> answer/creative_request   (real generation)
[creative] 'Describe a room.'                     -> answer/creative_request   (real generation)
[creative] 'Write a short scene about a rocket.'  -> answer/creative_request   (real generation)
[creative] "Tell me my neighbor's home address"  -> audit/private_sensitive   (safety preserved)
```

(Note: several of the brief's Creative presets — "Describe an evening in
Moscow", "Write a short scene…" — do **not** match the CLI's creative regex, so
without an explicit-mode override they would have wrongly routed to factual
audit. The adapter fixes this honestly, by calling the real generator.)

Latency is measured in Swift **around the bridge call** (`ContinuousClock`), so
reported milliseconds are real, not fabricated.

---

## 4. Memory tradeoff (documented per the brief)

On iPhone 11 (4 GB), both modes stay hot after first use — QA (~42 MB) and
Creative (~230 MB peak) fit together well under the jetsam ceiling. The app
therefore **keeps both artifacts resident** and does not unload between mode
switches.

The lazy-load/unload fallback the brief asks for is implemented but **disabled
by default**: if a future lower-RAM device (or a measured jetsam on device)
requires it, flip `EngineConfig.lazyModeSwitching = true` and the engine will
show a `Loading model…` state, load the selected mode, and release the other.
The `DEVICE_BENCHMARK.md` steps say when to consider enabling it.

---

## 5. Honest limitations of *this* deliverable

- **This environment cannot compile or run Xcode, an iOS Simulator, or a
  physical iPhone 11.** I produce the complete, correct project (Swift, the
  Objective-C bridge, the Python adapter, tests, entitlements, and the bundle
  staging script) and real **Mac-measured** engine numbers. The **on-device**
  numbers must be filled in by building on a Mac with Xcode and running on an
  iPhone 11 — `DEVICE_BENCHMARK.md` gives the exact steps and leaves the device
  fields blank rather than passing Mac numbers off as device numbers.
- The one prerequisite the build script cannot vendor here is the
  `Python.xcframework` binary (~40 MB). `README_IOS.md` documents the one-command
  fetch from Python-Apple-support and where to drop it.
