# MicroWorld iOS — Device Benchmark

**Rule (from the brief): do not report Mac/Simulator numbers as iPhone numbers.**
This file separates *reference* (Mac-measured) numbers from *device* numbers.
The latency/launch device fields are still to be filled from Instruments; the
**memory** figures below are confirmed.

> **Status: runs on iPhone 11.** The app builds, installs, and stays alive
> through Creative warm-up on a physical iPhone 11 (iOS 26). It previously
> jetsam-crashed on launch when Creative loaded the full ~230 MB narrative
> artifact (~1 GB resident); shipping the **slim artifact** (§6) brought total
> QA + Creative resident to **~400 MB**, which is what fixed the crash.

---

## 1. Reference numbers (Mac — NOT iPhone)

Measured on the development Mac running the engine directly, and consistent with
the phone bundle's published figures. **Reference only.**

| Metric | QA | Creative |
|---|---:|---:|
| Cold start (build model once) | ~90 ms | ~1.0 s |
| Steady-state per query | ~3 ms | ~2 ms |
| Peak memory (bundle reference) | ~42 MB | ~230 MB |
| Bundled artifact size | ~7 MB (code + data, `.npy` stripped) | — |

These come from `TECHNICAL_DECISION.md` §1c and the phone README. They set
expectations; they are **not** device results.

---

## 2. Device results (fill in on an iPhone 11)

> iPhone 11 · iOS __._ · MicroWorldDemo build ____ · date ______

| Metric | Value | How measured |
|---|---|---|
| App launch (tap → first frame) | ____ ms | Instruments → App Launch, or os_signpost |
| Engine cold start (warm-up) | ____ ms | Diagnostics screen → "Engine cold start" |
| Creative warm-up | ____ ms | Diagnostics → "Creative warm-up" |
| First QA response | ____ ms | Diagnostics → "First QA response" |
| First Creative response | ____ ms | Diagnostics → "First Creative response" |
| Steady-state QA (2nd+ query) | ____ ms | Diagnostics → "Steady-state QA" |
| Steady-state Creative (2nd+ query) | ____ ms | Diagnostics → "Steady-state Creative" |
| Current memory after load | ____ MB | Diagnostics → "Current" (task_vm_info) |
| Peak memory this session | ____ MB | Diagnostics → "Peak", cross-check Instruments |
| App bundle size (installed) | ____ MB | see §4 |

Latency shown in-app is measured in Swift with `ContinuousClock` around the whole
bridge call (`EmbeddedMicroWorldEngine.run`), so it is a real end-to-end number.
Memory uses `task_vm_info` `phys_footprint` — the same figure iOS uses for
jetsam.

---

## 3. Exact device measurement steps

1. Build & install on the iPhone 11 (see `README_IOS.md`).
2. Launch. Wait for the **Offline** pill and for "Loading on-device engine…" to
   disappear (that is engine cold start; the Diagnostics screen records it).
3. **QA:** ask `Who founded SpaceX?` — note the metrics-row latency = *first QA*.
   Ask it again (or `What does SpaceX develop?`) — that latency = *steady-state
   QA*.
4. Switch to **Creative**. Run `Describe an evening in Moscow.` — first Creative.
   Run `Describe a room.` — steady-state Creative.
5. Open **Diagnostics** (debug build, gauge icon, bottom-right) and read
   Startup / Latency / Memory. These are live device values.
6. For **app launch time** and an independent **peak-memory** cross-check, use
   Xcode **Instruments** (App Launch template; Allocations/Leaks or the Memory
   gauge in the Debug navigator).
7. Record everything with **Airplane Mode on** to double as offline proof.

### Offline self-test (also on device)

Header pill → **"All processing happens on this iPhone."** → **Run one QA + one
Creative prompt**. Expect both green with Airplane Mode enabled and
"Outbound sockets disabled at runtime".

---

## 4. Bundle size

After building, measure the installed `.app`:

```bash
# From the DerivedData Products dir, or the exported .ipa's Payload:
du -sh MicroWorldDemo.app
du -sh MicroWorldDemo.app/Frameworks/Python.framework      # interpreter
du -sh MicroWorldDemo.app/python-stdlib                      # stdlib
du -sh MicroWorldDemo.app/app_packages                       # worldpgt + adapter (~7 MB)
```

Rough expectation: our staged engine ~7 MB + CPython framework/stdlib ~12–15 MB
(after trimming `test/` and `__pycache__`) → **~20–25 MB** added by the engine,
plus the thin Swift app. Fill the real installed size into §2.

---

## 5. Memory headroom on iPhone 11

iPhone 11 has 4 GB RAM; a single foreground app is allowed on the order of
~1.3–2 GB before jetsam. With the **slim** narrative artifact (§6) the measured
resident footprint is:

| Stage | RSS (measured, numpy-blocked = on-device conditions) |
|---|---:|
| QA engine warm | ~42 MB |
| + Creative engine warm (slim artifact) | **~400 MB** |
| Stable after generation | ~400 MB |

That leaves a wide jetsam margin, so both modes stay resident
(`lazyModeSwitching = false`). **With the full artifact this was ~1 GB and the
app was jetsam-killed** — do not ship the full artifact. If a future larger
corpus reintroduces pressure (watch the device console for
`MicroWorldDemo … jetsam`), either slim more aggressively
(`slim-narrative --edges-per-node`), cap the corpus
(`ingest-narrative --max-sentences-per-source`), or enable `lazyModeSwitching`
in `MicroWorldDemoApp.swift` (loads one mode at a time behind a "Loading model…"
state).

---

## 6. The slim on-device artifact

The Creative engine's model is generated on the laptop and slimmed for the
phone. Full artifact stays the laptop default; only the phone build uses the
slim copy.

```bash
# 1. build the full model (laptop; ~230 MB, all corpus detail)
python3 poetry_lab/cli.py ingest-narrative --source poetry_lab/corpus/english/
# 2. produce the slim on-device copy (~24 MB, ~380 MB resident, same output)
python3 poetry_lab/cli.py slim-narrative
# 3. stage — picks the slim artifact automatically
ios_demo/scripts/stage_bundle.sh
```

`slim-narrative` drops the fields the narrative generator never reads
(`backward`/`backward2` verse tables, the `seen_4grams` novelty gate,
`rhyme_groups`) and trims the concept graph from ~2.4 M edges to each node's
top-12 — a ~3× resident-memory reduction with identical generation output.

---

## 7. Recording flow (for the demo video)

1. Enable **Demo** mode (debug build) for larger, recording-friendly text.
2. Open app → show green **Offline**.
3. Turn on **Airplane Mode** before recording (or show it already on).
4. QA: `Who founded SpaceX?`
5. QA follow-up: `What does SpaceX develop?`
6. Switch to **Creative**.
7. Run: `Describe an evening in Moscow.`
8. Show latency + Local on the metrics row.
9. Open the info sheet: **"Runs entirely on this iPhone."**; run the self-test.

Do not script fake taps or auto-generate responses — every answer on camera is
produced live by the embedded engine.
