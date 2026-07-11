# MicroWorld — iOS demo (fully offline, on-device)

A small, native SwiftUI app that runs the **real** MicroWorld engine on an
iPhone — QA and Creative modes — with **no server, no API, no WebView, no
network**. The responses are produced by the actual `worldpgt` engine embedded
in the app, not canned.

- Architecture decision & feasibility: [`TECHNICAL_DECISION.md`](TECHNICAL_DECISION.md)
- Device performance template: [`DEVICE_BENCHMARK.md`](DEVICE_BENCHMARK.md)

---

## How it runs the real engine (in one picture)

```
SwiftUI  ──►  MicroWorldEngine (protocol)
              └─ EmbeddedMicroWorldEngine (Swift, off the main actor, measures latency)
                 └─ MWPythonBridge (Objective-C, owns one CPython interpreter, GIL-safe)
                    └─ import mw_ios → mw_ios.run(prompt, mode) → JSON
                       └─ worldpgt.assistant_surface.AnswerOrchestrator   ← the real engine
```

- The answer path of `worldpgt` is **pure-Python, stdlib-only** — proven in
  `TECHNICAL_DECISION.md` §1 by running it with numpy/pydantic/fastapi blocked.
  So we embed CPython and run the unmodified engine; outputs are identical to the
  CLI by construction.
- `mw_ios.py` is a thin adapter that honours the app's explicit QA/Creative
  toggle and preserves the engine's hard-safety screen. It does not reimplement
  anything.

---

## Prerequisites

- macOS with **Xcode 15+** (iOS 16 SDK or newer).
- [`xcodegen`](https://github.com/yonaskolb/XcodeGen): `brew install xcodegen`.
- A one-time build-time internet connection to download CPython-for-iOS. (The
  *app* is offline; only this build step needs the network.)
- An Apple Developer account/team for running on a physical device (a free
  personal team works for development installs).

---

## Build in four commands

```bash
cd ios_demo

# 1. Stage the real engine (worldpgt + mw_ios.py) into the app resources.
#    Strips numpy-only *.npy caches; result ≈ 7 MB.
scripts/stage_bundle.sh

# 2. Download a prebuilt CPython-for-iOS (Python.xcframework + python-stdlib).
#    Places them where the project expects. See notes below if a release moves.
scripts/fetch_python.sh

# 3. Generate the Xcode project from project.yml.
cd MicroWorldDemo
xcodegen generate

# 4. Open it.
open MicroWorldDemo.xcodeproj
```

Then in Xcode: select the **MicroWorldDemo** scheme, pick your iPhone, set your
signing **Team** (target → Signing & Capabilities), and **Run** (⌘R).

The build phase signs every compiled `python-stdlib/lib-dynload/*.so` with that
same development identity. If Xcode says no signing identity was supplied,
select a Team, then choose **Product → Clean Build Folder** and run again.

### What each command produces

| Path | By | Committed? |
|---|---|---|
| `MicroWorldDemo/MicroWorldDemo/Python/app_packages/` | `stage_bundle.sh` | no (generated) |
| `MicroWorldDemo/Frameworks/Python.xcframework` | `fetch_python.sh` | no (downloaded) |
| `MicroWorldDemo/MicroWorldDemo/Resources/python-stdlib/` | `fetch_python.sh` | no (downloaded) |
| `MicroWorldDemo/MicroWorldDemo.xcodeproj` | `xcodegen` | no (generated) |

All four are `.gitignore`d — they are reproducible from the committed source +
scripts.

### If `fetch_python.sh` can't find the pieces

Release layouts of
[Python-Apple-support](https://github.com/beeware/Python-Apple-support/releases)
occasionally change. If the script prints a "not found" note:

1. Download the `Python-3.13-iOS-support.*.tar.gz` for a `3.13` tag manually.
2. Copy `Python.xcframework` → `MicroWorldDemo/Frameworks/Python.xcframework`.
3. Copy the device stdlib (`Python.xcframework/ios-arm64/lib/python3.13`) →
   `MicroWorldDemo/MicroWorldDemo/Resources/python-stdlib`.
4. Re-run `xcodegen generate`.

The folder references (`app_packages`, `python-stdlib`) must be **blue folder
references** in Xcode (they preserve directory structure at runtime). XcodeGen
creates them via the `type: folder` entries in `project.yml`; verify in Xcode
that they show as blue folders, not yellow groups.

---

## Install on a physical iPhone 11

1. Connect the iPhone via USB; trust the Mac.
2. Xcode → the device appears in the run-destination menu; select it.
3. Target → **Signing & Capabilities** → choose your Team; Xcode auto-manages a
   provisioning profile. Bundle id defaults to `com.microworld.demo` — change it
   if it collides with an existing profile.
4. **⌘R** to build, install, and launch.
5. First launch: on the iPhone, **Settings → General → VPN & Device Management →
   Developer App → Trust** (only needed for a free/personal team).
6. There is **no network entitlement**; the app needs no permissions to run.

---

## Prove it's offline (do this on camera)

1. Launch the app; the header shows a green **Offline** pill.
2. Put the iPhone in **Airplane Mode** (Control Centre).
3. Tap the pill (or "Runs entirely on this iPhone") → the sheet **"All
   processing happens on this iPhone."** Tap **Run one QA + one Creative
   prompt** — both return output with the network off.
4. Ask a QA question and generate a Creative passage — both work in Airplane
   Mode.

Why it's genuinely offline (all grep-verifiable in this target):

- **No network code.** No `URLSession`/`URLRequest`/`NWConnection` anywhere —
  enforced by `NetworkAbsenceTests`.
- **No network entitlement**, strict ATS default (`Info.plist`).
- **Runtime guard.** `mw_ios.enforce_offline()` disables outbound sockets in the
  interpreter, so any accidental connect raises loudly instead of silently
  reaching out. The engine never triggers it (verified in Phase 0).

---

## Using the app

- **Mode** — segmented control (`QA` / `Creative`). The placeholder, button
  label, and preset chips change with the mode. The engine is told the mode
  explicitly; hard-safety questions (e.g. a private address) still audit even in
  Creative mode.
- **Prompt chips** — tap to run a preset. These are *inputs only*; responses are
  generated live.
- **Output card** — selectable text, a subtle appear animation, and a metrics
  row (`x ms · Local · Deterministic/Fresh`). Tap the row for the full metrics
  sheet.
- **Recent** — last 5 prompts this session (in memory only). Tap to rerun.
- **Debug builds** also show a **Demo** toggle (larger text/spacing, autocorrect
  off — clean for screen recording) and a **Diagnostics** button (live device
  measurements).

---

## The two engines behind the modes

The modes are two genuinely different engines, both embedded and offline:

- **QA** — the factual `worldpgt` runtime. Answers only from grounded memory;
  audits on a gap. Deterministic. ~42 MB, single-digit-ms per query.
- **Creative** — the `poetry_lab` narrative engine, generating from an English
  literary corpus (Shakespeare + 8 public-domain classics: Conan Doyle, Austen,
  Wilde, Verne, Stevenson, Wells). It runs a real **three-layer pipeline**
  (knowledge → discourse reasoning → speech) — the same shape as QA's synthesis
  path — so a prompt yields a varied several-sentence passage, not a template.
  "Describe X" leans descriptive; "Write a scene/story about X" leans toward
  events (the topic *doing* something). Full write-up:
  [`../poetry_lab/README.md`](../poetry_lab/README.md#the-three-layer-creative-generator).

Creative output is always labelled `[Creative mode — generated … not verified
fact.]`, re-rolls a fresh passage each run, and never recites a corpus 4-gram.
Hard-safety wins first: a private/current-sensitive prompt audits even under a
creative framing (`mw_ios._run_creative` runs the router's safety screen before
generating).

### On-device memory (the reason there are two artifacts)

The full narrative model parses to **~1 GB of Python dicts**, which jetsam-kills
the app on an iPhone 11. The build therefore ships a **slim artifact**
(`poetry_lab/artifacts/narrative_model.phone.json`, produced by
`python3 poetry_lab/cli.py slim-narrative`): it drops the verse-only /
novelty phrase tables and trims the 2.4 M-edge concept graph to each node's
strongest edges. Generation output is unchanged.

| | full (laptop) | slim (on device) |
|---|---:|---:|
| Artifact on disk | ~230 MB | ~24 MB |
| Creative resident (RSS) | ~1 GB | **~380 MB** |
| QA + Creative peak | (OOM) | **~400 MB** |
| Creative warm-up | ~7 s | **~0.8 s** |

`stage_bundle.sh` stages the slim artifact automatically (falling back to the
full one with a warning if the slim copy is absent).

---

## Tests

`MicroWorldDemoTests` covers the brief's list:

| Test | File |
|---|---|
| Engine initialization | `EngineTests.testEngineInitialization` |
| QA returns non-empty text | `EngineTests.testQAReturnsNonEmptyText` |
| Creative returns non-empty text | `EngineTests.testCreativeReturnsNonEmptyText` |
| Deterministic repeat (QA) | `EngineTests.testQADeterministicRepeat` |
| Creative flagged non-deterministic | `EngineTests.testCreativeFlaggedNonDeterministic` |
| No network dependency | `NetworkAbsenceTests` |
| Mode switching | `EngineTests.testModeSwitching`, `ViewModelTests.testModeChangeClearsResult` |
| Error handling | `EngineTests.testEmptyPromptThrows`, `ViewModelTests.testRunFailure…` |
| Latency measurement | `EngineTests.testLatencyMeasured`, `ViewModelTests.testLatencyRecorded…` |
| UI state transitions | `ViewModelTests` (boot → ready → running → ready, history cap, etc.) |

Run: **⌘U** in Xcode, or
`xcodebuild test -scheme MicroWorldDemo -destination 'platform=iOS,name=<device>'`.

`EngineTests` run against the **real embedded engine** when the Python framework
+ staged bundle are present (device or a Simulator built with resources); they
fall back to the mock only if the interpreter can't initialise, so the Swift
contract is always exercised.

---

## Files in this deliverable

```
ios_demo/
  TECHNICAL_DECISION.md          Phase-0 feasibility + architecture decision
  README_IOS.md                  this file
  DEVICE_BENCHMARK.md            on-device performance template + exact steps
  .gitignore
  scripts/
    stage_bundle.sh              stage worldpgt + mw_ios.py into app resources
    fetch_python.sh              download CPython-for-iOS (xcframework + stdlib)
  MicroWorldDemo/
    project.yml                  XcodeGen spec (generates the .xcodeproj)
    MicroWorldDemo/
      MicroWorldDemoApp.swift    @main entry; wires the real embedded engine
      Info.plist                 portrait-only, strict ATS, no network keys
      Models/MicroWorldEngine.swift   protocol, EngineMode, EngineResult, errors
      Engine/
        EmbeddedMicroWorldEngine.swift  real engine (measures latency, off-main)
        MockMicroWorldEngine.swift      preview/test double (clearly labelled)
        NetworkGuard.swift              offline facts + assertion
      Bridge/
        MWPythonBridge.h/.m             Objective-C CPython owner (GIL-safe)
        MicroWorldDemo-Bridging-Header.h
      Python/
        mw_ios.py                       the faithful adapter (staged into app_packages)
        app_packages/                   (generated) staged worldpgt + mw_ios.py
      State/
        AppViewModel.swift              phases, history, diagnostics
        MemoryReporter.swift            real phys_footprint via task_vm_info
        Presets.swift
      Views/                            Header, inputs, OutputCard, Metrics,
                                        OfflineInfo, History, Diagnostics, ContentView
      Support/Theme.swift
      Resources/Assets.xcassets, python-stdlib/ (generated)
    MicroWorldDemoTests/
      EngineTests.swift  ViewModelTests.swift  NetworkAbsenceTests.swift
```

---

## Known limitations

- **Python.xcframework is not vendored** (~40 MB binary). `fetch_python.sh`
  downloads it; that is the only build-time network step.
- **Creative uses the separate `poetry_lab` narrative engine**, loaded from the
  slim English literary artifact (see "The two engines" above). It plans a scene
  through three layers before realizing sentences; it does not reuse the factual
  wiki/Reddit word graph. The app warms this artifact in the background after
  launch so the first Creative tap is usually instant.
- **Creative quality is bounded by the corpus.** Anachronistic prompts (e.g.
  "a rocket") have no grounding in Shakespeare/Victorian prose and produce an
  empty/audited passage rather than an invented one — an honest data limit, not
  a bug. On-topic prompts (love, the sea, a storm, a murder, the moon, death)
  generate rich, varied output.
- **Creative is non-deterministic by design** (re-rolls a fresh passage each
  run). The deterministic guarantee applies to QA.
- The embedded bridge releases CPython's GIL after startup. This is essential:
  a serial GCD queue may run later calls on a different system thread, and
  retaining the startup GIL would make the first Ask/Generate call wait forever.
- The `.npy` embedding caches are intentionally excluded; the numpy path is
  optional and unused on-device (Phase 0). Exact outputs are unaffected.

---

## Screen-recording checklist

See `DEVICE_BENCHMARK.md` for the full recording flow; the short version:

1. Open app → show green **Offline** pill.
2. Turn on **Airplane Mode** (or show it already on).
3. QA: `Who founded SpaceX?`
4. QA follow-up: `What does SpaceX develop?`
5. Switch to **Creative**.
6. `Describe an evening in Moscow.` → show the passage + `[Creative mode…]` label.
7. Tap the metrics row → latency + Local.
8. Tap **Runs entirely on this iPhone** → the offline sheet; run the self-test.

Enable **Demo** (debug build) first for larger, recording-friendly text.
