# MicroWorld iOS v2 — device run log

Date: 2026-07-16

## Build and bundle state

- `stage_bundle.sh` completed using the current root `worldpgt/` package, not
  the stale `microworld_cli/worldpgt` fallback.
- Staged bundle: 36 MB; `.npy` files remaining: 0.
- Bundled v2 serving graph: 1,266 items / 990 relations.
- iOS Simulator architecture build: passed.
- The exact staged Python layout passed all three demo prompts locally with
  third-party answer-path dependencies import-blocked. This is a bundled-host
  smoke, **not** an iPhone latency measurement.

## Physical iPhone 11

- Device discovered: `iPhone Арсений`, iPhone 11 (`iPhone12,1`), paired and
  available.
- Device launch/run: **not completed**.
- Reason: the project declares development team `5NBHFY736Q`, while the only
  local Apple Development signing identity belongs to team `FDB4UCG4MN`.
  The generic iOS build is intentionally unsigned and the phone rejected it
  with `0xe800801c (No code signature found)`. A provisioning build with the
  available identity did not produce an installable `.app`.

No on-device answer or latency is claimed. In particular, no Instruments value
and no informal latency value is recorded.

## Required final manual device step

In Xcode, select the signing team that owns the paired iPhone and run the
`MicroWorldDemo` scheme. Then enable Airplane Mode and run the three exact
prompts in `demo_script.md`; record the displayed Swift latency values here as
informal device observations if Instruments is not being used.
