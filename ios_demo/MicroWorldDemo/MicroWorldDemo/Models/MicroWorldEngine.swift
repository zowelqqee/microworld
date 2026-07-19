import Foundation

/// The mode the user has explicitly selected in the UI. The engine honours this
/// rather than inferring mode from phrasing (the CLI infers; a GUI toggle
/// should not) — hard-safety screening is still applied in either mode.
enum EngineMode: String, CaseIterable, Identifiable, Sendable {
    case auto
    case qa
    case creative

    var id: String { rawValue }

    var title: String {
        switch self {
        case .auto: return "Auto"
        case .qa: return "QA"
        case .creative: return "Creative"
        }
    }

    /// Value passed across the bridge to `mw_ios.run`.
    var wireValue: String { rawValue }

    var promptPlaceholder: String {
        switch self {
        case .auto: return "Ask anything…"
        case .qa: return "Ask a question…"
        case .creative: return "Describe what to write…"
        }
    }

    var primaryButtonTitle: String {
        switch self {
        case .auto: return "Ask"
        case .qa: return "Ask"
        case .creative: return "Generate"
        }
    }
}

/// The result of a single engine run. Every field is real — measured or returned
/// by the engine. Nothing is fabricated.
struct EngineResult: Equatable, Sendable {
    /// The answer / generated passage exactly as the engine produced it.
    let text: String
    /// Authoritative latency, measured in Swift around the whole bridge call.
    let latencyMilliseconds: Double
    let mode: EngineMode
    /// Creative gate is boolean (novelty pass/fail), not a numeric score, so this
    /// is `nil` — we do not invent a novelty number.
    let novelty: Double?
    /// QA is deterministic for a given prompt; Creative deliberately re-rolls a
    /// fresh passage each run, so this is `false` for Creative.
    let deterministic: Bool

    // Extra engine telemetry surfaced in the diagnostics / metrics sheet.
    let decision: String        // "answer" | "audit"
    let route: String           // e.g. "entity_relation", "creative_request"
    let supportKind: String
    let riskFlags: [String]
    /// Engine-side latency reported by Python (for diagnostics; `latencyMilliseconds`
    /// is the number shown to the user, since it includes the full round-trip).
    let engineMilliseconds: Double
}

/// Stable Swift-facing engine contract. The production implementation embeds
/// CPython and runs the real `worldpgt` engine; a mock implements the same
/// contract for previews and tests that must not spin up the interpreter.
protocol MicroWorldEngine: AnyObject, Sendable {
    /// Loads the engine once. Safe to call more than once (idempotent).
    func warmUp() async throws

    /// Optionally pre-builds the heavier Creative word-model so the first
    /// Creative run is instant. QA is usable without this.
    func warmCreative() async throws

    /// Runs one prompt in the given mode, off the main thread.
    func run(prompt: String, mode: EngineMode) async throws -> EngineResult

    /// Runs one QA + one Creative prompt and reports whether both produced text.
    func selfTest() async throws -> SelfTestReport
}

struct SelfTestReport: Equatable, Sendable {
    struct Item: Equatable, Sendable {
        let mode: EngineMode
        let prompt: String
        let ok: Bool
        let decision: String
        let characters: Int
    }
    let ok: Bool
    let offlineEnforced: Bool
    let items: [Item]
}

enum EngineError: LocalizedError {
    case notReady
    case emptyPrompt
    case bridge(String)
    case decoding(String)

    var errorDescription: String? {
        switch self {
        case .notReady:
            return "The on-device engine hasn't finished loading yet."
        case .emptyPrompt:
            return "Enter a prompt first."
        case .bridge(let message):
            return "Engine error: \(message)"
        case .decoding(let message):
            return "Couldn't read the engine's response: \(message)"
        }
    }
}
