import Foundation

/// A deterministic, interpreter-free engine for SwiftUI previews and for unit
/// tests that assert UI state transitions without booting CPython.
///
/// It is clearly labelled as a mock and is only wired up in previews / tests —
/// the shipping app always uses `EmbeddedMicroWorldEngine`. This is not a way to
/// fake the demo; it exists so the view layer can be exercised in isolation.
final class MockMicroWorldEngine: MicroWorldEngine, @unchecked Sendable {

    var warmUpDelay: Duration = .milliseconds(10)
    var runDelay: Duration = .milliseconds(5)
    var shouldFail = false

    func warmUp() async throws {
        try? await Task.sleep(for: warmUpDelay)
        if shouldFail { throw EngineError.bridge("mock warm-up failure") }
    }

    func warmCreative() async throws {
        try? await Task.sleep(for: warmUpDelay)
    }

    func run(prompt: String, mode: EngineMode) async throws -> EngineResult {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw EngineError.emptyPrompt }
        if shouldFail { throw EngineError.bridge("mock run failure") }

        let clock = ContinuousClock()
        let start = clock.now
        try? await Task.sleep(for: runDelay)
        let elapsed = start.duration(to: clock.now)
        let latencyMs = Double(elapsed.components.seconds) * 1000.0
            + Double(elapsed.components.attoseconds) / 1e15

        let text: String
        switch mode {
        case .auto:
            text = "[mock] Auto-routed answer for “\(trimmed)”."
        case .qa:
            text = "[mock] Answer for “\(trimmed)”."
        case .creative:
            text = "[Creative mode — generated, recombined from learned text, not verified fact.]\n\n[mock] A recombined passage seeded by “\(trimmed)”."
        }

        return EngineResult(
            text: text,
            latencyMilliseconds: latencyMs,
            mode: mode,
            novelty: nil,
            deterministic: mode != .creative,
            decision: "answer",
            route: mode == .creative ? "creative_request" : "entity_relation",
            supportKind: mode == .creative ? "creative_generated" : "semi_stable_relation",
            riskFlags: mode == .creative ? ["creative_generated"] : [],
            engineMilliseconds: latencyMs
        )
    }

    func selfTest() async throws -> SelfTestReport {
        SelfTestReport(
            ok: true,
            offlineEnforced: true,
            items: [
                .init(mode: .qa, prompt: "Who founded SpaceX?", ok: true,
                      decision: "answer", characters: 32),
                .init(mode: .creative, prompt: "Describe a room.", ok: true,
                      decision: "answer", characters: 238),
            ]
        )
    }
}
