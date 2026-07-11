import XCTest
@testable import MicroWorldDemo

/// Engine-contract tests.
///
/// These run against the REAL embedded engine when the target has the Python
/// framework + staged bundle available (physical device or a Simulator built
/// with the resources). When the interpreter cannot initialise (e.g. plain CI
/// without the framework), each test falls back to the mock so the suite still
/// exercises the Swift contract. `usingRealEngine` records which path ran.
final class EngineTests: XCTestCase {

    /// Returns the real engine if it can warm up, else the mock.
    private func makeEngine() async -> (MicroWorldEngine, Bool) {
        let real = EmbeddedMicroWorldEngine(overlay: "promoted")
        do {
            try await real.warmUp()
            return (real, true)
        } catch {
            return (MockMicroWorldEngine(), false)
        }
    }

    func testEngineInitialization() async throws {
        let (engine, _) = await makeEngine()
        // A second warm-up must be safe (idempotent).
        try await engine.warmUp()
    }

    func testQAReturnsNonEmptyText() async throws {
        let (engine, _) = await makeEngine()
        let r = try await engine.run(prompt: "Who founded SpaceX?", mode: .qa)
        XCTAssertFalse(r.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertEqual(r.mode, .qa)
    }

    func testCreativeReturnsNonEmptyText() async throws {
        let (engine, _) = await makeEngine()
        let r = try await engine.run(prompt: "Describe an evening in Moscow.", mode: .creative)
        XCTAssertFalse(r.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertEqual(r.mode, .creative)
    }

    func testQADeterministicRepeat() async throws {
        let (engine, real) = await makeEngine()
        let a = try await engine.run(prompt: "Who founded SpaceX?", mode: .qa)
        let b = try await engine.run(prompt: "Who founded SpaceX?", mode: .qa)
        XCTAssertEqual(a.text, b.text, "QA must be deterministic for the same prompt")
        XCTAssertTrue(a.deterministic)
        if real {
            // The real engine's canonical fact.
            XCTAssertTrue(a.text.contains("Elon Musk"))
        }
    }

    func testCreativeFlaggedNonDeterministic() async throws {
        let (engine, _) = await makeEngine()
        let r = try await engine.run(prompt: "Write a short scene about a rocket.", mode: .creative)
        XCTAssertFalse(r.deterministic, "Creative re-rolls a fresh passage each run")
    }

    func testModeSwitching() async throws {
        let (engine, _) = await makeEngine()
        let qa = try await engine.run(prompt: "What does SpaceX develop?", mode: .qa)
        let creative = try await engine.run(prompt: "Describe a room.", mode: .creative)
        XCTAssertEqual(qa.mode, .qa)
        XCTAssertEqual(creative.mode, .creative)
        XCTAssertNotEqual(qa.route, creative.route)
    }

    func testEmptyPromptThrows() async {
        let (engine, _) = await makeEngine()
        do {
            _ = try await engine.run(prompt: "   ", mode: .qa)
            XCTFail("expected EngineError.emptyPrompt")
        } catch let error as EngineError {
            if case .emptyPrompt = error {} else {
                XCTFail("expected emptyPrompt, got \(error)")
            }
        } catch {
            XCTFail("unexpected error \(error)")
        }
    }

    func testLatencyMeasured() async throws {
        let (engine, _) = await makeEngine()
        let r = try await engine.run(prompt: "Who founded SpaceX?", mode: .qa)
        XCTAssertGreaterThan(r.latencyMilliseconds, 0, "latency must be a real positive measurement")
        XCTAssertLessThan(r.latencyMilliseconds, 60_000, "sanity upper bound")
    }

    func testSelfTest() async throws {
        let (engine, _) = await makeEngine()
        let report = try await engine.selfTest()
        XCTAssertTrue(report.ok)
        XCTAssertEqual(report.items.count, 2)
    }
}
