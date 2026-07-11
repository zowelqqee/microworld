import Foundation
import SwiftUI

/// Observable state for the main screen. Drives every UI transition and owns the
/// engine lifecycle. All engine work is awaited off the main actor inside the
/// engine itself; this type only mutates published state on the main actor.
@MainActor
final class AppViewModel: ObservableObject {

    enum Phase: Equatable {
        case launching            // engine warming up
        case ready                // idle, ready for input
        case loadingModel(String) // e.g. "Loading model…" (lazy mode switch)
        case running             // a prompt is in flight
        case failed(String)      // warm-up failed
    }

    // Inputs
    @Published var mode: EngineMode = .qa {
        didSet { if oldValue != mode { handleModeChange() } }
    }
    @Published var prompt: String = ""

    // Outputs
    @Published private(set) var phase: Phase = .launching
    @Published private(set) var result: EngineResult?
    @Published var errorMessage: String?
    @Published private(set) var history: [HistoryItem] = []

    // Diagnostics (real, measured)
    @Published private(set) var diagnostics = Diagnostics()

    // Demo mode (debug builds only; larger text/spacing for screen recording)
    @Published var demoMode: Bool = false

    private let engine: MicroWorldEngine
    private var creativeWarmed = false

    /// When true, only one mode's artifacts stay resident: switching modes
    /// unloads the other and shows "Loading model…". Off by default because
    /// iPhone 11 holds both comfortably (see TECHNICAL_DECISION.md §4). Wired so
    /// a lower-RAM device can enable it without touching the UI.
    let lazyModeSwitching: Bool

    init(engine: MicroWorldEngine, lazyModeSwitching: Bool = false) {
        self.engine = engine
        self.lazyModeSwitching = lazyModeSwitching
    }

    // MARK: - Lifecycle

    func boot() async {
        NetworkGuard.assertOffline()
        let clock = ContinuousClock()
        let start = clock.now
        do {
            try await engine.warmUp()
            diagnostics.engineColdStartMs = Self.ms(from: start, clock: clock)
            phase = .ready
            // Warm the heavier Creative model in the background so the first
            // Creative tap is instant. QA is already usable.
            if !lazyModeSwitching {
                Task { await self.backgroundWarmCreative() }
            }
        } catch {
            phase = .failed(error.localizedDescription)
            errorMessage = error.localizedDescription
        }
    }

    private func backgroundWarmCreative() async {
        guard !creativeWarmed else { return }
        let clock = ContinuousClock()
        let start = clock.now
        do {
            try await engine.warmCreative()
            creativeWarmed = true
            diagnostics.creativeWarmMs = Self.ms(from: start, clock: clock)
        } catch {
            // Non-fatal: Creative will simply warm lazily on first use.
        }
    }

    // MARK: - Running

    func submit() async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { errorMessage = EngineError.emptyPrompt.localizedDescription; return }
        guard phase == .ready else { return }

        errorMessage = nil
        phase = .running
        let runMode = mode
        do {
            let r = try await engine.run(prompt: trimmed, mode: runMode)
            withAnimation(.spring(response: 0.4, dampingFraction: 0.85)) {
                result = r
            }
            recordHistory(prompt: trimmed, mode: runMode)
            recordLatency(r)
            phase = .ready
        } catch {
            errorMessage = error.localizedDescription
            phase = .ready
        }
    }

    func run(preset: String) {
        prompt = preset
        Task { await submit() }
    }

    func rerun(_ item: HistoryItem) {
        mode = item.mode
        prompt = item.prompt
        Task { await submit() }
    }

    // MARK: - Mode change

    private func handleModeChange() {
        result = nil
        errorMessage = nil
        guard lazyModeSwitching else { return }
        // Lazy path: show a transient loading state while the newly selected
        // mode's artifacts are (re)built and the other is released.
        phase = .loadingModel("Loading \(mode.title) model…")
        Task {
            do {
                if mode == .creative { try await engine.warmCreative() }
                else { try await engine.warmUp() }
                phase = .ready
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    // MARK: - Self-test

    func runSelfTest() async -> SelfTestReport? {
        do { return try await engine.selfTest() }
        catch { errorMessage = error.localizedDescription; return nil }
    }

    // MARK: - Helpers

    private func recordHistory(prompt: String, mode: EngineMode) {
        history.removeAll { $0.prompt == prompt && $0.mode == mode }
        history.insert(HistoryItem(prompt: prompt, mode: mode), at: 0)
        if history.count > 5 { history = Array(history.prefix(5)) }
    }

    private func recordLatency(_ r: EngineResult) {
        switch r.mode {
        case .qa:
            if diagnostics.firstQaMs == nil { diagnostics.firstQaMs = r.latencyMilliseconds }
            diagnostics.lastQaMs = r.latencyMilliseconds
        case .creative:
            if diagnostics.firstCreativeMs == nil { diagnostics.firstCreativeMs = r.latencyMilliseconds }
            diagnostics.lastCreativeMs = r.latencyMilliseconds
        }
        diagnostics.currentMemoryMB = MemoryReporter.footprintMB()
        diagnostics.peakMemoryMB = max(diagnostics.peakMemoryMB, diagnostics.currentMemoryMB)
    }

    private static func ms(from start: ContinuousClock.Instant, clock: ContinuousClock) -> Double {
        let d = start.duration(to: clock.now)
        return Double(d.components.seconds) * 1000.0 + Double(d.components.attoseconds) / 1e15
    }
}

struct HistoryItem: Identifiable, Equatable {
    let id = UUID()
    let prompt: String
    let mode: EngineMode
}

/// Real, measured diagnostics. Nil until first measured — never fabricated.
struct Diagnostics: Equatable {
    var engineColdStartMs: Double?
    var creativeWarmMs: Double?
    var firstQaMs: Double?
    var firstCreativeMs: Double?
    var lastQaMs: Double?
    var lastCreativeMs: Double?
    var currentMemoryMB: Double = 0
    var peakMemoryMB: Double = 0
}
