import Foundation

/// Production engine: embeds CPython (via `MWPythonBridge`) and runs the real
/// `worldpgt` engine through the `mw_ios` adapter. All interpreter work happens
/// off the main thread on a dedicated executor; the UI stays responsive.
///
/// Authoritative latency is measured *here in Swift* around the whole bridge
/// call, so the milliseconds shown to the user are real end-to-end numbers, not
/// the engine's self-report (which is surfaced separately in diagnostics).
///
/// Note on interop: the Objective-C bridge methods use the standard
/// `…error:(NSError **)` convention, so Swift imports them as `throws`. We map
/// any thrown `NSError` onto `EngineError.bridge`.
final class EmbeddedMicroWorldEngine: MicroWorldEngine, @unchecked Sendable {

    private let bridge = MWPythonBridge.shared()
    private let overlay: String
    // Serialises Swift-side access; the bridge itself is also serial. Using a
    // detached executor keeps CPython off the main actor.
    private let queue = DispatchQueue(label: "com.microworld.engine", qos: .userInitiated)

    init(overlay: String = "promoted") {
        self.overlay = overlay
    }

    func warmUp() async throws {
        try await onQueue {
            try Self.wrap { try self.bridge.initializeInterpreter() }
            let json = try Self.wrap {
                try self.bridge.warmUp(overlay: self.overlay, warmCreative: false)
            }
            _ = try Self.decodeObject(json)  // validates {"ok": true, ...}
        }
    }

    func warmCreative() async throws {
        try await onQueue {
            let json = try Self.wrap { try self.bridge.warmCreative() }
            _ = try Self.decodeObject(json)
        }
    }

    func run(prompt: String, mode: EngineMode) async throws -> EngineResult {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw EngineError.emptyPrompt }

        return try await onQueue {
            // ---- authoritative timing starts ----
            let clock = ContinuousClock()
            let start = clock.now
            let json = try Self.wrap {
                try self.bridge.run(prompt: trimmed, mode: mode.wireValue)
            }
            let elapsed = start.duration(to: clock.now)
            // ---- authoritative timing ends ----

            let obj = try Self.decodeObject(json)
            if let ok = obj["ok"] as? Bool, ok == false {
                throw EngineError.bridge((obj["error"] as? String) ?? "engine returned not-ok")
            }

            let latencyMs = Double(elapsed.components.seconds) * 1000.0
                + Double(elapsed.components.attoseconds) / 1e15

            return EngineResult(
                text: (obj["text"] as? String) ?? "",
                latencyMilliseconds: latencyMs,
                mode: mode,
                novelty: obj["novelty"] as? Double,
                deterministic: (obj["deterministic"] as? Bool) ?? (mode == .qa),
                decision: (obj["decision"] as? String) ?? "",
                route: (obj["route"] as? String) ?? "",
                supportKind: (obj["support_kind"] as? String) ?? "",
                riskFlags: (obj["risk_flags"] as? [String]) ?? [],
                engineMilliseconds: (obj["engine_ms"] as? Double) ?? 0
            )
        }
    }

    func selfTest() async throws -> SelfTestReport {
        try await onQueue {
            let json = try Self.wrap { try self.bridge.selfTest() }
            let obj = try Self.decodeObject(json)
            let rawItems = (obj["results"] as? [[String: Any]]) ?? []
            let items: [SelfTestReport.Item] = rawItems.map { r in
                let modeStr = (r["mode"] as? String) ?? "qa"
                return SelfTestReport.Item(
                    mode: EngineMode(rawValue: modeStr) ?? .qa,
                    prompt: (r["prompt"] as? String) ?? "",
                    ok: (r["ok"] as? Bool) ?? false,
                    decision: (r["decision"] as? String) ?? "",
                    characters: (r["chars"] as? Int) ?? 0
                )
            }
            return SelfTestReport(
                ok: (obj["ok"] as? Bool) ?? false,
                offlineEnforced: (obj["offline_enforced"] as? Bool) ?? false,
                items: items
            )
        }
    }

    // MARK: - Helpers

    private func onQueue<T>(_ work: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { cont in
            queue.async {
                do { cont.resume(returning: try work()) }
                catch { cont.resume(throwing: error) }
            }
        }
    }

    /// Runs a bridge call and maps a thrown NSError onto `EngineError.bridge`.
    private static func wrap<T>(_ body: () throws -> T) throws -> T {
        do { return try body() }
        catch let error as EngineError { throw error }
        catch { throw EngineError.bridge((error as NSError).localizedDescription) }
    }

    private static func decodeObject(_ json: String) throws -> [String: Any] {
        guard let data = json.data(using: .utf8) else {
            throw EngineError.decoding("non-utf8 response")
        }
        let obj = try JSONSerialization.jsonObject(with: data)
        guard let dict = obj as? [String: Any] else {
            throw EngineError.decoding("expected a JSON object")
        }
        return dict
    }
}
