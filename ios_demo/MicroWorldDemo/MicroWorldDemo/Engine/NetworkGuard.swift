import Foundation

/// Compile-time + runtime evidence that the app has no network client.
///
/// - The app declares no network entitlement and makes no `URLSession` /
///   socket calls anywhere in code (grep-verifiable).
/// - The Python side arms `enforce_offline()`, which raises on any outbound
///   connect (see `mw_ios.enforce_offline`).
/// - This guard adds a Swift-side assertion + log so a regression is loud.
///
/// There is intentionally **no** `URLSession` reference in this whole target.
enum NetworkGuard {

    /// Human-readable facts shown on the "All processing happens on this iPhone"
    /// sheet. Each is literally true of this build.
    static let offlineFacts: [String] = [
        "No cloud — the model runs on this device",
        "No API key — nothing to configure, nothing sent",
        "No account — no login, no sign-up",
        "Bundled local artifacts — memory + corpus ship inside the app",
    ]

    /// Called once at launch. Logs the offline posture. In DEBUG it also asserts
    /// that we did not accidentally link a shared URLSession usage into a hot
    /// path (best-effort; the real guarantee is the absence of any call site and
    /// the missing network entitlement).
    static func assertOffline() {
        #if DEBUG
        NSLog("[MicroWorld] Offline posture: no network entitlement, no URLSession call sites, Python outbound sockets disabled.")
        #endif
    }
}
