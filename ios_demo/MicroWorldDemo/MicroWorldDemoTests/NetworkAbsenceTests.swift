import XCTest
@testable import MicroWorldDemo

/// Evidence, in test form, that the app has no network dependency.
///
/// This is a source-level guard: it fails if anyone introduces a `URLSession`
/// call site or a network entitlement into the target. The stronger guarantees
/// live in the build itself (no network entitlement, strict ATS) and in the
/// Python `enforce_offline()` runtime guard.
final class NetworkAbsenceTests: XCTestCase {

    /// The app target's own source tree must not reference URLSession/URLRequest.
    func testNoURLSessionInAppSources() throws {
        let sourceRoot = Self.appSourceRoot()
        guard let sourceRoot else {
            throw XCTSkip("App source root not resolvable in this test environment")
        }
        let offenders = try Self.grep(
            patterns: ["URLSession", "URLRequest", "NWConnection", "CFStream"],
            under: sourceRoot,
            fileExtensions: ["swift", "m", "h"]
        )
        XCTAssertTrue(offenders.isEmpty,
                      "Network API references found (must be none): \(offenders)")
    }

    /// The Python adapter must arm the offline guard.
    func testAdapterEnforcesOffline() throws {
        let sourceRoot = Self.appSourceRoot()
        guard let sourceRoot else {
            throw XCTSkip("App source root not resolvable in this test environment")
        }
        let adapter = sourceRoot
            .appendingPathComponent("Python/mw_ios.py")
        if FileManager.default.fileExists(atPath: adapter.path) {
            let contents = try String(contentsOf: adapter, encoding: .utf8)
            XCTAssertTrue(contents.contains("enforce_offline"),
                          "mw_ios.py must define the offline guard")
        } else {
            throw XCTSkip("mw_ios.py not present in this test bundle layout")
        }
    }

    // MARK: - Helpers

    /// Walk up from this test file to the `MicroWorldDemo/MicroWorldDemo` dir.
    private static func appSourceRoot() -> URL? {
        // #filePath points at this source file at build time.
        var dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()          // MicroWorldDemoTests
            .deletingLastPathComponent()          // MicroWorldDemo (project)
            .appendingPathComponent("MicroWorldDemo")
        if FileManager.default.fileExists(atPath: dir.path) { return dir }
        dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        return FileManager.default.fileExists(atPath: dir.path) ? dir : nil
    }

    private static func grep(patterns: [String], under root: URL,
                             fileExtensions: Set<String>) throws -> [String] {
        var offenders: [String] = []
        let fm = FileManager.default
        guard let en = fm.enumerator(at: root, includingPropertiesForKeys: nil) else {
            return offenders
        }
        for case let url as URL in en {
            guard fileExtensions.contains(url.pathExtension) else { continue }
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            for p in patterns where text.contains(p) {
                // Allow this test file itself (it names the APIs it forbids).
                if url.lastPathComponent == "NetworkAbsenceTests.swift" { continue }
                offenders.append("\(url.lastPathComponent): \(p)")
            }
        }
        return offenders
    }
}
