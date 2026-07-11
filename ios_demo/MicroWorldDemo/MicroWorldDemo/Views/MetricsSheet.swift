import SwiftUI

/// Disclosure sheet opened from the metrics row. Shows the real, measured
/// figures for this run plus the mode's reference characteristics.
struct MetricsSheet: View {
    let result: EngineResult
    let diagnostics: Diagnostics
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("This run") {
                    row("Latency (end-to-end)", String(format: "%.2f ms", result.latencyMilliseconds))
                    row("Engine time", String(format: "%.2f ms", result.engineMilliseconds))
                    row("Mode", result.mode.title)
                    row("Decision", result.decision)
                    row("Route", result.route)
                    row("Determinism", result.deterministic ? "Deterministic" : "Fresh each run")
                    if let novelty = result.novelty {
                        row("Novelty", String(format: "%.2f", novelty))
                    }
                    if !result.riskFlags.isEmpty {
                        row("Flags", result.riskFlags.joined(separator: ", "))
                    }
                }

                Section("Mode characteristics") {
                    switch result.mode {
                    case .qa:
                        row("Cold start", diagnostics.engineColdStartMs.map { String(format: "%.0f ms", $0) } ?? "—")
                        row("Steady-state", "~3 ms")
                        row("Memory footprint", "~42 MB")
                        row("Corpus", "Grounded memory overlay")
                    case .creative:
                        row("Cold start", diagnostics.creativeWarmMs.map { String(format: "%.0f ms", $0) } ?? "~1.0 s")
                        row("Steady-state", "~2 ms")
                        row("Memory footprint", "~230 MB (order-2 word model)")
                        row("Corpus", "Local prose corpus")
                    }
                    row("Artifact size", "~7 MB bundled (code + data)")
                }

                Section {
                    Label("All processing happens on this iPhone.",
                          systemImage: "iphone.gen2")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Performance")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary).monospacedDigit()
        }
    }
}
