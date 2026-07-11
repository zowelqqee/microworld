import SwiftUI

/// Debug-only diagnostics screen. Every value is measured on the running device;
/// dashes mean "not measured yet", never a fabricated number.
struct DiagnosticsView: View {
    let diagnostics: Diagnostics
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Startup") {
                    row("Engine cold start", ms(diagnostics.engineColdStartMs))
                    row("Creative warm-up", ms(diagnostics.creativeWarmMs))
                }
                Section("Latency — measured this session") {
                    row("First QA response", ms(diagnostics.firstQaMs))
                    row("First Creative response", ms(diagnostics.firstCreativeMs))
                    row("Steady-state QA", ms(diagnostics.lastQaMs))
                    row("Steady-state Creative", ms(diagnostics.lastCreativeMs))
                }
                Section("Memory — task_vm_info phys_footprint") {
                    row("Current", mb(diagnostics.currentMemoryMB))
                    row("Peak this session", mb(diagnostics.peakMemoryMB))
                }
                Section {
                    Text("These are live device measurements. App launch time and bundle size are recorded in DEVICE_BENCHMARK.md from Xcode Instruments / the built .app.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Diagnostics")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func ms(_ v: Double?) -> String { v.map { String(format: "%.1f ms", $0) } ?? "—" }
    private func mb(_ v: Double) -> String { v > 0 ? String(format: "%.0f MB", v) : "—" }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary).monospacedDigit()
        }
    }
}
