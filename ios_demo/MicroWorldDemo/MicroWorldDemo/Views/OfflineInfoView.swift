import SwiftUI

/// "All processing happens on this iPhone." — the offline-proof sheet.
struct OfflineInfoView: View {
    var runSelfTest: () async -> SelfTestReport?
    @Environment(\.dismiss) private var dismiss
    @State private var report: SelfTestReport?
    @State private var testing = false

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(spacing: 10) {
                        Image(systemName: "wifi.slash")
                            .font(.system(size: 40, weight: .light))
                            .foregroundStyle(Theme.onlineGreen)
                        Text("All processing happens on this iPhone.")
                            .font(.headline)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .listRowBackground(Color.clear)
                }

                Section("Why this is offline") {
                    ForEach(NetworkGuard.offlineFacts, id: \.self) { fact in
                        Label(fact, systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.primary)
                            .symbolRenderingMode(.palette)
                            .foregroundStyle(.primary, Theme.onlineGreen)
                    }
                }

                Section("Offline self-test") {
                    Button {
                        Task {
                            testing = true
                            report = await runSelfTest()
                            testing = false
                        }
                    } label: {
                        HStack {
                            Text("Run one QA + one Creative prompt")
                            Spacer()
                            if testing { ProgressView() }
                        }
                    }
                    if let report {
                        Label(report.ok ? "Both produced output on-device"
                                        : "Self-test did not pass",
                              systemImage: report.ok ? "checkmark.seal.fill" : "xmark.seal.fill")
                            .foregroundStyle(report.ok ? Theme.onlineGreen : .red)
                        ForEach(report.items, id: \.prompt) { item in
                            HStack {
                                Text("\(item.mode.title): \(item.prompt)")
                                    .font(.footnote)
                                Spacer()
                                Text("\(item.characters) chars · \(item.decision)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        Label(report.offlineEnforced
                                ? "Outbound sockets disabled at runtime"
                                : "Offline guard not armed",
                              systemImage: "lock.shield")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section {
                    Text("Turn on Airplane Mode and run the self-test — the answers still appear, because nothing here talks to a server.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Runs on this iPhone")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.large])
    }
}
