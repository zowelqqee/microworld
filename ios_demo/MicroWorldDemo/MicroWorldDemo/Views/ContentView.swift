import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var vm: AppViewModel
    @Environment(\.colorScheme) private var scheme

    @State private var showInfo = false
    @State private var showMetrics = false
    @State private var showDiagnostics = false

    var body: some View {
        ZStack {
            backgroundGradient.ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: Theme.spacing) {
                    HeaderView(showInfo: { showInfo = true })

                    ModeSelector(mode: $vm.mode)

                    PresetChips(presets: Presets.chips(for: vm.mode)) { preset in
                        vm.run(preset: preset)
                    }

                    PromptEditor(
                        text: $vm.prompt,
                        placeholder: vm.mode.promptPlaceholder,
                        demoMode: vm.demoMode
                    )

                    PrimaryButton(
                        title: vm.mode.primaryButtonTitle,
                        running: vm.phase == .running,
                        enabled: isReady
                    ) {
                        Task { await vm.submit() }
                    }

                    statusArea

                    if let result = vm.result {
                        OutputCard(result: result, demoMode: vm.demoMode) {
                            showMetrics = true
                        }
                        .id(result.text)  // re-trigger appear animation per result
                    }

                    HistoryView(history: vm.history) { vm.rerun($0) }

                    footer
                }
                .padding(.horizontal, 20)
                .padding(.top, 8)
                .padding(.bottom, 40)
                .frame(maxWidth: 620)  // keeps it tidy if ever run on iPad
                .frame(maxWidth: .infinity)
            }
            .scrollDismissesKeyboard(.interactively)
        }
        .sheet(isPresented: $showInfo) {
            OfflineInfoView(runSelfTest: { await vm.runSelfTest() })
        }
        .sheet(isPresented: $showMetrics) {
            if let r = vm.result {
                MetricsSheet(result: r, diagnostics: vm.diagnostics)
            }
        }
        .sheet(isPresented: $showDiagnostics) {
            DiagnosticsView(diagnostics: vm.diagnostics)
        }
        .task { await vm.boot() }
    }

    private var isReady: Bool {
        if case .ready = vm.phase { return true }
        return false
    }

    @ViewBuilder private var statusArea: some View {
        switch vm.phase {
        case .launching:
            loadingBar("Loading on-device engine…")
        case .loadingModel(let label):
            loadingBar(label)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle")
                .font(.footnote)
                .foregroundStyle(.red)
                .padding(12)
                .cardSurface(corner: 12)
        case .ready, .running:
            if let error = vm.errorMessage {
                Label(error, systemImage: "exclamationmark.circle")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        }
    }

    private func loadingBar(_ text: String) -> some View {
        HStack(spacing: 10) {
            ProgressView()
            Text(text).font(.subheadline).foregroundStyle(Theme.subtle(scheme))
            Spacer()
        }
        .padding(12)
        .cardSurface(corner: 12)
    }

    private var footer: some View {
        HStack {
            Button {
                showInfo = true
            } label: {
                Label("Runs entirely on this iPhone", systemImage: "lock.shield")
                    .font(.caption)
            }
            .buttonStyle(.plain)
            .foregroundStyle(Theme.subtle(scheme))
            Spacer()
            #if DEBUG
            HStack(spacing: 14) {
                Toggle(isOn: $vm.demoMode) {
                    Text("Demo").font(.caption)
                }
                .toggleStyle(.button)
                .controlSize(.small)
                Button {
                    showDiagnostics = true
                } label: {
                    Image(systemName: "speedometer")
                }
                .foregroundStyle(Theme.subtle(scheme))
            }
            #endif
        }
        .padding(.top, 8)
    }

    private var backgroundGradient: some View {
        LinearGradient(
            colors: scheme == .dark
                ? [Color(white: 0.04), Color(white: 0.08)]
                : [Color(white: 0.98), Color(white: 0.94)],
            startPoint: .top, endPoint: .bottom
        )
    }
}

#Preview("Ready — Mock") {
    let vm = AppViewModel(engine: MockMicroWorldEngine())
    return ContentView().environmentObject(vm)
}
