import SwiftUI

@main
struct MicroWorldDemoApp: App {
    var body: some Scene {
        WindowGroup {
            AppRootView()
                .tint(Color(red: 0.30, green: 0.55, blue: 1.0))  // restrained accent
        }
    }
}

/// Shows a native first frame before constructing the CPython bridge.
///
/// `EmbeddedMicroWorldEngine` is intentionally created only after SwiftUI has
/// committed this lightweight view.  That keeps an iPhone from appearing to
/// remain on an empty launch screen while native framework initialisation is
/// scheduled, and it does not alter the engine, planner, or bundled graph.
private struct AppRootView: View {
    @State private var startEngine = false

    var body: some View {
        Group {
            if startEngine {
                EngineContentView()
            } else {
                LaunchingView()
            }
        }
        .task {
            // Yield once so the first SwiftUI frame is visibly committed before
            // the embedded CPython bridge is allocated.
            await Task.yield()
            startEngine = true
        }
    }
}

private struct EngineContentView: View {
    // The shipping app always uses the real embedded engine. Previews/tests use
    // the mock via their own initialisers.
    @StateObject private var viewModel = AppViewModel(
        engine: EmbeddedMicroWorldEngine(overlay: "promoted"),
        lazyModeSwitching: false  // iPhone 11 holds both modes (see TECHNICAL_DECISION.md §4)
    )

    var body: some View {
        ContentView().environmentObject(viewModel)
    }
}

private struct LaunchingView: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(white: 0.04), Color(white: 0.08)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 12) {
                Text("MicroWorld")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                ProgressView()
                    .tint(.white)
                Text("Preparing on-device engine…")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.65))
            }
            .foregroundStyle(.white)
        }
    }
}
