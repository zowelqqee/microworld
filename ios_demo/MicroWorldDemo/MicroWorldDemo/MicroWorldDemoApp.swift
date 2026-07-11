import SwiftUI

@main
struct MicroWorldDemoApp: App {
    // The shipping app always uses the real embedded engine. Previews/tests use
    // the mock via their own initialisers.
    @StateObject private var viewModel = AppViewModel(
        engine: EmbeddedMicroWorldEngine(overlay: "promoted"),
        lazyModeSwitching: false  // iPhone 11 holds both modes (see TECHNICAL_DECISION.md §4)
    )

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
                .tint(Color(red: 0.30, green: 0.55, blue: 1.0))  // restrained accent
        }
    }
}
