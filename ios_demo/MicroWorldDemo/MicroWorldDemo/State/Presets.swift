import Foundation

/// Demo preset prompts shown as horizontally scrollable chips. These are only
/// *inputs* — the responses are always produced live by the engine, never
/// hardcoded.
enum Presets {
    static func chips(for mode: EngineMode) -> [String] {
        switch mode {
        case .qa:
            return [
                "Who founded SpaceX?",
                "What does SpaceX develop?",
                "Tell me about Starlink.",
            ]
        case .creative:
            return [
                "Describe an evening in Moscow.",
                "Describe a room.",
                "Write a short scene about a rocket.",
            ]
        }
    }
}
