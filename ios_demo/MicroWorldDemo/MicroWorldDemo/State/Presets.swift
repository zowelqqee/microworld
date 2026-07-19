import Foundation

/// Demo preset prompts shown as horizontally scrollable chips. These are only
/// *inputs* — the responses are always produced live by the engine, never
/// hardcoded.
enum Presets {
    static func chips(for mode: EngineMode) -> [String] {
        switch mode {
        case .auto:
            return ["Who founded SpaceX?", "What if Elon Musk had not founded SpaceX?", "Why might SpaceX and Blue Origin be related?", "Write about SpaceX using only these facts", "Tell a fictional story about a space company"]
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
