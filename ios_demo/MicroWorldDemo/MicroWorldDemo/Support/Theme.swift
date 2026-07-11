import SwiftUI

/// Minimal, native-feeling design tokens. Clean, premium, slightly futuristic —
/// no fake-terminal aesthetic, no heavy gradients. Adapts to light/dark.
enum Theme {
    static let corner: CGFloat = 16
    static let cardCorner: CGFloat = 20
    static let spacing: CGFloat = 16

    /// A restrained accent used sparingly (button, active segment, status dot).
    static let accent = Color.accentColor

    static let onlineGreen = Color(red: 0.20, green: 0.78, blue: 0.35)

    static func card(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color(white: 0.11)
            : Color(white: 1.0)
    }

    static func cardStroke(_ scheme: ColorScheme) -> Color {
        scheme == .dark
            ? Color.white.opacity(0.07)
            : Color.black.opacity(0.06)
    }

    static func subtle(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color.white.opacity(0.55) : Color.black.opacity(0.5)
    }
}

/// A soft card surface used for the output and other panels.
struct CardBackground: ViewModifier {
    @Environment(\.colorScheme) private var scheme
    var corner: CGFloat = Theme.cardCorner

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: corner, style: .continuous)
                    .fill(Theme.card(scheme))
            )
            .overlay(
                RoundedRectangle(cornerRadius: corner, style: .continuous)
                    .strokeBorder(Theme.cardStroke(scheme), lineWidth: 1)
            )
    }
}

extension View {
    func cardSurface(corner: CGFloat = Theme.cardCorner) -> some View {
        modifier(CardBackground(corner: corner))
    }
}
