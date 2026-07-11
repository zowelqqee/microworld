import SwiftUI

struct HeaderView: View {
    var showInfo: () -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("MicroWorld")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                Text("Local intelligence. No cloud.")
                    .font(.subheadline)
                    .foregroundStyle(Theme.subtle(scheme))
            }
            Spacer()
            OfflinePill()
                .onTapGesture(perform: showInfo)
                .accessibilityAddTraits(.isButton)
                .accessibilityLabel("Offline. Tap for details.")
        }
    }
}

/// Small status pill: green dot + "Offline" + airplane glyph.
struct OfflinePill: View {
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(Theme.onlineGreen)
                .frame(width: 8, height: 8)
                .shadow(color: Theme.onlineGreen.opacity(0.6), radius: 3)
            Text("Offline")
                .font(.footnote.weight(.semibold))
            Image(systemName: "airplane")
                .font(.footnote)
                .foregroundStyle(Theme.subtle(scheme))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(
            Capsule().fill(scheme == .dark ? Color.white.opacity(0.08) : Color.black.opacity(0.05))
        )
    }
}
