import SwiftUI

/// Last-5 prompts for this session (in memory only, no persistence). Tap to rerun.
struct HistoryView: View {
    let history: [HistoryItem]
    var onRerun: (HistoryItem) -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        if !history.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Text("Recent")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(Theme.subtle(scheme))
                ForEach(history) { item in
                    Button { onRerun(item) } label: {
                        HStack(spacing: 10) {
                            Image(systemName: item.mode == .qa ? "questionmark.bubble" : "sparkles")
                                .font(.footnote)
                                .foregroundStyle(Theme.subtle(scheme))
                            Text(item.prompt)
                                .font(.subheadline)
                                .lineLimit(1)
                            Spacer()
                            Image(systemName: "arrow.counterclockwise")
                                .font(.caption)
                                .foregroundStyle(Theme.subtle(scheme))
                        }
                        .padding(.vertical, 10)
                        .padding(.horizontal, 12)
                        .cardSurface(corner: 12)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}
