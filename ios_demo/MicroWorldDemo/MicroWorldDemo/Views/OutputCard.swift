import SwiftUI

/// Displays the engine's result. Selectable text, subtle appear animation, and
/// a tappable metrics row. Renders only once a real result exists — there is no
/// token-by-token fake streaming.
struct OutputCard: View {
    let result: EngineResult
    var demoMode: Bool
    var onOpenMetrics: () -> Void
    @Environment(\.colorScheme) private var scheme

    private var creativeLabelAndBody: (label: String?, body: String) {
        // The engine prefixes creative output with its mandatory label line.
        let marker = "not verified fact.]"
        if let range = result.text.range(of: marker) {
            let label = String(result.text[..<range.upperBound])
            let rest = result.text[range.upperBound...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (label, rest)
        }
        return (nil, result.text)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if result.decision == "audit" {
                Label("Audit — no grounded answer", systemImage: "shield.lefthalf.filled")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.orange)
            }

            let parts = creativeLabelAndBody
            if let label = parts.label {
                Text(label)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(Theme.subtle(scheme))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(scheme == .dark ? Color.white.opacity(0.06) : Color.black.opacity(0.04))
                    )
            }

            Text(parts.body)
                .font(demoMode ? .title3 : .body)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)

            Divider().opacity(0.4)

            MetricsRow(result: result, demoMode: demoMode, onTap: onOpenMetrics)
        }
        .padding(demoMode ? 22 : 18)
        .cardSurface()
        .transition(.asymmetric(
            insertion: .opacity.combined(with: .move(edge: .bottom)),
            removal: .opacity
        ))
    }
}

/// Compact metrics beneath the response. Tap to open the full sheet.
struct MetricsRow: View {
    let result: EngineResult
    var demoMode: Bool
    var onTap: () -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 10) {
                metric(String(format: "%.1f ms", result.latencyMilliseconds), "bolt.fill")
                dot
                metric("Local", "iphone")
                dot
                metric(result.deterministic ? "Deterministic" : "Fresh each run",
                       result.deterministic ? "checkmark.seal" : "shuffle")
                Spacer()
                Image(systemName: "chevron.up.circle")
                    .foregroundStyle(Theme.subtle(scheme))
            }
            .font((demoMode ? Font.subheadline : Font.footnote).weight(.medium))
        }
        .buttonStyle(.plain)
        .accessibilityHint("Opens performance details")
    }

    private var dot: some View {
        Circle().fill(Theme.subtle(scheme).opacity(0.5)).frame(width: 3, height: 3)
    }

    private func metric(_ text: String, _ icon: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon).font(.caption2)
            Text(text)
        }
    }
}
