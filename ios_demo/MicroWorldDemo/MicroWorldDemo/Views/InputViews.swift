import SwiftUI

/// Native segmented control for QA / Creative.
struct ModeSelector: View {
    @Binding var mode: EngineMode

    var body: some View {
        Picker("Mode", selection: $mode) {
            ForEach(EngineMode.allCases) { m in
                Text(m.title).tag(m)
            }
        }
        .pickerStyle(.segmented)
    }
}

/// Multiline prompt editor with a mode-dependent placeholder.
struct PromptEditor: View {
    @Binding var text: String
    let placeholder: String
    var demoMode: Bool
    @Environment(\.colorScheme) private var scheme
    @FocusState private var focused: Bool

    var body: some View {
        ZStack(alignment: .topLeading) {
            if text.isEmpty {
                Text(placeholder)
                    .foregroundStyle(Theme.subtle(scheme))
                    .font(demoMode ? .title3 : .body)
                    .padding(.top, 10)
                    .padding(.leading, 6)
                    .allowsHitTesting(false)
            }
            TextEditor(text: $text)
                .font(demoMode ? .title3 : .body)
                .focused($focused)
                .scrollContentBackground(.hidden)
                .frame(minHeight: demoMode ? 120 : 96)
                // Predictable for screen recording: no surprise autocorrection.
                .autocorrectionDisabled(demoMode)
                .textInputAutocapitalization(demoMode ? .never : .sentences)
        }
        .padding(10)
        .cardSurface(corner: Theme.corner)
        .onTapGesture { focused = true }
    }
}

/// Horizontally scrollable preset chips. Inputs only — never canned answers.
struct PresetChips: View {
    let presets: [String]
    var onTap: (String) -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(presets, id: \.self) { p in
                    Button { onTap(p) } label: {
                        Text(p)
                            .font(.footnote.weight(.medium))
                            .padding(.horizontal, 14)
                            .padding(.vertical, 9)
                            .background(
                                Capsule().fill(scheme == .dark
                                    ? Color.white.opacity(0.08)
                                    : Color.black.opacity(0.05))
                            )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 1)
        }
    }
}

/// Primary action button ("Ask" / "Generate"), with a spinner while running.
struct PrimaryButton: View {
    let title: String
    let running: Bool
    let enabled: Bool
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if running {
                    ProgressView().tint(.white)
                }
                Text(running ? "Working…" : title)
                    .font(.headline)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .disabled(!enabled || running)
    }
}
