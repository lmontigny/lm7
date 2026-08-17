import ExecuTorch
import SwiftUI

/// Fixed test vector exported alongside the artifact by
/// tools/export_ios_fixture.py.
struct Golden: Decodable {
    let input_shape: [Int]
    let input: [Float]
    let output_shape: [Int]
    let expected: [Float]
}

struct Outcome {
    let passed: Bool
    let maxAbsDiff: Float
    let lines: [String]
}

/// Tolerance from docs/ios-device-testing.md for the first Core ML smoke.
let tolerance: Float = 1e-2

func runValidation() -> Outcome {
    var lines: [String] = []

    func log(_ s: String) {
        NSLog("LM7VALIDATOR %@", s)
        lines.append(s)
    }

    guard let ptePath = Bundle.main.path(forResource: "compiled_model", ofType: "pte"),
          let goldenURL = Bundle.main.url(forResource: "golden", withExtension: "json")
    else {
        log("FAIL: fixture missing from app bundle")
        return Outcome(passed: false, maxAbsDiff: .nan, lines: lines)
    }

    do {
        let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: goldenURL))
        log("artifact: compiled_model.pte (ExecuTorch Core ML)")
        log("input: float32\(golden.input_shape)")

        let module = Module(filePath: ptePath)
        try module.load("forward")

        let input = Tensor<Float>(golden.input, shape: golden.input_shape)
        let output = try Tensor<Float>(module.forward(input))
        let actual = try output.scalars()

        log("output: float32\(try output.shape)")

        guard actual.count == golden.expected.count else {
            log("FAIL: element count \(actual.count) != expected \(golden.expected.count)")
            return Outcome(passed: false, maxAbsDiff: .nan, lines: lines)
        }

        var maxAbsDiff: Float = 0
        for (a, e) in zip(actual, golden.expected) {
            maxAbsDiff = max(maxAbsDiff, abs(a - e))
        }

        let passed = maxAbsDiff <= tolerance
        log(String(format: "max_abs_diff: %.6e (tolerance %.0e)", maxAbsDiff, tolerance))
        log(passed ? "RESULT: PASS" : "RESULT: FAIL")
        return Outcome(passed: passed, maxAbsDiff: maxAbsDiff, lines: lines)
    } catch {
        log("FAIL: \(error)")
        return Outcome(passed: false, maxAbsDiff: .nan, lines: lines)
    }
}

@main
struct LM7ValidatorApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @State private var outcome: Outcome?

    var body: some View {
        VStack(spacing: 14) {
            Text("LM7").font(.system(size: 44, weight: .bold))

            if let outcome {
                Text(outcome.passed ? "PASS" : "FAIL")
                    .font(.system(size: 60, weight: .heavy))
                    .foregroundStyle(outcome.passed ? .green : .red)
                    .accessibilityIdentifier("result")

                ForEach(outcome.lines, id: \.self) { line in
                    Text(line)
                        .font(.system(size: 13, design: .monospaced))
                        .multilineTextAlignment(.center)
                }
            } else {
                ProgressView()
            }
        }
        .padding()
        .task { outcome = runValidation() }
    }
}
