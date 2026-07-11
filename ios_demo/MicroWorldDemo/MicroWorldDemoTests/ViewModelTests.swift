import XCTest
@testable import MicroWorldDemo

/// UI-state-transition tests using the mock engine (no interpreter needed).
@MainActor
final class ViewModelTests: XCTestCase {

    func testBootTransitionsToReady() async {
        let vm = AppViewModel(engine: MockMicroWorldEngine())
        XCTAssertEqual(vm.phase, .launching)
        await vm.boot()
        XCTAssertEqual(vm.phase, .ready)
    }

    func testBootFailureSurfacesError() async {
        let mock = MockMicroWorldEngine()
        mock.shouldFail = true
        let vm = AppViewModel(engine: mock)
        await vm.boot()
        if case .failed = vm.phase {} else { XCTFail("expected failed phase") }
        XCTAssertNotNil(vm.errorMessage)
    }

    func testSubmitProducesResultAndHistory() async {
        let vm = AppViewModel(engine: MockMicroWorldEngine())
        await vm.boot()
        vm.prompt = "Who founded SpaceX?"
        await vm.submit()
        XCTAssertNotNil(vm.result)
        XCTAssertEqual(vm.phase, .ready)
        XCTAssertEqual(vm.history.count, 1)
        XCTAssertEqual(vm.history.first?.prompt, "Who founded SpaceX?")
    }

    func testEmptySubmitShowsError() async {
        let vm = AppViewModel(engine: MockMicroWorldEngine())
        await vm.boot()
        vm.prompt = "   "
        await vm.submit()
        XCTAssertNil(vm.result)
        XCTAssertNotNil(vm.errorMessage)
    }

    func testModeChangeClearsResult() async {
        let vm = AppViewModel(engine: MockMicroWorldEngine())
        await vm.boot()
        vm.prompt = "Who founded SpaceX?"
        await vm.submit()
        XCTAssertNotNil(vm.result)
        vm.mode = .creative
        XCTAssertNil(vm.result, "switching mode clears the previous result")
    }

    func testHistoryCappedAtFive() async {
        let vm = AppViewModel(engine: MockMicroWorldEngine())
        await vm.boot()
        for i in 0..<7 {
            vm.prompt = "prompt \(i)"
            await vm.submit()
        }
        XCTAssertEqual(vm.history.count, 5)
        XCTAssertEqual(vm.history.first?.prompt, "prompt 6")
    }

    func testRunFailureSurfacesErrorButStaysReady() async {
        let mock = MockMicroWorldEngine()
        let vm = AppViewModel(engine: mock)
        await vm.boot()
        mock.shouldFail = true
        vm.prompt = "anything"
        await vm.submit()
        XCTAssertNotNil(vm.errorMessage)
        XCTAssertEqual(vm.phase, .ready)
    }

    func testLatencyRecordedInDiagnostics() async {
        let vm = AppViewModel(engine: MockMicroWorldEngine())
        await vm.boot()
        vm.prompt = "Who founded SpaceX?"
        await vm.submit()
        XCTAssertNotNil(vm.diagnostics.firstQaMs)
        XCTAssertNotNil(vm.diagnostics.engineColdStartMs)
    }
}
