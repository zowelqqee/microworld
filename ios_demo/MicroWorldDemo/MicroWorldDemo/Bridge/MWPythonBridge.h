#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Objective-C owner of the embedded CPython interpreter.
///
/// Responsibilities:
///   * initialise CPython exactly once, pointing PYTHONHOME/PYTHONPATH at the
///     bundled stdlib + the staged `worldpgt` engine and `mw_ios` adapter;
///   * expose the adapter's `warm_up` / `run` / `self_test` as string calls;
///   * serialise all interpreter access (CPython is single-interpreter; we hold
///     the GIL and run on one dedicated queue) so Swift can call from any actor.
///
/// All methods return the adapter's raw JSON string, or populate `error`.
@interface MWPythonBridge : NSObject

/// Shared bridge. The interpreter is initialised lazily on first use.
+ (instancetype)shared;

/// Initialise the interpreter if needed. Returns YES on success.
/// (Swift: `try bridge.initializeInterpreter()`.)
- (BOOL)initializeInterpreterWithError:(NSError **)error
    NS_SWIFT_NAME(initializeInterpreter());

/// Whether the interpreter has been initialised.
@property (nonatomic, readonly) BOOL isInitialized;

/// Call `mw_ios.warm_up(overlay, warm_creative)`. Returns JSON or nil.
/// (Swift: `try bridge.warmUp(overlay:warmCreative:)`.)
- (nullable NSString *)warmUpWithOverlay:(NSString *)overlay
                            warmCreative:(BOOL)warmCreative
                                   error:(NSError **)error
    NS_SWIFT_NAME(warmUp(overlay:warmCreative:));

/// Call `mw_ios.warm_creative()`. Returns JSON or nil.
/// (Swift: `try bridge.warmCreative()`.)
- (nullable NSString *)warmCreativeWithError:(NSError **)error
    NS_SWIFT_NAME(warmCreative());

/// Call `mw_ios.run(prompt, mode)`. Returns JSON or nil.
/// (Swift: `try bridge.run(prompt:mode:)`.)
- (nullable NSString *)runPrompt:(NSString *)prompt
                            mode:(NSString *)mode
                           error:(NSError **)error
    NS_SWIFT_NAME(run(prompt:mode:));

/// Call `mw_ios.self_test()`. Returns JSON or nil.
/// (Swift: `try bridge.selfTest()`.)
- (nullable NSString *)selfTestWithError:(NSError **)error
    NS_SWIFT_NAME(selfTest());

@end

NS_ASSUME_NONNULL_END
