#import "MWPythonBridge.h"
// Textual (quoted) include, NOT `<Python/Python.h>`. The framework's
// module.modulemap `exclude`s the cpython/* headers, so importing Python as a
// Clang *module* fails ("'cpython/pyatomic_gcc.h' file not found"). A quoted
// include with the framework's Headers dir on HEADER_SEARCH_PATHS resolves
// Python.h and its internal `cpython/...` includes textually via search-path
// fallback — no module build. See project.yml HEADER_SEARCH_PATHS.
#import "Python.h"

static NSString *const MWBridgeErrorDomain = @"MicroWorldPythonBridge";

@implementation MWPythonBridge {
    dispatch_queue_t _queue;   // serialises all interpreter access
    BOOL _initialized;
    PyObject *_module;         // the imported `mw_ios` module (owned)
}

+ (instancetype)shared {
    static MWPythonBridge *shared;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ shared = [[MWPythonBridge alloc] init]; });
    return shared;
}

- (instancetype)init {
    if ((self = [super init])) {
        // A single serial queue guarantees one-thread interpreter access. We
        // still acquire the GIL defensively inside each call.
        _queue = dispatch_queue_create("com.microworld.python", DISPATCH_QUEUE_SERIAL);
        _initialized = NO;
        _module = NULL;
    }
    return self;
}

- (BOOL)isInitialized {
    return _initialized;
}

#pragma mark - Interpreter lifecycle

- (BOOL)initializeInterpreterWithError:(NSError **)error {
    __block BOOL ok = NO;
    __block NSError *localError = nil;
    dispatch_sync(_queue, ^{
        ok = [self _initLocked:&localError];
    });
    if (!ok && error) { *error = localError; }
    return ok;
}

// MUST be called on _queue.
- (BOOL)_initLocked:(NSError **)error {
    if (_initialized) { return YES; }

    NSBundle *bundle = [NSBundle mainBundle];

    // The Python framework ships a `python-stdlib` resource dir; the staged
    // engine + adapter live under `app_packages`. Both are bundled folder refs.
    NSString *stdlibPath = [bundle pathForResource:@"python-stdlib" ofType:nil];
    NSString *appPackages = [bundle pathForResource:@"app_packages" ofType:nil];

    if (stdlibPath == nil) {
        if (error) {
            *error = [NSError errorWithDomain:MWBridgeErrorDomain code:1
                        userInfo:@{NSLocalizedDescriptionKey:
                            @"Missing bundled `python-stdlib` resource. "
                            @"Add Python.xcframework's python-stdlib to the app "
                            @"(see README_IOS.md)."}];
        }
        return NO;
    }
    if (appPackages == nil) {
        if (error) {
            *error = [NSError errorWithDomain:MWBridgeErrorDomain code:2
                        userInfo:@{NSLocalizedDescriptionKey:
                            @"Missing bundled `app_packages`. Run "
                            @"ios_demo/scripts/stage_bundle.sh, then add the "
                            @"app_packages folder reference to the target."}];
        }
        return NO;
    }

    // Configure an isolated interpreter (no site, no user env, no network).
    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    config.write_bytecode = 0;   // read-only bundle: don't try to write .pyc
    config.user_site_directory = 0;
    config.site_import = 0;      // no site.py; we set paths explicitly below
    config.buffered_stdio = 0;

    // PYTHONHOME → the stdlib prefix.
    PyStatus status = PyConfig_SetBytesString(&config, &config.home,
                                              stdlibPath.UTF8String);
    if (PyStatus_Exception(status)) {
        return [self _failConfig:&config status:status error:error];
    }

    // Build module search path: stdlib, stdlib/lib-dynload, app_packages.
    config.module_search_paths_set = 1;
    NSArray<NSString *> *paths = @[
        stdlibPath,
        [stdlibPath stringByAppendingPathComponent:@"lib-dynload"],
        appPackages,
    ];
    for (NSString *p in paths) {
        // Py_DecodeLocale is CPython's canonical char* -> wchar_t* conversion.
        wchar_t *w = Py_DecodeLocale(p.UTF8String, NULL);
        if (w == NULL) {
            return [self _failConfig:&config status:PyStatus_Error("Py_DecodeLocale failed") error:error];
        }
        status = PyWideStringList_Append(&config.module_search_paths, w);
        PyMem_RawFree(w);
        if (PyStatus_Exception(status)) {
            return [self _failConfig:&config status:status error:error];
        }
    }

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        if (error) {
            *error = [NSError errorWithDomain:MWBridgeErrorDomain code:3
                        userInfo:@{NSLocalizedDescriptionKey:
                            [NSString stringWithFormat:@"Py_Initialize failed: %s",
                                status.err_msg ? status.err_msg : "unknown"]}];
        }
        return NO;
    }

    // Import the adapter once; keep a strong ref.
    PyObject *module = PyImport_ImportModule("mw_ios");
    if (module == NULL) {
        [self _capturePythonError:error];
        return NO;
    }
    _module = module;  // owned
    _initialized = YES;

    // Py_InitializeFromConfig leaves the GIL held by whichever worker thread
    // happened to initialise the interpreter. A serial DispatchQueue does not
    // promise to reuse that same OS thread for later bridge calls: without
    // releasing it here, the next call on another worker blocks forever in
    // PyGILState_Ensure (the UI remains stuck at “Working…”). We deliberately
    // keep the interpreter alive for the app lifetime, so there is no matching
    // Py_Finalize call; each later call temporarily acquires/releases the GIL.
    PyEval_SaveThread();
    return YES;
}

- (BOOL)_failConfig:(PyConfig *)config status:(PyStatus)status error:(NSError **)error {
    PyConfig_Clear(config);
    if (error) {
        *error = [NSError errorWithDomain:MWBridgeErrorDomain code:4
                    userInfo:@{NSLocalizedDescriptionKey:
                        [NSString stringWithFormat:@"PyConfig failed: %s",
                            status.err_msg ? status.err_msg : "unknown"]}];
    }
    return NO;
}

#pragma mark - Calls

// `buildArgs` is invoked AFTER the interpreter is guaranteed initialised and
// WITH the GIL held; it must return a new tuple reference (or NULL for no args).
- (nullable NSString *)_callFunction:(const char *)name
                           buildArgs:(PyObject *(^)(void))buildArgs
                               error:(NSError **)error {
    __block NSString *result = nil;
    __block NSError *localError = nil;
    dispatch_sync(_queue, ^{
        if (!self->_initialized && ![self _initLocked:&localError]) {
            return;
        }
        PyGILState_STATE gil = PyGILState_Ensure();
        PyObject *args = buildArgs ? buildArgs() : PyTuple_New(0);
        PyObject *fn = PyObject_GetAttrString(self->_module, name);
        if (fn == NULL || !PyCallable_Check(fn)) {
            [self _capturePythonError:&localError];
            Py_XDECREF(fn);
            Py_XDECREF(args);
            PyGILState_Release(gil);
            return;
        }
        PyObject *ret = PyObject_CallObject(fn, args);
        Py_DECREF(fn);
        Py_XDECREF(args);
        if (ret == NULL) {
            [self _capturePythonError:&localError];
            PyGILState_Release(gil);
            return;
        }
        // Adapter functions all return `str` (JSON).
        if (PyUnicode_Check(ret)) {
            const char *utf8 = PyUnicode_AsUTF8(ret);
            if (utf8) { result = [NSString stringWithUTF8String:utf8]; }
        }
        if (result == nil) {
            localError = [NSError errorWithDomain:MWBridgeErrorDomain code:5
                            userInfo:@{NSLocalizedDescriptionKey:
                                @"Adapter did not return a string."}];
        }
        Py_DECREF(ret);
        PyGILState_Release(gil);
    });
    if (result == nil && error) { *error = localError; }
    return result;
}

- (nullable NSString *)warmUpWithOverlay:(NSString *)overlay
                            warmCreative:(BOOL)warmCreative
                                   error:(NSError **)error {
    NSString *overlayCopy = [overlay copy];
    return [self _callFunction:"warm_up" buildArgs:^PyObject *{
        return Py_BuildValue("(sO)", overlayCopy.UTF8String,
                             warmCreative ? Py_True : Py_False);
    } error:error];
}

- (nullable NSString *)warmCreativeWithError:(NSError **)error {
    return [self _callFunction:"warm_creative" buildArgs:nil error:error];
}

- (nullable NSString *)runPrompt:(NSString *)prompt
                            mode:(NSString *)mode
                           error:(NSError **)error {
    NSString *promptCopy = [prompt copy];
    NSString *modeCopy = [mode copy];
    return [self _callFunction:"run" buildArgs:^PyObject *{
        return Py_BuildValue("(ss)", promptCopy.UTF8String, modeCopy.UTF8String);
    } error:error];
}

- (nullable NSString *)selfTestWithError:(NSError **)error {
    return [self _callFunction:"self_test" buildArgs:nil error:error];
}

#pragma mark - Errors

// MUST be called with the GIL held.
- (void)_capturePythonError:(NSError **)error {
    NSString *message = @"Unknown Python error";
    if (PyErr_Occurred()) {
        PyObject *type = NULL, *value = NULL, *tb = NULL;
        PyErr_Fetch(&type, &value, &tb);
        PyErr_NormalizeException(&type, &value, &tb);
        if (value) {
            PyObject *str = PyObject_Str(value);
            if (str) {
                const char *utf8 = PyUnicode_AsUTF8(str);
                if (utf8) { message = [NSString stringWithUTF8String:utf8]; }
                Py_DECREF(str);
            }
        }
        Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(tb);
        PyErr_Clear();
    }
    if (error) {
        *error = [NSError errorWithDomain:MWBridgeErrorDomain code:6
                    userInfo:@{NSLocalizedDescriptionKey: message}];
    }
}

@end
