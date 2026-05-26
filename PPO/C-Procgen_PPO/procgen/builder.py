import threading
import os
import contextlib
import subprocess as sp
import shutil
import json
import sys
import platform
import multiprocessing as mp

import gym3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


global_build_lock = threading.Lock()
global_builds = set()


class RunFailure(Exception):
    pass


@contextlib.contextmanager
def nullcontext():
    # this is here for python 3.6 support
    yield


@contextlib.contextmanager
def chdir(newdir):
    curdir = os.getcwd()
    try:
        os.chdir(newdir)
        yield
    finally:
        os.chdir(curdir)


def _resolve_cmake_executable():
    """
    Resolve cmake from PATH first, then from the Python cmake package.
    This avoids WinError 2 when cmake is installed in a venv but not on PATH.
    """
    cmake_exe = shutil.which("cmake")
    if cmake_exe:
        return cmake_exe

    try:
        import cmake  # type: ignore

        exe_name = "cmake.exe" if platform.system() == "Windows" else "cmake"
        package_cmake = os.path.join(cmake.CMAKE_BIN_DIR, exe_name)
        if os.path.exists(package_cmake):
            return package_cmake
    except Exception:
        pass

    return "cmake"


def _resolve_windows_sdk_tools():
    """
    Return (sdk_bin_dir, rc_exe, mt_exe) if found, else None.
    """
    if platform.system() != "Windows":
        return None

    sdk_root = r"C:\Program Files (x86)\Windows Kits\10\bin"
    if not os.path.isdir(sdk_root):
        return None

    for version in sorted(os.listdir(sdk_root), reverse=True):
        sdk_bin_dir = os.path.join(sdk_root, version, "x64")
        rc_exe = os.path.join(sdk_bin_dir, "rc.exe")
        mt_exe = os.path.join(sdk_bin_dir, "mt.exe")
        if os.path.exists(rc_exe) and os.path.exists(mt_exe):
            return sdk_bin_dir, rc_exe, mt_exe

    return None


def _get_python_qt_cmake_prefixes():
    """
    Return known Qt CMake prefixes from Python packages if present.
    """
    candidates = [
        os.path.join(sys.prefix, "Lib", "site-packages", "PySide6", "Qt", "lib", "cmake"),
        os.path.join(sys.prefix, "Lib", "site-packages", "PyQt5", "Qt5", "lib", "cmake"),
    ]
    return [path for path in candidates if os.path.isdir(path)]


def _get_local_qt_cmake_prefixes():
    """
    Return common local Qt SDK CMake prefixes if present.
    """
    candidates = [
        r"C:\Qt\5.15.2\msvc2019_64\lib\cmake",
        r"C:\Qt\5.15.2\msvc2022_64\lib\cmake",
    ]
    return [path for path in candidates if os.path.isdir(path)]


def run(cmd):
    if cmd and cmd[0] == "cmake":
        cmake_exe = _resolve_cmake_executable()
        if cmake_exe == "cmake" and shutil.which("cmake") is None:
            # Fall back to module execution in case scripts are unavailable.
            try:
                import cmake  # type: ignore  # noqa: F401

                cmd = [sys.executable, "-m", "cmake", *cmd[1:]]
            except Exception as exc:
                raise RunFailure(
                    "cmake was not found. Install CMake or the Python 'cmake' package."
                ) from exc
        else:
            cmd = [
                cmake_exe,
                *cmd[1:],
            ]

    if cmd and not os.path.isabs(cmd[0]):
        resolved = shutil.which(cmd[0])
        if resolved is not None:
            cmd = [resolved, *cmd[1:]]

    env = None
    sdk_tools = _resolve_windows_sdk_tools()
    if sdk_tools is not None:
        sdk_bin_dir, rc_exe, mt_exe = sdk_tools
        env = os.environ.copy()
        env["PATH"] = sdk_bin_dir + os.pathsep + env.get("PATH", "")
        env.setdefault("RC", rc_exe)
        env.setdefault("CMAKE_MT", mt_exe)

    try:
        return sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, encoding="utf8", env=env)
    except FileNotFoundError as exc:
        raise RunFailure(
            f"failed to execute build command: {cmd[0]} (not found)"
        ) from exc


def check(proc, verbose):
    if proc.returncode != 0:
        print(f"RUN FAILED {proc.args}:\n{proc.stdout}")
        raise RunFailure("failed to build procgen from source")
    if verbose:
        print(f"RUN {proc.args}:\n{proc.stdout}")


def _attempt_configure(build_type, package):
    if "PROCGEN_CMAKE_PREFIX_PATH" in os.environ:
        cmake_prefix_paths = [os.environ["PROCGEN_CMAKE_PREFIX_PATH"]]
    else:
        # guess some common qt cmake paths, it's unclear why cmake can't find qt without this
        cmake_prefix_paths = []
        if platform.system() != "Windows":
            cmake_prefix_paths.append("/usr/local/opt/qt5/lib/cmake")
        conda_exe = shutil.which("conda")
        if conda_exe is not None:
            conda_info = json.loads(
                sp.run(["conda", "info", "--json"], stdout=sp.PIPE).stdout
            )
            conda_prefix = conda_info["active_prefix"]
            if conda_prefix is None:
                conda_prefix = conda_info["conda_prefix"]
            if platform.system() == "Windows":
                conda_prefix = os.path.join(conda_prefix, "library")
            conda_cmake_path = os.path.join(conda_prefix, "lib", "cmake", "Qt5")
            # prepend this qt since it's likely to be loaded already by the python process
            cmake_prefix_paths.insert(0, conda_cmake_path)

    for qt_prefix in _get_python_qt_cmake_prefixes():
        if qt_prefix not in cmake_prefix_paths:
            cmake_prefix_paths.insert(0, qt_prefix)

    for qt_prefix in _get_local_qt_cmake_prefixes():
        if qt_prefix not in cmake_prefix_paths:
            cmake_prefix_paths.insert(0, qt_prefix)

    def configure_cmd_for_generator(generator_name):
        extra_configure_options = []
        if platform.system() == "Windows" and generator_name.startswith("Visual Studio"):
            extra_configure_options.extend(["-A", "x64"])

        cmd = [
            "cmake",
            "-G",
            generator_name,
            *extra_configure_options,
            f"-DLIBENV_DIR={gym3.libenv.get_header_dir()}",
            "../..",
        ]
        if cmake_prefix_paths:
            cmd.append("-DCMAKE_PREFIX_PATH=" + ";".join(cmake_prefix_paths))
        if package:
            cmd.append("-DPROCGEN_PACKAGE=ON")
        if platform.system() != "Windows":
            # this is not used on windows, the option needs to be passed to cmake --build instead
            cmd.append(f"-DCMAKE_BUILD_TYPE={build_type}")
        return cmd

    if platform.system() != "Windows":
        check(run(configure_cmd_for_generator("Unix Makefiles")), verbose=package)
        return

    configured_generator = os.environ.get("PROCGEN_CMAKE_GENERATOR") or os.environ.get(
        "CMAKE_GENERATOR"
    )
    if configured_generator:
        candidate_generators = [configured_generator]
    else:
        candidate_generators = [
            "Visual Studio 17 2022",
            "Visual Studio 16 2019",
        ]
        if shutil.which("ninja") is not None:
            candidate_generators.append("Ninja")
        if shutil.which("nmake") is not None:
            candidate_generators.append("NMake Makefiles")
        if shutil.which("mingw32-make") is not None:
            candidate_generators.append("MinGW Makefiles")

    last_proc = None
    for generator in candidate_generators:
        # CMake caches the chosen generator in this build directory.
        # Clear cache files so we can retry with a different generator.
        if os.path.exists("CMakeCache.txt"):
            os.remove("CMakeCache.txt")
        if os.path.isdir("CMakeFiles"):
            shutil.rmtree("CMakeFiles")

        proc = run(configure_cmd_for_generator(generator))
        if proc.returncode == 0:
            if package:
                print(f"RUN {proc.args}:\n{proc.stdout}")
            return
        last_proc = proc

    if last_proc is not None:
        check(last_proc, verbose=package)


def build(package=False, debug=False):
    """
    Build the requested environment in a process-safe manner and only once per process.
    """
    build_dir = os.path.join(SCRIPT_DIR, ".build")
    os.makedirs(build_dir, exist_ok=True)

    build_type = "relwithdebinfo"
    if debug:
        build_type = "debug"

    with chdir(build_dir), global_build_lock:
        # check if we have built yet in this process
        if build_type not in global_builds:
            if package:
                # avoid the filelock dependency when building from setup.py
                lock_ctx = nullcontext()
            else:
                # prevent multiple processes from trying to build at the same time
                import filelock

                lock_ctx = filelock.FileLock(".build-lock")
            with lock_ctx:
                sys.stdout.write("Building C-Procgen...")
                sys.stdout.flush()
                try:
                    os.makedirs(build_type, exist_ok=True)
                    with chdir(build_type):
                        _attempt_configure(build_type, package)
                except RunFailure:
                    # cmake can get into a weird state, so nuke the build directory and retry once
                    sys.stdout.write("retrying configure due to failure...")
                    sys.stdout.flush()
                    shutil.rmtree(build_type)
                    os.makedirs(build_type, exist_ok=True)
                    with chdir(build_type):
                        _attempt_configure(build_type, package)

                if "MAKEFLAGS" not in os.environ:
                    os.environ["MAKEFLAGS"] = f"-j{mp.cpu_count()}"

                with chdir(build_type):
                    build_cmd = ["cmake", "--build", ".", "--config", build_type]
                    check(run(build_cmd), verbose=package)
                print("done")

            global_builds.add(build_type)

    lib_dir = os.path.join(build_dir, build_type)
    if platform.system() == "Windows":
        # the built library is in a different location on windows
        lib_dir = os.path.join(lib_dir, build_type)
    return lib_dir
