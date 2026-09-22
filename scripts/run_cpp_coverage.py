#!/usr/bin/env python3
"""Automate C++ code coverage for scikit-kamila using gcovr or OpenCppCoverage.

Usage:
    python scripts/run_cpp_coverage.py [--html] [--xml]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_command(cmd, cwd=ROOT_DIR):
    """Run a shell command and print output."""
    print(f"\n[+] Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str))
    if res.returncode != 0:
        print(f"[-] Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run C++ code coverage for scikit-kamila.")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report.")
    parser.add_argument("--xml", action="store_true", help="Generate XML coverage report.")
    parser.add_argument("--output-dir", default="coverage_cpp", help="Directory for output reports.")
    args = parser.parse_args()

    is_windows = platform.system() == "Windows"
    out_dir = ROOT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_windows and shutil.which("OpenCppCoverage"):
        # Windows with OpenCppCoverage available
        print("[+] Detected Windows with OpenCppCoverage.")
        # Rebuild in Debug mode so PDB symbols are generated
        run_command([
            sys.executable, "-m", "pip", "install", "-e", ".",
            "--no-build-isolation", "--config-settings=cmake.build-type=Debug"
        ])

        occ_cmd = [
            "OpenCppCoverage",
            "--sources", str(ROOT_DIR / "src"),
            "--modules", "*_kamila_cpp*",
        ]
        if args.html:
            occ_cmd.extend(["--export_type", f"html:{out_dir / 'html'}"])
        if args.xml:
            occ_cmd.extend(["--export_type", f"cobertura:{out_dir / 'coverage-cpp.xml'}"])

        occ_cmd.extend(["--", sys.executable, "-m", "pytest", "kamila"])
        run_command(occ_cmd)
        print(f"\n[+] C++ coverage report completed in {out_dir}")

    else:
        # GCC / Clang / Linux / macOS with gcov + gcovr
        print("[+] Building with C++ coverage instrumentation (--coverage)...")
        run_command([
            sys.executable, "-m", "pip", "install", "-e", ".",
            "--no-build-isolation",
            "--config-settings=cmake.define.ENABLE_COVERAGE=ON"
        ])

        print("[+] Running test suite...")
        run_command([sys.executable, "-m", "pytest", "kamila"])

        if shutil.which("gcovr"):
            print("\n[+] Generating C++ coverage summary with gcovr...")
            gcovr_cmd = ["gcovr", "--root", str(ROOT_DIR), "--filter", str(ROOT_DIR / "src")]
            
            # Print summary to terminal
            run_command(gcovr_cmd + ["--print-summary"])

            if args.html:
                html_path = out_dir / "index.html"
                run_command(gcovr_cmd + ["--html-details", "-o", str(html_path)])
                print(f"[+] HTML coverage report generated at {html_path}")

            if args.xml:
                xml_path = out_dir / "coverage-cpp.xml"
                run_command(gcovr_cmd + ["--xml", "-o", str(xml_path)])
                print(f"[+] XML coverage report generated at {xml_path}")
        else:
            print("[-] 'gcovr' not found in PATH. Install via 'pip install gcovr' to generate coverage reports.")


if __name__ == "__main__":
    main()
