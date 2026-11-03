"""Step 00 - Check setup.

Teaching point: fail fast, with the fix printed. Ten minutes into a live session is the
worst time to discover a missing package or an un-pulled model, so we verify everything
the later steps need: Python version, packages, .env, Ollama server, and both models.

Run:  python steps/step_00_check_setup.py
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

# Make `import rag` work when this file is run directly (python steps/step_00_...py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_PACKAGES: list[str] = ["ollama", "chromadb", "python-dotenv", "pypdf", "python-docx"]


def check_python() -> bool:
    version = sys.version_info
    ok = version >= (3, 10)
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
    print(f"         interpreter: {sys.executable}")
    return ok


def check_packages() -> bool:
    all_ok = True
    for package in REQUIRED_PACKAGES:
        try:
            version = importlib.metadata.version(package)
            print(f"  [OK  ] {package} {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  [FAIL] {package} is not installed")
            all_ok = False
    if not all_ok:
        print("         fix: pip install -r requirements.txt  (with your .venv activated)")
    return all_ok


def check_env_file() -> None:
    # Not fatal: config.py has defaults for everything.
    if (PROJECT_ROOT / ".env").exists():
        print("  [OK  ] .env found")
    else:
        print("  [WARN] .env not found - using defaults. Create one: copy .env.example .env")


def check_ollama() -> bool:
    from rag.config import load_settings
    from rag.embeddings import health_check

    settings = load_settings()
    report = health_check(settings)
    for message in report.messages:
        status = "OK  " if ("NOT" not in message and "Cannot" not in message) else "FAIL"
        print(f"  [{status}] {message}")
    for fix in report.fixes:
        print(f"         fix: {fix}")
    return report.ok


def main() -> None:
    print("=== Step 00: check setup ===\n")

    print("Python")
    python_ok = check_python()

    print("\nPackages")
    packages_ok = check_packages()

    print("\nConfiguration")
    check_env_file()

    print("\nOllama")
    ollama_ok = check_ollama() if packages_ok else False
    if not packages_ok:
        print("  [SKIP] install packages first")

    print()
    if python_ok and packages_ok and ollama_ok:
        print("All checks passed.")
    else:
        print("Some checks failed - apply the fixes above and re-run this step.")
        print("(Steps 01 and 02 work without Ollama, so you can continue with them.)")

    print("\nNext: python steps/step_01_load.py")


if __name__ == "__main__":
    main()
