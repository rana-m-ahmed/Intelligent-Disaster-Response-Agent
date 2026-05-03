import os
import subprocess
import sys


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root)
    if result.returncode == 0:
        print("ALL TESTS PASSED")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
