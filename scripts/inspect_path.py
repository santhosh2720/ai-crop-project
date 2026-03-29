from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts\\inspect_path.py <path>")
    raw = " ".join(sys.argv[1:]).strip().strip('"')
    path = Path(raw)
    print(f"path={path}")
    print(f"exists={path.exists()}")
    if path.exists():
        if path.is_dir():
            for child in sorted(path.iterdir()):
                print(child)
        else:
            print(path.read_text(encoding='utf-8', errors='ignore')[:4000])


if __name__ == "__main__":
    main()
