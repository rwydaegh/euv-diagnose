from pathlib import Path

from euv_diagnose.demo import build_bundle

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "app/public/data/demo.json"
    build_bundle(root, output)
    print(f"Wrote {output}")
