import re
from pathlib import Path


def normalize(path):
    data = path.read_bytes()
    if len(data) < 128 or not data.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError(f"Not a MATLAB 5 file: {path}")
    header = data[:116].rstrip(b" \x00")
    header = re.sub(rb", Created on:.*$|, \d{4}-\d{2}-\d{2} .* UTC$", b"", header)
    normalized = header.ljust(116, b" ") + data[116:]
    if normalized != data:
        path.write_bytes(normalized)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "research/results").glob("*.mat")):
        normalize(path)
