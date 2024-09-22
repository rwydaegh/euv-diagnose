import argparse
from pathlib import Path
from .demo import build_bundle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export"])
    parser.add_argument("--output", type=Path, default=Path("app/public/data/demo.json"))
    args = parser.parse_args()
    build_bundle(Path.cwd(), args.output)
