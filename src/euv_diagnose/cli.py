"""Small local command line for reproducible result export and the browser demo."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import sys
from pathlib import Path

from .demo import build_bundle


def workspace(path: str) -> Path:
    root = Path(path).resolve()
    if not (root / "research/sources/sherwin/manifest.json").is_file():
        raise ValueError(
            "Run from the EUV-Diagnose checkout, or pass --workspace /path/to/checkout"
        )
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore and reproduce EUV reflectometry results.")
    parser.add_argument(
        "--workspace", default=".", help="Repository checkout (default: current directory)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Recompute the small browser result bundle")
    export.add_argument(
        "--output", type=Path, help="Output JSON (default: app/public/data/demo.json)"
    )
    commands.add_parser("inspect", help="Print the available demo's provenance and evidence")
    demo = commands.add_parser("demo", help="Serve the built interactive demo locally")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        root = workspace(args.workspace)
        if args.command == "export":
            output = args.output or root / "app/public/data/demo.json"
            data = build_bundle(root, output)
            print(
                f"Wrote {output} ({output.stat().st_size:,} bytes; schema {data['schemaVersion']})"
            )
        elif args.command == "inspect":
            file = root / "app/public/data/demo.json"
            if not file.is_file():
                raise ValueError("Result bundle is missing. Run euv-diagnose export first.")
            data = json.loads(file.read_text())
            print(
                json.dumps(
                    {
                        "revision": data["revision"],
                        "synthetic": data["synthetic"]["metadata"],
                        "measured_available": "measured" in data,
                        "learning_available": "learning" in data,
                    },
                    indent=2,
                )
            )
        else:
            if not 0 <= args.port <= 65535:
                raise ValueError("Port must be between 0 and 65535")
            directory = root / "app/dist"
            if not (directory / "index.html").is_file():
                raise ValueError("Build the demo first: cd app && npm ci && npm run build")
            handler = functools.partial(
                http.server.SimpleHTTPRequestHandler, directory=str(directory)
            )
            with http.server.ThreadingHTTPServer((args.host, args.port), handler) as server:
                host, port = server.server_address[:2]
                print(f"EUV-Diagnose: http://{host}:{port} (Ctrl+C to stop)", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    print("\nDemo stopped.")
        return 0
    except (ValueError, OSError) as exc:
        print(f"euv-diagnose: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
