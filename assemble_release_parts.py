"""Reassemble the four large Release ZIPs from verified 64 MiB parts.

Example: python assemble_release_parts.py --parts-dir downloads --output-dir downloads
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("RELEASE_PARTS_MANIFEST.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["version"] != 1:
        raise SystemExit("Unsupported parts manifest version")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, record in manifest["archives"].items():
        output = args.output_dir / name
        if output.exists():
            if output.stat().st_size != record["size"] or sha256(output) != record["sha256"]:
                raise SystemExit(f"Existing archive differs; refusing overwrite: {output}")
            print(f"VERIFIED EXISTING {output}")
            continue
        for part in record["parts"]:
            source = args.parts_dir / part["name"]
            if not source.is_file() or source.stat().st_size != part["size"] or sha256(source) != part["sha256"]:
                raise SystemExit(f"Missing or mismatched part: {source}")
        temp = output.with_name(output.name + ".assembling")
        if temp.exists():
            raise SystemExit(f"Temporary output exists; inspect before retrying: {temp}")
        try:
            with temp.open("xb") as target:
                for part in record["parts"]:
                    with (args.parts_dir / part["name"]).open("rb") as source:
                        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                            target.write(block)
            if temp.stat().st_size != record["size"] or sha256(temp) != record["sha256"]:
                raise SystemExit(f"Reassembled archive failed SHA-256: {name}")
            temp.rename(output)
            print(f"ASSEMBLED {output}")
        except BaseException:
            temp.unlink(missing_ok=True)
            raise


if __name__ == "__main__":
    main()
