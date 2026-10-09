"""Install benchmark adapter copies at their frozen RiverMind paths, without overwriting files."""
import hashlib
import pathlib
import shutil

root = pathlib.Path(__file__).resolve().parents[1]
if root != pathlib.Path("/root/rivermind-data/mu_functions_v2_main9_20260929"):
    raise SystemExit("Clone or copy this repository to its frozen path under /root/rivermind-data first.")
for family, destination in (
    ("mmw2021", root.parent / "maliar_benchmarks_20260930/MMW2021/workspace_numeric_fix2_20261001"),
    ("mmv2010", root.parent / "maliar_benchmarks_20260930/MMV2010/workspace"),
):
    source = root / "benchmarks" / family
    for file in source.rglob("*"):
        if not file.is_file():
            continue
        target = destination / file.relative_to(source)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(file.read_bytes()).digest():
                raise SystemExit(f"Refusing to overwrite changed file: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    print(f"Installed {family} adapter at {destination}")
