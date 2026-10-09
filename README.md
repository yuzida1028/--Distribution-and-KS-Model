# Model B: distribution-conditioned neural policy comparison

This package accompanies the JEDC2010 Model B study. The main displayed methods are full-distribution DeepONet, full-distribution MLP, capital-only MLP, and the MMW2021 Model B adaptation, each with seeds 9711–9713. The frozen registry also contains three MMV2010 matched runs. The four-method display was chosen after the 15-policy evaluation and does not replace the frozen registry.

## Package contents

`src/`, `scripts/`, `server/`, and `config/` contain the formal neural workflow. `benchmarks/mmw2021/` contains the **numeric_fix2** code actually used by the completed MMW runs. `benchmarks/mmv2010/` contains only the project's adaptation code, not the original authors' software. `protocol/` preserves the original frozen JSON bytes. `results/` contains audit and compact per-run summaries. `figures/` contains the eight four-method plots and drawing source CSVs.

The accompanying Release contains `release_data.zip`, `release_policies.zip`, `release_raw_final.zip`, and `release_tiff.zip` as numbered 64 MiB parts, plus the separately licensed single file `release_thirdparty_mmv.zip`. Download every `*.partNN` attachment into one folder, then run `python assemble_release_parts.py --parts-dir /path/to/downloads --output-dir /path/to/downloads`. The script verifies each part against `RELEASE_PARTS_MANIFEST.json`, joins the parts, and verifies the original ZIP hashes in `SHA256SUMS.txt`. The reconstructed ZIPs contain exact RiverMind-relative paths. Extract them beneath `/root/rivermind-data` for the unmodified historical evaluator, whose frozen registry and benchmark bindings record those absolute paths. The original registry SHA256 is `8abc0fba54a71ac6904cb1703b1e21c86594241eda12446ac34103ff50f906dc`; do not edit it in place. Moving to another directory requires a separately documented path relocation and new derivative manifest.

For the historical exact-layout check, place this repository at `/root/rivermind-data/mu_functions_v2_main9_20260929`, extract Release assets into `/root/rivermind-data`, and run `python scripts/install_frozen_benchmarks.py`. This installer copies the adapted benchmark sources/configs to their recorded sibling directories and refuses to overwrite changed files. Run `python scripts/focused_final_compare_20261002.py --help` for the original evaluator's interface. These exact-layout instructions have been checked statically; the full evaluator was not re-run during packaging.

`release_data.zip` includes the 24/6/12 generated function-group data, aggregate arrays, and sealed paths; these are generated experimental states, not survey microdata. `release_policies.zip` has the exact recorded weights, training histories and required run manifests. `release_raw_final.zip` has all 15 final per-state/per-path outputs, including MMV archival results. `release_tiff.zip` completes the figure audit's TIFF entries; SVG/PDF/PNG are in the repository.

The paper's common-state and own-policy-path metrics refer to different state domains. MMW is an adaptation of an existing distribution-aware method. The capital-only MLP is a DEQN-style information-limited baseline, not a line-by-line reproduction of published DEQN code.

The optional MMV original-software asset is governed by its own `LICENSE_AGREEMENT.txt`, which restricts use to noncommercial research and education. The project changed the method for Model B in 2026 and provides its changed Python adaptation in `benchmarks/mmv2010/`; the original software remains in its untouched archive inside that separately labeled asset. If the MMV appendix is omitted, this third-party asset need not be distributed. The frozen 15-policy verifier does require its archived bytes when run unchanged. Cite Maliar, Maliar and Valli (2010) when using those results.

## Environment and verification

Recorded formal neural/MMW runs used Python 3.12.13 and PyTorch 2.12.1+cu130 on an NVIDIA GeForce RTX 2080 Ti (22,528 MiB reported). The present `requirements.txt` lists dependencies but is **not** a lockfile; exact versions for NumPy/SciPy/Matplotlib were not recorded in the supplied run manifests. Install a PyTorch build suitable for your hardware. Validate release assets with `SHA256SUMS.txt` and all selected source files with `PACKAGE_MANIFEST.json`.

The unmodified `scripts/focused_final_compare_20261002.py` checks the original 15 policies and their original absolute locations before evaluating. The original data, config and policy hashes are preserved. No new training or evaluation was performed to make this package.

## Input provenance

Model B calibration is in `data/jedc2010_calibration.json`. The derived initial histogram is in `data/jedc2010/initial_distribution.npz`. Original suite inputs are available from the [JEDC computational suite](https://www.wouterdenhaan.com/datasuite.htm); this package omits the original `pdist.xls`, `ModelB.txt`, shock text, and Young archive pending separate redistribution review. Cite the original suite and method papers in the manuscript. `benchmarks/mmv2010/` is the project's separate adaptation only; the original MMV software is in the separately licensed asset.

`protocol/experiment_manifest.json`, `protocol/split_manifest.json`, and `protocol/focused_final_registry.json` are copied byte-for-byte from the frozen materials. The data SHA256 checks are also recorded in the split manifest and registry. Large generated files and checkpoints belong in a Release or equivalent object storage, not ordinary Git history.
