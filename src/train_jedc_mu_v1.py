"""Train PI-DeepONet or the MLP control on the same KS residual."""

import argparse
import csv
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ks_model import KSEconomy
from networks import PolicyNetwork
from networks_mu_v2 import MuPolicyNetwork
from jedc_distribution_v1 import IntervalEconomy


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_manifest(config_path, calibration_path, train_path, valid_path, args, device):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "run_id": args.run_id, "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_cuda": torch.version.cuda,
        "config": json.loads(config_path.read_text(encoding="utf-8")),
        "sha256": {
            "config": file_hash(config_path), "calibration": file_hash(calibration_path),
            "train_data": file_hash(train_path), "valid_data": file_hash(valid_path),
            "ks_model_source": file_hash(ROOT / "src/ks_model.py"),
            "network_source": file_hash(ROOT / "src/networks.py"),
            "train_source": file_hash(ROOT / "src/train_jedc_mu_v1.py"),
            "interval_source": file_hash(ROOT / "src/jedc_distribution_v1.py"),
            "v2_network_source": file_hash(ROOT / "src/networks_mu_v2.py"),
        },
    }


def load_states(name, config_name, device, source=None):
    source = ROOT / source if source else ROOT / "data" / f"{name}_{config_name}.npz"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}. Run scripts/make_data.py first.")
    with np.load(source) as data:
        mu = torch.from_numpy(data["mu"].copy()).to(device)
        z = torch.from_numpy(data["z"].copy()).to(device)
        grid = torch.from_numpy(data["grid"].copy()).to(device)
    return mu, z, grid


def numeric(diagnostics):
    return {name: float(value.detach().cpu()) for name, value in diagnostics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("deeponet", "mlp"), default="deeponet")
    parser.add_argument("--resume", action="store_true", help="resume a same-device run from checkpoint.pt")
    parser.add_argument("--run-id", default="", help="unique suffix for a new run; required for auditable Model B runs")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("status", "")).startswith("draft"):
        raise ValueError("draft configuration is not approved for training; freeze a new config after Stage B")
    model_b = ("jedc" in config_path.stem or "model_b" in config_path.stem or
               "stage_a_v" in config.get("train_data_file", ""))
    if config_path.stem == "full" or (model_b and "calibration_file" not in config):
        raise ValueError("Model B requires an explicit calibration_file; legacy full.json is not a formal run")
    if model_b and not args.run_id:
        raise ValueError("Model B runs require a unique --run-id")
    if args.run_id and (not args.run_id.replace("-", "").replace("_", "").isalnum()):
        raise ValueError("run-id may contain only letters, numbers, hyphens and underscores")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available() else config["device"] if config["device"] != "auto" else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    else:
        torch.cuda.reset_peak_memory_stats(device)
    state_data_stem = config.get("state_data_stem", config_path.stem)
    train_path = (ROOT / config["train_data_file"] if "train_data_file" in config
                  else ROOT / "data" / f"train_{state_data_stem}.npz")
    valid_path = (ROOT / config["valid_data_file"] if "valid_data_file" in config
                  else ROOT / "data" / f"valid_{state_data_stem}.npz")
    train_mu, train_z, grid = load_states("train", state_data_stem, device, train_path)
    valid_mu, valid_z, valid_grid = load_states("valid", state_data_stem, device, valid_path)
    if not torch.equal(grid, valid_grid):
        raise ValueError("train and validation grids differ")
    calibration_path = ROOT / config.get("calibration_file", "data/ks_calibration.json")
    policy_mode = config.get("policy_mode", "sigmoid")
    with np.load(train_path) as interval_data:
        edges = torch.from_numpy(interval_data["edges"].copy()).to(device)
    with np.load(valid_path) as interval_data:
        if not np.array_equal(interval_data["edges"], edges.cpu().numpy()):
            raise ValueError("train/dev interval edges differ")
    if config.get("transport") != "atom_uniform_intervals_v1":
        raise ValueError("this trainer requires interval v1")
    economy = IntervalEconomy(
        calibration_path, edges, device, policy_mode=policy_mode,
        low_asset_multiplier=config.get("low_asset_multiplier", 1.0),
        low_asset_cutoff=config.get("low_asset_cutoff", 5.0),
        high_asset_multiplier=config.get("high_asset_multiplier", 1.0),
        high_asset_cutoff=config.get("high_asset_cutoff", 20.0),
        complementarity_asset_scale=config.get("complementarity_asset_scale"),
        mass_weight=config.get("mass_weight", 0.5))
    hidden = config["hidden"] if args.model == "deeponet" else config.get("mlp_hidden", config["hidden"])
    network_class = MuPolicyNetwork if config.get("status", "").startswith("frozen_mu_functions_v2") else PolicyNetwork
    network = network_class(args.model, len(grid), config["asset_max"], hidden, config["basis"],
                            macro_features=config.get("macro_features", "none"), grid=grid).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
    suffix = f"_{args.run_id}" if args.run_id else ""
    run_dir = ROOT / "runs" / f"{args.model}_{config_path.stem}{suffix}"
    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = run_manifest(config_path, calibration_path, train_path, valid_path, args, device)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    generator = torch.Generator(device=device).manual_seed(seed)
    history = []
    start_step = 1
    checkpoint_path = run_dir / "checkpoint.pt"
    if args.resume:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        current = run_manifest(config_path, calibration_path, train_path, valid_path, args, device)
        if manifest["sha256"] != current["sha256"]:
            raise ValueError("source, configuration or data changed since this run; cannot resume")
        # Our own resume checkpoint includes optimizer and NumPy/Python RNG
        # metadata, not just tensors. PyTorch>=2.6 defaults to weights_only=True.
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["config"] != config or saved["model"] != args.model or saved["device"] != str(device):
            raise ValueError("checkpoint configuration/model/device differs; start a new run")
        network.load_state_dict(saved["network"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        history = saved["history"]
        start_step = saved["step"] + 1
    if device.type == "cuda": torch.cuda.synchronize()
    previous_seconds = saved.get("effective_seconds", 0.0) if args.resume else 0.0
    train_started = time.perf_counter()
    for step in range(start_step, config["steps"] + 1):
        indices = torch.randint(len(train_mu), (config["batch_size"],), generator=generator, device=device)
        optimizer.zero_grad()
        loss, diagnostics = economy.residuals(network, train_mu[indices], train_z[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        row = {"step": step, **numeric(diagnostics)}
        history.append(row)
        if step % config.get("checkpoint_every", 500) == 0 or step == config["steps"]:
            torch.save({
                "step": step, "model": args.model, "device": str(device), "config": config,
                "network": network.state_dict(), "optimizer": optimizer.state_dict(),
                "generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "history": history,
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "effective_seconds": previous_seconds + time.perf_counter() - train_started,
            }, checkpoint_path)
        if step == 1 or step == config["steps"] or step % max(1, config["steps"] // 10) == 0:
            print(f"{args.model} step {step}/{config['steps']}: loss={row['loss']:.6g} "
                  f"Euler={row['mean_abs_euler_gap']:.6g} mass={row['max_mass_error']:.2g}", flush=True)
    if device.type == "cuda": torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    torch.save(network.state_dict(), run_dir / "policy.pt")
    network.eval()
    with torch.no_grad():
        chunks = []
        for start in range(0, len(valid_mu), config["batch_size"]):
            _, details = economy.residuals(network, valid_mu[start:start + config["batch_size"]],
                                           valid_z[start:start + config["batch_size"]])
            chunks.append((len(valid_mu[start:start + config["batch_size"]]), numeric(details)))
    total = sum(size for size, _ in chunks)
    validation = {key: sum(size * row[key] for size, row in chunks) / total for key in chunks[0][1]}
    validation["max_mass_error"] = max(row["max_mass_error"] for _, row in chunks)
    report = {
        "model": args.model,
        "config": config_path.name,
        "calibration_file": str(calibration_path.relative_to(ROOT)),
        "policy_mode": policy_mode,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_cuda": torch.version.cuda,
        "parameters": sum(p.numel() for p in network.parameters()),
        "train_seconds_this_invocation": train_seconds,
        "train_seconds_accumulated": previous_seconds + train_seconds,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "updated_steps_this_invocation": max(0, config["steps"] - start_step + 1),
        "training_mode": "economic equations only; no policy labels; Young not required",
        "validation": validation,
        "interpretation": "experimental run; independent Euler/KKT and trajectory checks are required before claiming a solution",
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
