"""Discrete-distribution KS economy and equation-informed training residual."""

import json
from pathlib import Path

import torch


class KSEconomy:
    def __init__(self, calibration_path, grid, device, policy_mode="sigmoid",
                 low_asset_multiplier=1.0, low_asset_cutoff=5.0,
                 complementarity_asset_scale=None, mass_weight=0.5,
                 high_asset_multiplier=1.0, high_asset_cutoff=20.0):
        raw = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        self.grid = grid.to(device)
        self.alpha = raw["alpha"]
        self.beta = raw["beta"]
        self.delta = raw["delta"]
        self.insurance = raw["unemployment_insurance_rate"]
        self.hours = raw["hours_if_employed"]
        self.z_values = torch.tensor(raw["z_values"], dtype=grid.dtype, device=device)
        self.transition = torch.tensor(raw["joint_transition"], dtype=grid.dtype, device=device)
        if not torch.allclose(self.transition.sum(dim=1),
                              torch.ones(4, dtype=grid.dtype, device=device), atol=2e-4):
            raise ValueError("joint transition rows must sum to one")
        self.asset_max = float(grid[-1])
        if policy_mode not in ("sigmoid", "hard_sigmoid", "consumption_softplus"):
            raise ValueError("policy_mode must be sigmoid, hard_sigmoid, or consumption_softplus")
        self.policy_mode = policy_mode
        if low_asset_multiplier < 1.0 or low_asset_cutoff < 0:
            raise ValueError("invalid low-asset loss weighting")
        self.low_asset_multiplier = float(low_asset_multiplier)
        self.low_asset_cutoff = float(low_asset_cutoff)
        if high_asset_multiplier < 1.0 or high_asset_cutoff < 0:
            raise ValueError("invalid high-asset loss weighting")
        self.high_asset_multiplier = float(high_asset_multiplier)
        self.high_asset_cutoff = float(high_asset_cutoff)
        if not 0.0 <= mass_weight <= 1.0:
            raise ValueError("mass_weight must lie in [0, 1]")
        self.mass_weight = float(mass_weight)
        self.complementarity_asset_scale = (self.asset_max if complementarity_asset_scale is None
                                            else float(complementarity_asset_scale))
        if self.complementarity_asset_scale <= 0:
            raise ValueError("complementarity asset scale must be positive")

    def queries(self, batch):
        assets = self.grid[None, None, :].expand(batch, 2, -1)
        employment = torch.arange(2, device=self.grid.device)[None, :, None].expand_as(assets)
        return assets, employment

    def prices(self, mu, z):
        capital = (mu * self.grid[None, None, :]).sum(dim=(1, 2)).clamp_min(0.2)
        unemployed = mu[:, 0, :].sum(dim=1)
        employed = mu[:, 1, :].sum(dim=1)
        labor = (employed * self.hours).clamp_min(0.05)
        productivity = self.z_values[z.long()]
        ratio = capital / labor
        wage = (1.0 - self.alpha) * productivity * ratio ** self.alpha
        interest = self.alpha * productivity * ratio ** (self.alpha - 1.0) - self.delta
        tax = self.insurance * unemployed / labor
        return capital, wage, interest, tax

    def cash(self, mu, z, assets, employment):
        _, wage, interest, tax = self.prices(mu, z)
        work_income = (1.0 - tax) * self.hours * wage
        benefit = self.insurance * wage
        income = torch.where(employment.long() == 1,
                             work_income[:, None, None], benefit[:, None, None])
        return (1.0 + interest[:, None, None]) * assets + income

    def policy(self, network, mu, z, assets, employment):
        cash = self.cash(mu, z, assets, employment)
        upper = torch.minimum((cash - 1e-4).clamp_min(0.0),
                              torch.full_like(cash, self.asset_max))
        raw = network(mu, z, assets, employment)
        if self.policy_mode == "consumption_softplus":
            desired_consumption = torch.nn.functional.softplus(raw) + 1e-4
            savings = (cash - desired_consumption).clamp_min(0.0)
            savings = torch.minimum(savings, upper)
            return savings, cash - savings
        if self.policy_mode == "hard_sigmoid":
            # Projected linear sigmoid: finite logits can attain the exact
            # borrowing bound. Its flat regions require separate diagnostics.
            share = (0.5 + raw / 4.0).clamp(0.0, 1.0)
        else:
            # Retained for interpreting checkpoints from the original pilot.
            share = torch.sigmoid(raw)
        savings = share * upper
        consumption = cash - savings
        return savings, consumption

    def aggregate_probability(self, z, next_z):
        first = self.transition[z.long() * 2, next_z * 2:next_z * 2 + 2].sum(dim=-1)
        second = self.transition[z.long() * 2 + 1, next_z * 2:next_z * 2 + 2].sum(dim=-1)
        if not torch.allclose(first, second, atol=2e-4):
            raise ValueError("aggregate transition differs across current employment states")
        return first

    def push_distribution(self, mu, z, savings, next_z):
        """Conditional next distribution given realized aggregate shock next_z."""
        aggregate_prob = self.aggregate_probability(z, next_z)
        hi = torch.searchsorted(self.grid.contiguous(), savings.detach().contiguous()).clamp(1, self.grid.numel() - 1)
        lo = hi - 1
        fraction = ((savings - self.grid[lo]) / (self.grid[hi] - self.grid[lo])).clamp(0.0, 1.0)
        by_employment = []
        for next_e in range(2):
            next_mass = torch.zeros_like(mu[:, 0, :])
            for current_e in range(2):
                joint = self.transition[z.long() * 2 + current_e, next_z * 2 + next_e]
                conditional = joint / aggregate_prob
                mass = mu[:, current_e, :] * conditional[:, None]
                next_mass = next_mass.scatter_add(1, lo[:, current_e, :], mass * (1.0 - fraction[:, current_e, :]))
                next_mass = next_mass.scatter_add(1, hi[:, current_e, :], mass * fraction[:, current_e, :])
            by_employment.append(next_mass)
        return torch.stack(by_employment, dim=1)

    def euler_terms(self, network, mu, z, policy_fn=None):
        if policy_fn is None:
            policy_fn = self.policy
        batch = mu.shape[0]
        assets, employment = self.queries(batch)
        savings, consumption = policy_fn(network, mu, z, assets, employment)
        expected = torch.zeros_like(consumption)
        next_masses = []
        for next_z in range(2):
            next_mu = self.push_distribution(mu, z, savings, next_z)
            next_masses.append(next_mu)
            next_z_tensor = torch.full_like(z, next_z)
            _, _, next_interest, _ = self.prices(next_mu, next_z_tensor)
            for next_e in range(2):
                next_employment = torch.full_like(employment, next_e)
                _, next_consumption = policy_fn(network, next_mu, next_z_tensor, savings, next_employment)
                probability_unemployed = self.transition[z.long() * 2, next_z * 2 + next_e]
                probability_employed = self.transition[z.long() * 2 + 1, next_z * 2 + next_e]
                probability = torch.stack((probability_unemployed, probability_employed), dim=1)
                expected = expected + probability[:, :, None] * (1.0 + next_interest[:, None, None]) / next_consumption
        # At an interior choice, gap=0. At the borrowing bound, gap>=0.
        gap = 1.0 - self.beta * expected * consumption
        return savings, consumption, gap, next_masses

    def residuals(self, network, mu, z, policy_fn=None):
        savings, consumption, gap, next_masses = self.euler_terms(
            network, mu, z, policy_fn=policy_fn)
        scaled_savings = savings / self.complementarity_asset_scale
        fb = torch.sqrt(gap.square() + scaled_savings.square() + 1e-12) - gap - scaled_savings
        sample_weight = (self.mass_weight * mu +
                         (1.0 - self.mass_weight) / (2 * self.grid.numel()))
        if self.low_asset_multiplier != 1.0 or self.high_asset_multiplier != 1.0:
            multiplier = ((1.0 + (self.low_asset_multiplier - 1.0) *
                           (self.grid <= self.low_asset_cutoff).to(mu.dtype)) *
                          (1.0 + (self.high_asset_multiplier - 1.0) *
                           (self.grid >= self.high_asset_cutoff).to(mu.dtype)))[None, None, :]
            sample_weight = sample_weight * multiplier
            sample_weight = sample_weight / sample_weight.sum(dim=(1, 2), keepdim=True)
        soft_interior = savings / (savings + 0.1)
        loss = (sample_weight * (fb.square() + 0.5 * soft_interior * gap.square())).sum(dim=(1, 2)).mean()
        interior = (savings > 0.1).to(mu.dtype)
        interior_weight = sample_weight * interior
        diagnostics = {
            "loss": loss,
            "mean_abs_euler_gap": (sample_weight * gap.abs()).sum(dim=(1, 2)).mean(),
            "interior_abs_euler_gap": ((interior_weight * gap.abs()).sum(dim=(1, 2)) /
                                        interior_weight.sum(dim=(1, 2)).clamp_min(1e-8)).mean(),
            "kkt_violation": (sample_weight * torch.relu(-gap)).sum(dim=(1, 2)).mean(),
            "max_mass_error": torch.stack([
                (next_mu.sum(dim=(1, 2)) - 1.0).abs() for next_mu in next_masses
            ]).max(),
            "min_consumption": consumption.min(),
            "mean_savings": (mu * savings).sum(dim=(1, 2)).mean(),
            "borrowing_share": (mu * (savings < 0.01)).sum(dim=(1, 2)).mean(),
            "exact_borrowing_share": (mu * (savings == 0)).sum(dim=(1, 2)).mean(),
            "mean_capital": (mu * self.grid[None, None, :]).sum(dim=(1, 2)).mean(),
            "mean_next_capital": (mu * savings).sum(dim=(1, 2)).mean(),
        }
        return loss, diagnostics
