"""Policy representations with identical economic inputs and outputs."""

import torch
from torch import nn


def mlp(in_features, hidden, out_features):
    return nn.Sequential(
        nn.Linear(in_features, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_features),
    )


class PolicyNetwork(nn.Module):
    def __init__(self, kind, asset_points, asset_max, hidden, basis,
                 macro_features="none", grid=None, policy_variant="base",
                 gated_asset_cutoff=5.0):
        super().__init__()
        self.kind = kind
        self.asset_max = asset_max
        if policy_variant not in ("base", "unemp_lowasset_gated", "employed_midasset_gated"):
            raise ValueError("unknown policy_variant")
        self.policy_variant = policy_variant
        self.gated_asset_cutoff = float(gated_asset_cutoff)
        if self.gated_asset_cutoff <= 0:
            raise ValueError("gated_asset_cutoff must be positive")
        if policy_variant == "unemp_lowasset_gated":
            self.low_asset_unemployed_coeff = nn.Parameter(torch.zeros(2))
        if policy_variant == "employed_midasset_gated":
            self.mid_asset_employed_coeff = nn.Parameter(torch.zeros(2))
        if macro_features not in ("none", "capital", "capital_only", "moments", "cdf", "cdf_capital", "cdf_capital_v2"):
            raise ValueError("unknown macro_features mode")
        if macro_features in ("capital", "capital_only", "moments", "cdf_capital", "cdf_capital_v2"):
            if grid is None or len(grid) != asset_points:
                raise ValueError("explicit macro features require the training asset grid")
            self.register_buffer("macro_grid", torch.as_tensor(grid).clone(), persistent=False)
        self.macro_features = macro_features
        extra_features = {"none": 0, "capital": 1, "capital_only": 1, "moments": 4,
                          "cdf": 0, "cdf_capital": 1, "cdf_capital_v2": 1}[macro_features]
        branch_features = (0 if macro_features == "capital_only" else 2 * asset_points) + 2 + extra_features
        if kind == "deeponet":
            self.branch = mlp(branch_features, hidden, basis)
            self.trunk = mlp(2, hidden, basis)
            self.bias = nn.Parameter(torch.zeros(()))
        elif kind == "mlp":
            self.single = mlp(branch_features + 2, hidden, 1)
        else:
            raise ValueError("model must be deeponet or mlp")

    def macro_inputs(self, mu, z):
        batch = mu.shape[0]
        z_one_hot = torch.nn.functional.one_hot(z.long(), num_classes=2).to(mu.dtype)
        # Preserve cdf_capital's historical mass+K semantics for old checkpoints.
        # Correct cumulative inputs use a new version; old weights must not be
        # reinterpreted as if they had been trained on CDF coordinates.
        distribution = torch.cumsum(mu, dim=-1) if self.macro_features in ("cdf", "cdf_capital_v2") else mu
        features = ([z_one_hot] if self.macro_features == "capital_only"
                    else [distribution.reshape(batch, -1), z_one_hot])
        if self.macro_features in ("capital", "capital_only", "moments", "cdf_capital", "cdf_capital_v2"):
            capital = (mu * self.macro_grid[None, None, :]).sum(dim=(1, 2))
            features.append((capital / self.asset_max)[:, None])
            if self.macro_features == "moments":
                variance = (mu * (self.macro_grid[None, None, :] -
                                  capital[:, None, None]).square()).sum(dim=(1, 2))
                conditional_means = ((mu * self.macro_grid[None, None, :]).sum(dim=2) /
                                     mu.sum(dim=2).clamp_min(1e-6))
                features.extend([(variance.clamp_min(0).sqrt() / self.asset_max)[:, None],
                                 conditional_means / self.asset_max])
        return torch.cat(features, dim=-1)

    def forward(self, mu, z, asset, employment):
        """Return an unbounded policy output for queried household states."""
        macro = self.macro_inputs(mu, z)
        micro = torch.stack((asset / self.asset_max, employment.to(mu.dtype)), dim=-1)
        if self.kind == "deeponet":
            branch = self.branch(macro)
            trunk = self.trunk(micro)
            output = (branch[:, None, None, :] * trunk).sum(dim=-1) / branch.shape[-1] ** 0.5 + self.bias
        else:
            expanded = macro[:, None, None, :].expand(*asset.shape, macro.shape[-1])
            output = self.single(torch.cat((expanded, micro), dim=-1)).squeeze(-1)
        if self.policy_variant == "unemp_lowasset_gated":
            gate = (1.0 - asset / self.gated_asset_cutoff).clamp(0.0, 1.0) * (employment.long() == 0).to(mu.dtype)
            local_asset = asset.clamp(0.0, self.gated_asset_cutoff) / self.gated_asset_cutoff
            output = output + gate * (self.low_asset_unemployed_coeff[0] +
                                      self.low_asset_unemployed_coeff[1] * local_asset)
        if self.policy_variant == "employed_midasset_gated":
            # A smooth tent gate confines this two-parameter correction to
            # employed households with assets strictly between 20 and 50.
            gate = torch.minimum((asset - 20.0) / 5.0, (50.0 - asset) / 5.0).clamp(0.0, 1.0)
            gate = gate * (employment.long() == 1).to(mu.dtype)
            local_asset = (asset - 35.0) / 15.0
            output = output + gate * (self.mid_asset_employed_coeff[0] +
                                      self.mid_asset_employed_coeff[1] * local_asset)
        return output
