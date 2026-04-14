#!/usr/bin/env python3
"""
Consolidated toy experiment for learning a 2D GMM prior with either vanilla RF or EC-RF,
optionally using mini-batch OT coupling, and then using the learned prior inside a
FLOWER-style Bayesian posterior sampler.

Features
--------
- Prior training with:
    * vanilla RF      : target = x1 - x0
    * EC-RF / AF-style: target = x1 - xt
- Coupling choices:
    * independent random pairing
    * mini-batch OT pairing (via torchcfm)
- Evaluation / plotting:
    * loss curve
    * prior target vs generated samples
    * density heatmaps
    * trajectories
    * vector fields
- Exact posterior for linear Gaussian measurement with GMM prior
- FLOWER-style posterior sampling with proximal refinement
- Comparison plot for exact posterior vs FLOWER gamma=0 and gamma=1

Notes
-----
1. For EC-RF / AF-style prior training, the model is interpreted as predicting x1 - xt,
   so the destination estimate is x1_hat = xt + v_theta(xt, t).
2. For vanilla RF, the model predicts x1 - x0, which is not directly a destination residual.
   In this script, FLOWER posterior sampling is only enabled for mode='ecrf'.
3. mini-batch OT requires torchcfm to be installed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchcfm
    HAS_TORCHCFM = True
except Exception:
    torchcfm = None
    HAS_TORCHCFM = False

import ot as pot


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Prior training
    mode: str = "rf"              # {'rf', 'ecrf'}
    coupling: str = "independent" # {'independent', 'mbot'}
    batch_size: int = 2048
    train_steps: int = 20_000
    lr: float = 1e-3
    hidden_dim: int = 256
    target: str = "gmm"
    radius: float = 1.0

    # Toy GMM prior
    sigma: float = 0.25  # covariance = sigma^2 * I
    means_scale: float = 0.25

    # Sampling / evaluation
    num_gen_steps: int = 200
    num_vis_samples: int = 5000
    num_traj_samples: int = 300
    eval_every: int = 1000
    log_every: int = 200
    checkpoint_every: int = 1000
    posterior_dist: str = "gmm"

    # Posterior / FLOWER toy
    posterior_hx: float = 1.5
    posterior_hy: float = 1.5
    posterior_y: float = 1.0
    posterior_sigma_n: float = 0.25
    posterior_num_steps: int = 1000
    posterior_num_samples: int = 5000

    # Paths
    out_dir: str = "flow_gmm_runs"
    run_name: str = "toy_gmm"

    # Resume
    resume: bool = False
    resume_checkpoint: str = ""

    # Plots
    vf_grid_size: int = 25
    vf_times: Optional[List[float]] = None
    plot_range: float = 1.5

    # Workflow switches
    train_prior: bool = True
    make_prior_plots: bool = True
    make_posterior_plots: bool = True


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def parse_float_list(s: str) -> List[float]:
    if not s.strip():
        return []
    return [float(v.strip()) for v in s.split(",")]


# ============================================================
# Target prior: 3-component GMM in R^2
# ============================================================

class TargetGMM:
    def __init__(self, device: str = "cpu", sigma: float = 0.25, means_scale: float = 0.25):
        self.device = device
        self.sigma = float(sigma)
        s = float(means_scale)
        self.means = torch.tensor(
            [
                [-s, -s],
                [-s,  s],
                [ s, -s],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.num_components = self.means.shape[0]

    def sample(self, n: int) -> torch.Tensor:
        # comp_ids = torch.randint(
        #     low=0,
        #     high=self.num_components,
        #     size=(n,),
        #     device=self.device,
        # )
        # Guaranteed equal counts (assumes n divisible by 3)
        base = n // self.num_components
        remainder = n % self.num_components
    
        # Give the remainder to the first `remainder` components
        counts = [base + (1 if i < remainder else 0) for i in range(self.num_components)]
        
        comp_ids = torch.cat([
            torch.full((count,), i, device=self.device)
            for i, count in enumerate(counts)
        ])
        comp_ids = comp_ids[torch.randperm(len(comp_ids), device=self.device)]
        chosen_means = self.means[comp_ids]
        noise = self.sigma * torch.randn(n, 2, device=self.device)
        return chosen_means + noise

    def cov(self) -> torch.Tensor:
        return (self.sigma ** 2) * torch.eye(2, device=self.device)


class TargetCircularGMM:
    def __init__(
        self,
        device: str = "cpu",
        radius: float = 0.5,
        sigma: float = 0.0,          # optional radial jitter (0 = perfect ring)
        means_scale: float = 0.25,
    ):
        self.device = device
        self.radius = float(radius)
        self.sigma = float(sigma)
        s = float(means_scale)
        self.means = torch.tensor(
            [
                [-s,  s],
                [ s, -s],
                [ -s, -s],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.num_components = self.means.shape[0]

    def sample(self, n: int) -> torch.Tensor:
        # Equal counts per component (handles remainder too)
        base = n // self.num_components
        remainder = n % self.num_components
        counts = [base + (1 if i < remainder else 0) for i in range(self.num_components)]

        comp_ids = torch.cat([
            torch.full((count,), i, device=self.device)
            for i, count in enumerate(counts)
        ])
        comp_ids = comp_ids[torch.randperm(len(comp_ids), device=self.device)]

        chosen_means = self.means[comp_ids]           # (n, 2)

        # Sample a uniform angle in [0, 2π) for each point
        angles = 2 * math.pi * torch.rand(n, device=self.device)  # (n,)

        # Convert to unit-circle direction vectors
        directions = torch.stack([angles.cos(), angles.sin()], dim=-1)  # (n, 2)

        # Place each point on (or near) the ring
        r = self.radius
        if self.sigma > 0:
            r = r + self.sigma * torch.randn(n, device=self.device)   # radial jitter
            r = r.unsqueeze(-1)                                        # (n, 1)

        return chosen_means + r * directions

    def cov(self) -> torch.Tensor:
        """
        Approximate marginal covariance for one component.
        For a perfect ring: Cov = (r²/2) I
        With radial jitter σ: Cov = ((r² + σ²)/2) I
        """
        var = (self.radius ** 2 + self.sigma ** 2) / 2.0
        return var * torch.eye(2, device=self.device)
# ============================================================
# Coupling
# ============================================================

def minibatch_ot_coupling(x0: torch.Tensor, x1: torch.Tensor, matcher=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Re-pair x0 and x1 inside the minibatch using exact OT.
    """
    if not HAS_TORCHCFM:
        raise ImportError("mini-batch OT coupling requested, but torchcfm is not installed.")
    if matcher is None:
        matcher = torchcfm.ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    x0_c, x1_c = matcher.ot_sampler.sample_plan(x0, x1)
    return x0_c, x1_c

def get_nn(x0, x1):
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1) ** 2
    j = M.argmin(axis=-1)
    return x0, x1[j]
# ============================================================
# Model
# ============================================================

class VelocityMLP(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


# ============================================================
# Prior training batch generation
# ============================================================

def sample_base(n: int, device: str) -> torch.Tensor:
    return torch.randn(n, 2, device=device)


def sample_training_batch(
    target_dist: TargetGMM,
    batch_size: int,
    device: str,
    coupling: str = "independent",
    ot_matcher=None,
):
    x0 = sample_base(batch_size, device)
    x1 = target_dist.sample(batch_size)

    if coupling == "mbot":
        x0, x1 = minibatch_ot_coupling(x0, x1, matcher=ot_matcher)
    elif coupling =="nn":
        x0, x1 = get_nn(x0, x1)
    elif coupling != "independent":
        raise ValueError(f"Unknown coupling mode: {coupling}")

    t = torch.rand(batch_size, 1, device=device)
    xt = (1.0 - t) * x0 + t * x1
    return xt, t, x0, x1


def training_target(mode: str, xt: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    if mode == "rf":
        return x1 - x0
    if mode == "ecrf":
        return x1 - xt
    raise ValueError(f"Unknown training mode: {mode}")


# ============================================================
# Prior sampling
# ============================================================

@torch.no_grad()
def sample_prior_flow(model: nn.Module, n: int, device: str, num_steps: int = 200, mode: str = "rf") -> torch.Tensor:
    model.eval()
    x = sample_base(n, device)

    for i in range(num_steps):
        t = torch.full((n, 1), i / num_steps, device=device)
        v = model(x, t)

        if mode == "rf":
            step_size = 1.0 / num_steps
        elif mode == "ecrf":
            step_size =  1.0 / (num_steps - i)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")

        x = x + step_size * v

    return x


@torch.no_grad()
def sample_prior_trajectories(model: nn.Module, n: int, device: str, num_steps: int = 100, mode: str = "rf"):
    model.eval()
    x = sample_base(n, device)
    traj = [x.detach().cpu().clone()]

    for i in range(num_steps):
        t = torch.full((n, 1), i / num_steps, device=device)
        v = model(x, t)

        if mode == "rf":
            step_size = 1.0 / num_steps
        elif mode == "ecrf":
            step_size = 1 / (num_steps - i)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")

        x = x + step_size * v
        traj.append(x.detach().cpu().clone())

    return traj


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, losses: List[float], config: Config) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "losses": losses,
        "config": asdict(config),
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, map_location: str = "cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0), ckpt.get("losses", []), ckpt.get("config", {})


# ============================================================
# Plotting helpers
# ============================================================

def plot_loss_curve(losses: List[float], save_path: str | Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Prior training loss")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_target_vs_generated(target_samples: np.ndarray, gen_samples: np.ndarray, save_path: str | Path, title_suffix: str = "", plot_range: float = 1.5) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(target_samples[:, 0], target_samples[:, 1], s=4, alpha=0.4)
    axes[0].set_title("Target GMM" + title_suffix)
    axes[0].set_aspect("equal")

    axes[1].scatter(gen_samples[:, 0], gen_samples[:, 1], s=4, alpha=0.4)
    axes[1].set_title("Generated prior samples" + title_suffix)
    axes[1].set_aspect("equal")

    for ax in axes:
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_density_heatmaps(target_samples: np.ndarray, gen_samples: np.ndarray, save_path: str | Path, bins: int = 120, plot_range: float = 1.5) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist2d(target_samples[:, 0], target_samples[:, 1], bins=bins, range=[[-plot_range, plot_range], [-plot_range, plot_range]])
    axes[0].set_title("Target density heatmap")
    axes[0].set_aspect("equal")

    axes[1].hist2d(gen_samples[:, 0], gen_samples[:, 1], bins=bins, range=[[-plot_range, plot_range], [-plot_range, plot_range]])
    axes[1].set_title("Generated density heatmap")
    axes[1].set_aspect("equal")

    for ax in axes:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_trajectories(traj, save_path: str | Path, max_lines: int = 100, plot_range: float = 1.5) -> None:
    n = traj[0].shape[0]
    m = min(n, max_lines)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(m):
        xs = [traj_t[i, 0].item() for traj_t in traj]
        ys = [traj_t[i, 1].item() for traj_t in traj]
        ax.plot(xs, ys, alpha=0.35, linewidth=1.0, color="purple")
        ax.scatter(xs[0], ys[0], s=8, alpha=0.5, color="red")
        ax.scatter(xs[-1], ys[-1], s=8, alpha=0.5, color="blue")

    ax.set_title("Prior sample trajectories")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


@torch.no_grad()
def plot_vector_field(model: nn.Module, save_path: str | Path, times: List[float], grid_size: int = 25, plot_range: float = 1.5) -> None:
    model.eval()
    ncols = len(times)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    xs = np.linspace(-plot_range, plot_range, grid_size)
    ys = np.linspace(-plot_range, plot_range, grid_size)
    X, Y = np.meshgrid(xs, ys)
    XY = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)

    device = next(model.parameters()).device
    XY_torch = torch.tensor(XY, dtype=torch.float32, device=device)

    for ax, tval in zip(axes, times):
        t = torch.full((XY_torch.shape[0], 1), float(tval), device=device)
        V = model(XY_torch, t)
        V = to_numpy(V)
        U = V[:, 0].reshape(grid_size, grid_size)
        W = V[:, 1].reshape(grid_size, grid_size)
        speed = np.sqrt(U ** 2 + W ** 2)

        ax.quiver(X, Y, U, W, speed)
        ax.set_title(f"v_theta(x, t={tval:.2f})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ============================================================
# Exact posterior for linear Gaussian observation + GMM prior
# ============================================================

def exact_posterior_params(means: torch.Tensor, sigma: float, h: torch.Tensor, y: float, sigma_n: float):
    device = means.device
    Sigma = (sigma ** 2) * torch.eye(2, device=device)
    h = h.reshape(2, 1)

    s2 = (h.T @ Sigma @ h).squeeze() + sigma_n ** 2
    post_cov = Sigma - (Sigma @ h @ h.T @ Sigma) / s2

    proj_means = (means @ h).squeeze(-1)
    gain = (Sigma @ h / s2).squeeze(-1)
    post_means = means + (y - proj_means).unsqueeze(-1) * gain.unsqueeze(0)

    logw = -0.5 * ((y - proj_means) ** 2) / s2 - 0.5 * torch.log(2 * torch.pi * s2)
    logw = logw - torch.logsumexp(logw, dim=0)
    post_weights = torch.exp(logw)

    return post_weights, post_means, post_cov


def sample_exact_posterior(means: torch.Tensor, sigma: float, h: torch.Tensor, y: float, sigma_n: float, n: int) -> torch.Tensor:
    post_weights, post_means, post_cov = exact_posterior_params(means, sigma, h, y, sigma_n)
    comp_ids = torch.multinomial(post_weights, num_samples=n, replacement=True)
    chosen_means = post_means[comp_ids]
    L = torch.linalg.cholesky(post_cov)
    noise = torch.randn(n, 2, device=means.device) @ L.T
    return chosen_means + noise

def circular_posterior_params(
    means: torch.Tensor,       # (K, 2)  component means
    radius: float,
    h: torch.Tensor,           # (2,)    measurement vector
    y: float,                  # scalar  observation
    sigma_n: float,            # observation noise std
    n_grid: int = 2048,        # angle grid resolution
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns posterior weights and angle distributions for each component.

    For component k, the posterior over angle θ is:
        p(θ | y, k)  ∝  exp( -(y - h·μ_k - r·(h·d(θ)))² / (2 σ_n²) )
    where d(θ) = [cos θ, sin θ].

    Returns
    -------
    comp_weights  : (K,)       posterior weight of each component
    log_angle_pdf : (K, G)     log-pdf over the angle grid for each component
                               (already normalised within each component)
    angles        : (G,)       the angle grid in [0, 2π)
    """
    device = means.device
    K = means.shape[0]
    h = h.reshape(2).to(device)
    h_norm = h.norm()
    h_unit = h / (h_norm + 1e-8)        # direction only
    sigma_n_eff = sigma_n / (h_norm + 1e-8)  # rescaled noise
    y_eff = y / (h_norm + 1e-8)              # rescaled observat

    # Angle grid  (G,)
    angles = torch.linspace(0, 2 * math.pi, n_grid + 1, device=device)[:-1]
    d = torch.stack([angles.cos(), angles.sin()], dim=-1)  # (G, 2)

    # h·d(θ)  (G,)
    h_dot_d = d @ h_unit  # (G,)

    # For each component k: residual = y - h·μ_k - r·(h·d(θ))
    h_dot_mu = means @ h_unit          # (K,)
    residuals = (y_eff - h_dot_mu).unsqueeze(-1) - radius * h_dot_d.unsqueeze(0)  # (K, G)

    # Log-likelihood over grid for each component
    log_liks = -0.5 * (residuals ** 2) / (sigma_n_eff ** 2)  # (K, G)

    # Normalise within each component → posterior angle pdf
    log_angle_pdf = log_liks - torch.logsumexp(log_liks, dim=-1, keepdim=True)  # (K, G)

    # Marginal log-likelihood per component (log-mean-exp over uniform angle prior)
    # log p(y | k) = log (1/G Σ_θ p(y|θ,k))  =  logsumexp - log G
    log_comp_liks = torch.logsumexp(log_liks, dim=-1) - math.log(n_grid)  # (K,)

    # Equal component priors → posterior weights
    log_comp_weights = log_comp_liks - torch.logsumexp(log_comp_liks, dim=0)
    comp_weights = torch.exp(log_comp_weights)  # (K,)

    return comp_weights, log_angle_pdf, angles


def sample_circular_posterior(
    comp_weights: torch.Tensor,   # (K,)
    log_angle_pdf: torch.Tensor,  # (K, G)
    angles: torch.Tensor,         # (G,)
    means: torch.Tensor,          # (K, 2)
    radius: float,
    n: int,
    sigma: float = 0.0,           # optional radial jitter
) -> torch.Tensor:
    """
    Draw n samples from the circular posterior by:
      1. Sampling a component from comp_weights.
      2. Sampling an angle from that component's discrete pdf.
      3. Converting (component, angle) → 2D point on the ring.
    """
    device = means.device
    K = comp_weights.shape[0]

    # Equal counts per component weighted by posterior
    comp_ids = torch.multinomial(comp_weights, num_samples=n, replacement=True)  # (n,)

    # For each sample, draw an angle from the component's angle distribution
    # Gather each sample's log-pdf row
    angle_log_pdfs = log_angle_pdf[comp_ids]   # (n, G)
    angle_probs = torch.exp(angle_log_pdfs)    # (n, G)  — already normalised

    sampled_angle_idx = torch.multinomial(angle_probs, num_samples=1).squeeze(-1)  # (n,)
    sampled_angles = angles[sampled_angle_idx]  # (n,)

    # Optional: add small angular jitter so samples aren't grid-locked
    dtheta = angles[1] - angles[0]
    sampled_angles = sampled_angles + dtheta * (torch.rand(n, device=device) - 0.5)

    # Convert to 2D
    directions = torch.stack([sampled_angles.cos(), sampled_angles.sin()], dim=-1)  # (n, 2)
    chosen_means = means[comp_ids]  # (n, 2)

    r = radius
    if sigma > 0:
        r = radius + sigma * torch.randn(n, device=device).unsqueeze(-1)

    return chosen_means + r * directions


# ============================================================
# FLOWER-style posterior sampler
# ============================================================

def prox_linear_likelihood(
    z: torch.Tensor,
    h: torch.Tensor,
    y: float,
    sigma_n: float,
    lam: float,
) -> torch.Tensor:
    """
    Proximal operator of F_y(x) = (1 / 2*sigma_n^2) * (h^T x - y)^2:
 
        prox_{lam * F_y}(z) = z - [lam / (sigma_n^2 + lam*|h|^2)] * (h^T z - y) * h
 
    Derivation: set gradient of  0.5||x-z||^2 + lam*F_y(x)  to zero.
    """
    if lam <= 0.0:
        return z
 
    h_row      = h.reshape(1, 2)                              # (1, 2)
    hz_minus_y = (z * h_row).sum(dim=1, keepdim=True) - y    # (n, 1)
    h_sq       = (h_row * h_row).sum()                        # scalar
    denom      = sigma_n ** 2 + lam * h_sq                    # scalar
    step       = (lam / denom) * hz_minus_y * h_row           # (n, 2)
    return z - step
#return z - step


@torch.no_grad()
def sample_flower_posterior(
    model: nn.Module,
    n: int,
    device: str,
    h: torch.Tensor,
    y: float,
    sigma_n: float,
    num_steps: int = 1000,
    gamma: float = 1.0,
    mode: str = "ecrf",
    return_traj: bool = False,
):
    """
    FLOWER posterior sampler (Algorithm 1) for a linear measurement model:
        y = h^T x1 + noise,   noise ~ N(0, sigma_n^2)
 
    Supports two model parameterisations:
        "rf"   — model predicts v ≈ x1 - x0  (vanilla rectified flow)
                 destination estimate:  x1_hat = xt + (1-t) * v
                 uncertainty schedule:  nu_t   = (1-t) / sqrt(t^2 + (1-t)^2)
 
        "ecrf" — model predicts v ≈ x1 - xt  (endpoint-conditioned RF)
                 destination estimate:  x1_hat = xt + v
                 uncertainty schedule:  nu_t   = 1   / sqrt(t^2 + (1-t)^2)
                 (the (1-t) numerator of the RF formula drops out because the
                  ECRF model already absorbs that scaling into its prediction)
 
    Both modes share the same interpolant xt = (1-t)*x0 + t*x1 and the same
    Steps 2 and 3 from Algorithm 1.
 
    Args:
        model:      pretrained velocity network, callable as model(x, t)
        n:          number of posterior samples to draw
        device:     torch device string
        h:          measurement vector, shape (2,)
        y:          scalar measurement value
        sigma_n:    measurement noise standard deviation
        num_steps:  number of discretisation steps N
        gamma:      uncertainty flag in {0, 1}; 1 enables stochastic refinement
        mode:       "rf" or "ecrf"
        return_traj: if True, return list of intermediate states as well
 
    Returns:
        x:    posterior samples at t=1, shape (n, 2)
        traj: list of intermediate tensors (only if return_traj=True)
    """
    model.eval()
 
    dt = 1.0 / num_steps
 
    # Pre-compute measurement geometry (same for all steps)
    h_row = h.reshape(1, 2)          # (1, 2)  — row vector for dot products
    h_col = h.reshape(2, 1)          # (2, 1)  — column vector
    HtH   = h_col @ h_col.T         # (2, 2)  — H^T H for 1-D measurement
 
    # ------------------------------------------------------------------ #
    # Initialise: x0 ~ p_{X0} = N(0, I)   [Algorithm 1, line 2]
    # ------------------------------------------------------------------ #
    x    = torch.randn(n, 2, device=device)
    traj = [x.detach().cpu().clone()] if return_traj else None
 
    for k in range(num_steps):
        t_scalar = k * dt
        t        = torch.full((n, 1), t_scalar, device=device)
        
 
        # ---------------------------------------------------------- #
        # Step 1 — Destination estimate  [Algorithm 1, line 5]
        # ---------------------------------------------------------- #
        v = model(x, t)
 
        if mode == "rf":
            # v ≈ x1 - x0  =>  x1_hat = xt + (1-t)*v
            x1_hat = x + (1.0 - t_scalar) * v
 
            # nu_t from Proposition 1 (paper eq.)
            nu_t = (1.0 - t_scalar) / math.sqrt(
                t_scalar ** 2 + (1.0 - t_scalar) ** 2 + 1e-12
            )
 
        elif mode == "ecrf":
            # v ≈ x1 - xt  =>  x1_hat = xt + v
            # if t_scalar > 0.8:
            #     t = torch.full((n, 1), 0.8, device=device)
            # v = model(x, t)
            x1_hat = x + v
 
            # Same Proposition-1 derivation but the (1-t) numerator
            # drops out because the ECRF prediction already scales with it.
            # nu_t = (1.0 - t_scalar) / math.sqrt(
            #     t_scalar ** 2 + (1.0 - t_scalar) ** 2 + 1e-12
            # )
            nu_t = (1.0 - t_scalar) / math.sqrt(
                t_scalar ** 2 + (1.0 - t_scalar) ** 2 + 1e-12
            )
 
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'rf' or 'ecrf'.")
 
        # ---------------------------------------------------------- #
        # Step 2 — Posterior refinement  [Algorithm 1, lines 7-10]
        #
        # Gaussian approximation:  p(x1 | xt, y) ≈ N(mu_t, Sigma_t)
        #
        #   Sigma_t = (nu_t^{-2} I + sigma_n^{-2} H^T H)^{-1}
        #   mu_t    = prox_{nu_t^2 * F_y}(x1_hat)
        #           = argmin_x  0.5||x - x1_hat||^2 + nu_t^2 * F_y(x)
        # ---------------------------------------------------------- #
        lam  = nu_t ** 2   # proximal weight = nu_t^2
 
        # Refinement mean via proximal operator
        mu_t = prox_linear_likelihood(
            z=x1_hat, h=h, y=y, sigma_n=sigma_n, lam=lam
        )
 
        if gamma > 0.0:
            # Posterior covariance  [Algorithm 1, line 8]
            A       = (1.0 / nu_t ** 2) * torch.eye(2, device=device) \
                    + (1.0 / sigma_n ** 2) * HtH
            Sigma_t = torch.linalg.inv(A)
 
            # Reparameterised sample from N(0, Sigma_t)  [Algorithm 1, line 9]
            #   kappa_t = Sigma_t (nu_t^{-1} eps1 + sigma_n^{-1} H^T eps2)
            # Covariance check:
            #   Cov[kappa_t] = Sigma_t (nu_t^{-2} I + sigma_n^{-2} H^T H) Sigma_t
            #                = Sigma_t A Sigma_t = Sigma_t  ✓
            eps1       = torch.randn(n, 2, device=device)          # N(0, I_d)
            eps2       = torch.randn(n, 1, device=device)          # N(0, I_M)
            noise_term = (1.0 / nu_t) * eps1 \
                       + (1.0 / sigma_n) * (eps2 * h_row)         # (n, 2)
            kappa_t    = noise_term @ Sigma_t.T                    # (n, 2)
            x1_tilde = mu_t + gamma * kappa_t  # [Algorithm 1, line 10]
        else:
            x1_tilde = mu_t
 
        # ---------------------------------------------------------- #
        # Step 3 — Time progression  [Algorithm 1, lines 11-12]
        #
        #   x_{t+dt} = (1 - t - dt) * eps  +  (t + dt) * x1_tilde
        #
        # eps is resampled fresh each step (per Algorithm 1, line 11).
        # ---------------------------------------------------------- #
        eps = torch.randn(n, 2, device=device)

        # if mode == "ecrf":
        #     if k != num_steps:
        #         step_size = (1.0 - t_scalar) / (num_steps - k)
        #     else:
        #         step_size = (1.0 - t_scalar) / (num_steps - k)

        #     x   = (1.0 - t_scalar - step_size) * eps + (t_scalar + step_size) * x1_tilde
        # else:
        x   = (1.0 - t_scalar - dt) * eps + (t_scalar + dt) * x1_tilde
        if k % 200 == 0:
            print("X: ", x[0])
            print("X1_ti: ", x1_tilde[0])
            print("X1_hat: ", x1_hat[0])
        if return_traj:
            traj.append(x.detach().cpu().clone())
 
    if return_traj:
        return x, traj
    return x


# ============================================================
# Evaluation bundles
# ============================================================

@torch.no_grad()
def run_prior_evaluation(model: nn.Module, target_dist: TargetGMM, out_dir: str | Path, step: int, device: str, num_vis_samples: int, num_gen_steps: int, num_traj_samples: int, vf_times: List[float], vf_grid_size: int, plot_range: float, mode: str) -> None:
    eval_dir = Path(out_dir) / "prior_eval"
    ensure_dir(eval_dir)

    target_samples = to_numpy(target_dist.sample(num_vis_samples))
    gen_samples = to_numpy(sample_prior_flow(model, num_vis_samples, device, num_steps=num_gen_steps, mode=mode))
    traj = sample_prior_trajectories(model, num_traj_samples, device, num_steps=100, mode=mode)

    plot_target_vs_generated(target_samples, gen_samples, eval_dir / f"samples_step_{step:06d}.png", title_suffix=f" (step {step})", plot_range=plot_range)
    plot_density_heatmaps(target_samples, gen_samples, eval_dir / f"heatmap_step_{step:06d}.png", plot_range=plot_range)
    plot_trajectories(traj, eval_dir / f"traj_step_{step:06d}.png", max_lines=100, plot_range=plot_range)
    plot_vector_field(model, eval_dir / f"vector_field_step_{step:06d}.png", times=vf_times, grid_size=vf_grid_size, plot_range=plot_range)

    np.save(eval_dir / f"target_samples_step_{step:06d}.npy", target_samples)
    np.save(eval_dir / f"gen_samples_step_{step:06d}.npy", gen_samples)


@torch.no_grad()
def plot_posterior_comparison(model: nn.Module, target_dist: TargetGMM, save_path: str | Path, h: torch.Tensor, y: float, sigma_n: float, num_samples: int = 5000, num_steps: int = 1000, mode: str = "ecrf", plot_range: float = 1.5, post_target:str ="gmm") -> None:
    
    if post_target == "gmm":
        exact_samples = sample_exact_posterior(
            means=target_dist.means,
            sigma=target_dist.sigma,
            h=h.to(target_dist.means.device),
            y=y,
            sigma_n=sigma_n,
            n=num_samples,
        ).cpu().numpy()
    elif post_target == "circle":
        comp_weights, log_angle_pdf, angles = circular_posterior_params(
        means=target_dist.means,
        radius=target_dist.radius,
        h=h.to(target_dist.means.device),
        y=y,
        sigma_n=sigma_n,
    )
        exact_samples = sample_circular_posterior(
            comp_weights=comp_weights,
            log_angle_pdf=log_angle_pdf,
            angles=angles,
            means=target_dist.means,
            radius=target_dist.radius,
            n=num_samples,
            sigma=target_dist.sigma,
        ).cpu().numpy()
    else:
        raise NotImplementedError("AHHHH")
    if mode == "ecrf" or mode =="rf":
        flower_g0 = sample_flower_posterior(
            model=model,
            n=num_samples,
            device=target_dist.means.device,
            h=h.to(target_dist.means.device),
            y=y,
            sigma_n=sigma_n,
            num_steps=num_steps,
            gamma=0.0,
            mode=mode,
        ).cpu().numpy()

        flower_g1 = sample_flower_posterior(
            model=model,
            n=num_samples,
            device=target_dist.means.device,
            h=h.to(target_dist.means.device),
            y=y,
            sigma_n=sigma_n,
            num_steps=num_steps,
            gamma=1.0,
            mode=mode,
        ).cpu().numpy()
    else:
        flower_g0 = np.zeros_like(exact_samples)
        flower_g1 = np.zeros_like(exact_samples)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(exact_samples[:, 0], exact_samples[:, 1], s=4, alpha=0.35)
    axes[0].set_title("Exact posterior")

    if mode == "ecrf" or mode == "rf":
        axes[1].scatter(flower_g0[:, 0], flower_g0[:, 1], s=4, alpha=0.35)
        axes[1].set_title("FLOWER, gamma = 0")

        axes[2].scatter(flower_g1[:, 0], flower_g1[:, 1], s=4, alpha=0.35)
        axes[2].set_title("FLOWER, gamma = 1")
    else:
        axes[1].text(0.5, 0.5, "FLOWER not defined\nfor vanilla RF", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("FLOWER, gamma = 0")
        axes[2].text(0.5, 0.5, "FLOWER not defined\nfor vanilla RF", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("FLOWER, gamma = 1")

    h_np = h.detach().cpu().numpy()
    xs = np.linspace(-plot_range, plot_range, 200)
    if abs(h_np[1]) > 1e-8:
        ys = (y - h_np[0] * xs) / h_np[1]
        for ax in axes:
            ax.plot(xs, ys, linestyle="--", linewidth=1)
    else:
        for ax in axes:
            ax.axvline(y / h_np[0], linestyle="--", linewidth=1)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ============================================================
# Training
# ============================================================

def train_prior(config: Config):
    set_seed(config.seed)

    if config.vf_times is None:
        config.vf_times = [0.0, 0.25, 0.5, 0.75, 1.0]

    run_dir = Path(config.out_dir) / config.run_name
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    if config.target == "circle":
        target_dist = TargetCircularGMM(device=config.device, sigma=config.sigma, means_scale=config.means_scale, radius=config.radius)
    else:
        target_dist = TargetGMM(device=config.device, sigma=config.sigma, means_scale=config.means_scale)
    model = VelocityMLP(hidden_dim=config.hidden_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    ot_matcher = torchcfm.ExactOptimalTransportConditionalFlowMatcher(sigma=0.0) if (config.coupling == "mbot" and HAS_TORCHCFM) else None

    losses: List[float] = []
    start_step = 0

    if config.resume and config.resume_checkpoint:
        start_step, losses, _ = load_checkpoint(config.resume_checkpoint, model, optimizer=optimizer, map_location=config.device)
        print(f"Resumed from checkpoint: {config.resume_checkpoint}")
        print(f"Starting at step {start_step}")

    model.train()
    for step in range(start_step + 1, config.train_steps + 1):
        xt, t, x0, x1 = sample_training_batch(
            target_dist=target_dist,
            batch_size=config.batch_size,
            device=config.device,
            coupling=config.coupling,
            ot_matcher=ot_matcher,
        )

        pred = model(xt, t)
        target = training_target(config.mode, xt, x0, x1)
        if config.mode == "ecrf":
            # t_clamped = t.clamp(max=0.95)
            weight = 1.0 / ((1 - t).pow(2).clamp(0.01, 1))
            loss = ((weight * ((pred - target).pow(2)).sum(dim=1)).sum() / weight.sum())
            # weight_corr = 1.0 / (target.norm(dim=1).pow(2) * (1 - t).pow(2) + eps)
            # loss = (weight_corr * ((pred - target).pow(2)).norm(dim=1)).sum() / weight_corr.sum()
            # loss = F.mse_loss(pred, target)
        elif config.mode == "rf":
            loss = F.mse_loss(pred, target)
        else:
            raise NotImplementedError(f"Did not implement {config.mode}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % config.log_every == 0 or step == 1:
            print(f"step {step:6d} | loss = {loss.item():.6f}")

        if step % config.checkpoint_every == 0 or step == config.train_steps:
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, step, losses, config)
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, step, losses, config)

        if config.make_prior_plots and (step % config.eval_every == 0 or step == config.train_steps):
            plot_loss_curve(losses, run_dir / "loss_curve.png")
            run_prior_evaluation(
                model=model,
                target_dist=target_dist,
                out_dir=run_dir,
                step=step,
                device=config.device,
                num_vis_samples=config.num_vis_samples,
                num_gen_steps=config.num_gen_steps,
                num_traj_samples=config.num_traj_samples,
                vf_times=config.vf_times,
                vf_grid_size=config.vf_grid_size,
                plot_range=config.plot_range,
                mode=config.mode,
            )

    return model, target_dist, losses, run_dir


# ============================================================
# CLI
# ============================================================

def build_argparser():
    parser = argparse.ArgumentParser(description="Train / evaluate a 2D toy GMM prior with RF or EC-RF, with optional mini-batch OT, and run FLOWER posterior sampling.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    parser.add_argument("--mode", type=str, default="rf", choices=["rf", "ecrf"])
    parser.add_argument("--coupling", type=str, default="independent", choices=["independent", "mbot", "nn"])
    parser.add_argument("--target", type=str, default="gmm", choices=["gmm", "circle", "nn"])

    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--means-scale", type=float, default=0.25)

    parser.add_argument("--num-gen-steps", type=int, default=200)
    parser.add_argument("--num-vis-samples", type=int, default=5000)
    parser.add_argument("--num-traj-samples", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=1000)

    parser.add_argument("--posterior-hx", type=float, default=1.5)
    parser.add_argument("--posterior-hy", type=float, default=1.5)
    parser.add_argument("--posterior-y", type=float, default=1.0)
    parser.add_argument("--posterior-sigma-n", type=float, default=0.25)
    parser.add_argument("--posterior-num-steps", type=int, default=1000)
    parser.add_argument("--posterior-num-samples", type=int, default=5000)
    parser.add_argument("--posterior-dist", type=str, default="gmm")

    parser.add_argument("--out-dir", type=str, default="flow_gmm_runs")
    parser.add_argument("--run-name", type=str, default="toy_gmm")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=str, default="")

    parser.add_argument("--vf-grid-size", type=int, default=25)
    parser.add_argument("--vf-times", type=str, default="0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--plot-range", type=float, default=1.5)

    parser.add_argument("--skip-train-prior", action="store_true")
    parser.add_argument("--skip-prior-plots", action="store_true")
    parser.add_argument("--skip-posterior-plots", action="store_true")

    parser.add_argument("--radius", type=float, default=1.00)
    return parser


def config_from_args(args) -> Config:
    return Config(
        seed=args.seed,
        device=args.device,
        mode=args.mode,
        coupling=args.coupling,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        sigma=args.sigma,
        means_scale=args.means_scale,
        num_gen_steps=args.num_gen_steps,
        num_vis_samples=args.num_vis_samples,
        num_traj_samples=args.num_traj_samples,
        eval_every=args.eval_every,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        posterior_hx=args.posterior_hx,
        posterior_hy=args.posterior_hy,
        posterior_y=args.posterior_y,
        posterior_sigma_n=args.posterior_sigma_n,
        posterior_num_steps=args.posterior_num_steps,
        posterior_num_samples=args.posterior_num_samples,
        out_dir=args.out_dir,
        run_name=args.run_name,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
        vf_grid_size=args.vf_grid_size,
        vf_times=parse_float_list(args.vf_times),
        plot_range=args.plot_range,
        train_prior=not args.skip_train_prior,
        make_prior_plots=not args.skip_prior_plots,
        make_posterior_plots=not args.skip_posterior_plots,
        target= args.target,
        radius = args.radius,
        posterior_dist = args.posterior_dist,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = build_argparser()
    args = parser.parse_args()
    config = config_from_args(args)

    if config.coupling == "mbot" and not HAS_TORCHCFM:
        raise ImportError("--coupling mbot requested, but torchcfm is not installed.")

    print("Configuration:")
    print(json.dumps(asdict(config), indent=2))

    run_dir = Path(config.out_dir) / config.run_name
    ensure_dir(run_dir)

    if config.train_prior:
        model, target_dist, losses, run_dir = train_prior(config)
    else:
        if config.target == "gmm":
            target_dist = TargetGMM(device=config.device, sigma=config.sigma, means_scale=config.means_scale)
        elif config.target == "circle":
            target_dist = TargetCircularGMM(device=config.device, sigma=config.sigma, means_scale=config.means_scale, radius=config.radius)
        model = VelocityMLP(hidden_dim=config.hidden_dim).to(config.device)
        if not config.resume_checkpoint:
            raise ValueError("When --skip-train-prior is used, provide --resume-checkpoint to load a trained model.")
        _, _, _ = load_checkpoint(config.resume_checkpoint, model, optimizer=None, map_location=config.device)
        print(f"Loaded prior model from: {config.resume_checkpoint}")

    if config.make_posterior_plots:
        run_prior_evaluation(
                model=model,
                target_dist=target_dist,
                out_dir=run_dir,
                step=20000,
                device=config.device,
                num_vis_samples=config.num_vis_samples,
                num_gen_steps=config.num_gen_steps,
                num_traj_samples=config.num_traj_samples,
                vf_times=config.vf_times,
                vf_grid_size=config.vf_grid_size,
                plot_range=config.plot_range,
                mode=config.mode,)
        h = torch.tensor([config.posterior_hx, config.posterior_hy], dtype=torch.float32, device=config.device)
        plot_posterior_comparison(
            model=model,
            target_dist=target_dist,
            save_path=run_dir / "posterior_comparison.png",
            h=h,
            y=config.posterior_y,
            sigma_n=config.posterior_sigma_n,
            num_samples=config.posterior_num_samples,
            num_steps=config.posterior_num_steps,
            mode=config.mode,
            plot_range=config.plot_range,
            post_target=config.posterior_dist
        )
        print(f"Posterior comparison saved to: {run_dir / 'posterior_comparison.png'}")

    print(f"\nRun complete. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
