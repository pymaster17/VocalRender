import torch
import torch.nn as nn


class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim: int = 64, scale: int = 9):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.scale = scale

        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)
    
    def forward(self, hidden, do_sample: bool = False, fsq_temperature: float = 1.0):
        """Forward pass with optional stochastic rounding.

        Args:
            hidden: Input tensor of shape (..., in_dim).
            do_sample: If True, use stochastic (probabilistic) rounding instead
                of deterministic ``torch.round``.  Only effective during eval.
            fsq_temperature: Controls randomness of stochastic rounding.
                * 1.0  – raw probability (fractional part as-is)
                * < 1.0 – more deterministic (sharper, closer to greedy)
                * > 1.0 – more random (flatter, more diversity)
        """
        hidden = self.in_proj(hidden)
        hidden = torch.tanh(hidden)

        if self.training:
            quantized = torch.round(hidden * self.scale) / self.scale
            hidden = hidden + (quantized - hidden).detach()
        else:
            if do_sample and fsq_temperature > 0.0:
                hidden = self._stochastic_round(hidden, fsq_temperature)
            else:
                hidden = torch.round(hidden * self.scale) / self.scale

        return self.out_proj(hidden)

    def _stochastic_round(self, hidden: torch.Tensor, temperature: float) -> torch.Tensor:
        """Stochastic rounding with optional temperature scaling.

        For each scalar value ``x = hidden * scale``, the probability of
        rounding *up* is the fractional part ``Δ = x - floor(x)``.  When
        ``temperature != 1.0``, we rescale the corresponding logit
        ``log(Δ / (1-Δ)) / T`` to sharpen or flatten the distribution.
        """
        scaled = hidden * self.scale
        floor_val = torch.floor(scaled)
        frac = scaled - floor_val  # ∈ [0, 1)

        if temperature != 1.0:
            # Logit-based temperature scaling
            eps = 1e-7
            frac_clamped = frac.clamp(eps, 1.0 - eps)
            logit = torch.log(frac_clamped / (1.0 - frac_clamped))
            prob_ceil = torch.sigmoid(logit / temperature)
        else:
            prob_ceil = frac

        # Bernoulli sample: 1 → round up, 0 → round down
        ceil_mask = torch.bernoulli(prob_ceil)
        quantized = (floor_val + ceil_mask) / self.scale
        return quantized