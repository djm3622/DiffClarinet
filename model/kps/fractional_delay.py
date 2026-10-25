"""Continuous loop/position delays synthesized with Lagrange fractional delays.

The interpolator is the fifth-order centered Lagrange FIR used in the
2026-10-23 local fractional-delay experiment, adapted from the diffKS
implementation (github.com/ptablasdpaula/diffKS_torchLPC). Integer estimates
are decoded by rounding both delays.
"""

import torch
from torch.nn import functional as F

from .relaxation_pair import ContinuousDelayPairMixin

LAGRANGE_ORDER = 5


def lagrange_delay_fir(delay: torch.Tensor, size: int) -> torch.Tensor:
    """Fifth-order centered Lagrange FIR, shifted at the causal left boundary.

    At an integer delay the taps are exactly a unit impulse at that delay.
    """
    nodes = torch.arange(LAGRANGE_ORDER + 1, dtype=delay.dtype, device=delay.device)
    start = max(0, int(delay.detach().floor()) - LAGRANGE_ORDER // 2)
    u = delay - start
    weights = torch.stack([
        torch.prod((u - nodes[nodes != k]) / (k - nodes[nodes != k]))
        for k in range(LAGRANGE_ORDER + 1)
    ])
    return F.pad(weights, (start, size - start - LAGRANGE_ORDER - 1))


def delayed_loop_filter(numerator: torch.Tensor, delay: torch.Tensor,
                        size: int) -> torch.Tensor:
    """Convolve the three-tap loop numerator with a fractional delay."""
    fir = lagrange_delay_fir(delay, size)
    return sum(numerator[i] * F.pad(fir, (i, 0))[:size] for i in range(3))


class LagrangeDelayPairMixin(ContinuousDelayPairMixin):
    """Continuous L and position; integer decoding rounds both delays."""

    def continuous_position(self) -> torch.Tensor:
        # Rounded decoding must satisfy round(position) < round(L).
        upper = self.continuous_L().detach().round() - 1
        position = self.position_parameter.clamp(
            float(self.position_candidates[0]), float(self.position_candidates[-1]))
        return torch.minimum(position, upper)

    def selected_delays(self) -> tuple[int, int]:
        return (int(self.continuous_L().detach().round()),
                int(self.continuous_position().detach().round()))

    def synthesis_delays(self, straight_through: bool = False
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fractional delays, or rounded delays with straight-through gradients."""
        L, position = self.continuous_L(), self.continuous_position()
        if straight_through:
            L = L + (L.round() - L).detach()
            position = position + (position.round() - position).detach()
        return L, position

    def _loop_coefficients(self, L: torch.Tensor, size: int):
        K, a = self.scaled_gain(), self.scaled_allplus()
        numerator = torch.stack((a / 2, (a + 1) / 2, a * 0 + 0.5))
        denominator = (F.pad(torch.stack((a * 0 + 1, a)), (0, size - 2))
                       - K * delayed_loop_filter(numerator, L, size))
        return K, numerator, denominator
