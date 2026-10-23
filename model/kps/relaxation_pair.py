"""Continuous loop/position delays with valid integer causal decoding."""

import torch
from torch import nn


class ContinuousDelayPairMixin:
    """Share continuous delay parameterization between comb and triangle models."""

    def _init_continuous_pair(self, delay_len_init: float | None,
                              position_init: float | None,
                              position_name: str) -> None:
        self._position_name = position_name
        candidates = self.position_candidates
        low = max(float(self.L_candidates[0]), float(candidates[0]) + 1)
        high = float(self.L_candidates[-1])
        random_delays = delay_len_init is None
        if random_delays:
            if position_init is not None:
                low = max(low, int(position_init) + 1)
            if low > high:
                raise ValueError("No valid L for the configured initial position.")
            delay_len_init = float(self.L_logits.new_empty(()).uniform_(low, high))
        if not low <= delay_len_init <= high:
            raise ValueError("Continuous L initialization outside valid candidate bounds.")
        if position_init is None:
            upper = min(float(candidates[-1]), int(delay_len_init) - 1)
            if random_delays:
                bound = self.L_logits.new_tensor(int(delay_len_init))
                upper = min(float(candidates[-1]), float(torch.nextafter(
                    bound, bound.new_tensor(-torch.inf))))
                position_init = float(self.L_logits.new_empty(()).uniform_(
                    float(candidates[0]), upper))
            else:
                position_init = (float(candidates[0]) + upper) / 2
        if not (float(candidates[0]) <= position_init <= float(candidates[-1])
                and int(position_init) < int(delay_len_init)):
            raise ValueError("Continuous position initialization must satisfy floor(position) < floor(L).")
        self.L_logits.requires_grad_(False)
        getattr(self, position_name + "_logits").requires_grad_(False)
        self.delay_len_parameter = nn.Parameter(self.L_logits.new_tensor(delay_len_init))
        self.position_parameter = nn.Parameter(self.L_logits.new_tensor(position_init))

    @property
    def position_candidates(self) -> torch.Tensor:
        return getattr(self, self._position_name + "_candidates")

    def continuous_L(self) -> torch.Tensor:
        return self.delay_len_parameter.clamp(
            max(float(self.L_candidates[0]), float(self.position_candidates[0]) + 1),
            float(self.L_candidates[-1]))

    def scaled_delay_len(self) -> torch.Tensor:
        return self.continuous_L()

    def continuous_position(self) -> torch.Tensor:
        # Integer causal decoding must satisfy floor(position) < floor(L),
        # including when optimization moves L across an integer boundary.
        integer_L = self.continuous_L().detach().floor()
        upper = torch.nextafter(integer_L, integer_L.new_tensor(-torch.inf))
        position = self.position_parameter.clamp(
            float(self.position_candidates[0]), float(self.position_candidates[-1]))
        return torch.minimum(position, upper)

    def selected_delays(self) -> tuple[int, int]:
        return (int(self.continuous_L().detach().floor()),
                int(self.continuous_position().detach().floor()))

    @torch.no_grad()
    def project_delays_(self) -> None:
        self.delay_len_parameter.copy_(self.continuous_L())
        self.position_parameter.copy_(self.continuous_position())
