"""Causal Karplus--Strong model with a discrete pluck-position delay."""

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from .dkps_fixed import KarplusStrongFixed
from .relaxation_pair import ContinuousDelayPairMixin


class KarplusStrongPluck(KarplusStrongFixed):
    """Learn integer loop delay L and pluck-filter delay dp from a unit impulse.

    The MATLAB generator uses input -K * (delta[n] - delta[n-dp]) and
    feedback +K * y[n-L]. Candidate pairs with dp >= L are excluded.
    """

    def __init__(
        self,
        delay_len_min: int,
        delay_len_max: int,
        dp_min: int,
        dp_max: int,
        **kwargs,
    ) -> None:
        if delay_len_min < 1 or delay_len_min >= delay_len_max:
            raise ValueError("Invalid loop-delay range.")
        if dp_min < 1 or dp_min >= dp_max or dp_max >= delay_len_max:
            raise ValueError("Invalid pluck-delay range.")
        super().__init__(delay_len=delay_len_min, **kwargs)
        if not self.all_plus:
            raise ValueError("The pluck model requires the all-pass loop filter.")
        self.register_buffer(
            "L_candidates", torch.arange(delay_len_min, delay_len_max + 1)
        )
        self.register_buffer("dp_candidates", torch.arange(dp_min, dp_max + 1))
        self.L_logits = nn.Parameter(torch.zeros(self.L_candidates.numel()))
        self.dp_logits = nn.Parameter(torch.zeros(self.dp_candidates.numel()))
        if kwargs.get("random_init", True):
            nn.init.uniform_(self.L_logits, -1e-3, 1e-3)
            nn.init.uniform_(self.dp_logits, -1e-3, 1e-3)
        self.fixed_L: int | None = None

    def _joint_logits(self) -> torch.Tensor:
        logits = self.L_logits[:, None] + self.dp_logits[None, :]
        valid = self.dp_candidates[None, :] < self.L_candidates[:, None]
        if self.fixed_L is not None:
            valid = valid & (self.L_candidates[:, None] == self.fixed_L)
        return logits.masked_fill(~valid, -torch.inf).flatten()

    def delay_distribution(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self._joint_logits())

    def valid_pair_count(self) -> int:
        return int(torch.isfinite(self._joint_logits()).sum().item())

    def _decode_index(self, index: int) -> tuple[int, int]:
        dp_count = self.dp_candidates.numel()
        return (
            int(self.L_candidates[index // dp_count].item()),
            int(self.dp_candidates[index % dp_count].item()),
        )

    def selected_delays(self) -> tuple[int, int]:
        return self._decode_index(int(torch.argmax(self._joint_logits()).item()))

    def all_pairs(self) -> list[tuple[int, int]]:
        return [self._decode_index(int(index)) for index in
                torch.isfinite(self._joint_logits()).nonzero().flatten()]

    def select_pair(self, pair: tuple[int, int]) -> None:
        if pair not in self.all_pairs():
            raise ValueError("Pair is outside the valid candidate grid.")
        self.set_selected_delays(*pair)

    def scaled_delay_len(self) -> int:
        return self.selected_delays()[0]

    def scaled_dp(self) -> int:
        return self.selected_delays()[1]

    def sample_delay_pairs(
        self, num_samples: int
    ) -> tuple[list[tuple[int, int]], torch.Tensor]:
        distribution = self.delay_distribution()
        indices = distribution.sample((num_samples,))
        return (
            [self._decode_index(int(index.item())) for index in indices],
            distribution.log_prob(indices),
        )

    def sample_gumbel_pair(
        self, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temperature <= 0:
            raise ValueError("Gumbel temperature must be positive.")
        selection = F.gumbel_softmax(
            self._joint_logits(), tau=temperature, hard=True
        )
        L_values = self.L_candidates[:, None].expand(
            -1, self.dp_candidates.numel()
        ).flatten().to(selection.dtype)
        dp_values = self.dp_candidates[None, :].expand(
            self.L_candidates.numel(), -1
        ).flatten().to(selection.dtype)
        return (selection * L_values).sum(), (selection * dp_values).sum()

    def set_selected_delays(self, L: int, dp: int) -> None:
        self._resolve_delays(L, dp)
        with torch.no_grad():
            self.L_logits.fill_(-20.0)
            self.dp_logits.fill_(-20.0)
            self.L_logits[L - int(self.L_candidates[0])] = 20.0
            self.dp_logits[dp - int(self.dp_candidates[0])] = 20.0

    def fix_L(self, L: int) -> None:
        if not int(self.L_candidates[0]) <= L <= int(self.L_candidates[-1]):
            raise ValueError("Fixed L is outside the candidate range.")
        self.fixed_L = L
        self.L_logits.requires_grad_(False)

    def logits_smoothness(self) -> torch.Tensor:
        dp_penalty = (self.dp_logits.diff().square().mean()
                      if self.dp_logits.numel() > 1 else self.dp_logits.new_zeros(()))
        if self.fixed_L is not None:
            return dp_penalty
        return self.L_logits.diff().square().mean() + dp_penalty

    def _resolve_delays(
        self, delay_len: int | None, dp: int | None
    ) -> tuple[int, int]:
        if delay_len is None or dp is None:
            selected_L, selected_dp = self.selected_delays()
        L = selected_L if delay_len is None else int(delay_len)
        pluck_delay = selected_dp if dp is None else int(dp)
        if not (
            int(self.L_candidates[0]) <= L <= int(self.L_candidates[-1])
            and int(self.dp_candidates[0]) <= pluck_delay <= int(self.dp_candidates[-1])
            and pluck_delay < L
        ):
            raise ValueError("Require candidate delays with 1 <= dp < L.")
        return L, pluck_delay

    @staticmethod
    def _validate_unit_impulse(unit_impulse: torch.Tensor) -> torch.Tensor:
        source = unit_impulse.flatten()
        if source.numel() != 1 or source.item() != 1.0:
            raise ValueError("The pluck model requires a one-sample unit impulse.")
        return source

    @classmethod
    def _filtered_impulse(
        cls, unit_impulse: torch.Tensor, n_samples: int, dp: int
    ) -> torch.Tensor:
        source = cls._validate_unit_impulse(unit_impulse)
        excitation = source.new_zeros(n_samples)
        copied = min(source.numel(), n_samples)
        excitation[:copied] = source[:copied]
        if dp < n_samples:
            delayed = min(copied, n_samples - dp)
            excitation[dp:dp + delayed] -= source[:delayed]
        return excitation

    def forward(
        self,
        unit_impulse: torch.Tensor,
        delay_len: int | None = None,
        dp: int | None = None,
    ) -> torch.Tensor:
        L, pluck_delay = self._resolve_delays(delay_len, dp)
        return self.spectral_response(unit_impulse, L, pluck_delay)

    def spectral_response(
        self,
        unit_impulse: torch.Tensor,
        delay_len: int | torch.Tensor,
        dp: int | torch.Tensor,
        delay_gain: float | torch.Tensor | None = None,
        allpass: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Circular transfer function, including the pluck-position comb."""
        self._validate_unit_impulse(unit_impulse)
        K = self.scaled_gain() if delay_gain is None else delay_gain
        a = self.scaled_allplus() if allpass is None else allpass
        z = self.z
        numerator = a / 2 + (a + 1) / 2 * z.pow(-1) + 0.5 * z.pow(-2)
        denominator = 1 + a * z.pow(-1) - K * numerator * z.pow(-delay_len)
        return -K * (1 - z.pow(-dp)) * numerator / denominator

    def time_domain_synth(
        self,
        n_samples: int,
        unit_impulse: torch.Tensor,
        delay_len: int | None = None,
        dp: int | None = None,
    ) -> torch.Tensor:
        L, pluck_delay = self._resolve_delays(delay_len, dp)
        K = self.scaled_gain()
        excitation = -K * self._filtered_impulse(
            unit_impulse, n_samples, pluck_delay
        )
        a = torch.as_tensor(
            self.scaled_allplus(),
            dtype=excitation.dtype,
            device=excitation.device,
        )
        numerator = torch.stack((a / 2, (a + 1) / 2, a * 0 + 0.5))
        b = excitation.new_zeros(L + 3)
        b[:3] = numerator
        denominator = excitation.new_zeros(L + 3)
        denominator[0] = 1
        denominator[1] = a
        denominator[L:L + 3] -= K * numerator
        return torchaudio.functional.lfilter(
            excitation, denominator, b, clamp=False
        )


class KarplusStrongPluckRelaxation(ContinuousDelayPairMixin, KarplusStrongPluck):
    """Continuous L and dp; causal synthesis floors both delays."""

    def __init__(self, *args, delay_len_init: float | None = None,
                 position_init: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_continuous_pair(delay_len_init, position_init, "dp")

    def scaled_dp(self) -> int:
        return self.selected_delays()[1]
