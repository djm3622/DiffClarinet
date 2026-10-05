"""Causal Karplus--Strong model with a discrete pluck-position delay."""

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from .dkps_fixed import KarplusStrongFixed


class KarplusStrongPluck(KarplusStrongFixed):
    """Learn integer loop delay L and pluck-filter delay dp from a unit impulse.

    The MATLAB generator uses input -K * (delta[n] - delta[n-dp]) and
    feedback -K * y[n-L]. Candidate pairs with dp >= L are excluded.
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
        dp_penalty = self.dp_logits.diff().square().mean()
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
        denominator = 1 + a * z.pow(-1) + K * numerator * z.pow(-delay_len)
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
        denominator[L:L + 3] += K * numerator
        return torchaudio.functional.lfilter(
            excitation, denominator, b, clamp=False
        )


class KarplusStrongPluckRelaxation(KarplusStrongPluck):
    """Continuous-L surrogate with a categorical integer pluck delay."""

    def __init__(self, *args, delay_len_init: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        lower = float(self.L_candidates[0])
        upper = float(self.L_candidates[-1])
        if not lower <= delay_len_init <= upper:
            raise ValueError("Initial continuous L must be in the candidate range.")
        self.L_logits.requires_grad_(False)
        self.delay_len_parameter = nn.Parameter(
            torch.tensor(float(delay_len_init))
        )

    def scaled_delay_len(self) -> torch.Tensor:
        return self.delay_len_parameter.clamp(
            float(self.L_candidates[0]), float(self.L_candidates[-1])
        )

    def dp_distribution(self) -> torch.distributions.Categorical:
        integer_L = int(self.scaled_delay_len().detach().floor().item())
        logits = self.dp_logits.masked_fill(
            self.dp_candidates >= integer_L, -torch.inf
        )
        return torch.distributions.Categorical(logits=logits)

    def scaled_dp(self) -> int:
        return int(self.dp_candidates[torch.argmax(
            self.dp_distribution().logits
        )].item())

    def _resolve_delays(
        self, delay_len: int | None, dp: int | None
    ) -> tuple[int, int]:
        L = (
            int(self.scaled_delay_len().detach().floor().item())
            if delay_len is None else int(delay_len)
        )
        pluck_delay = self.scaled_dp() if dp is None else int(dp)
        if not (
            int(self.L_candidates[0]) <= L <= int(self.L_candidates[-1])
            and int(self.dp_candidates[0]) <= pluck_delay <= int(self.dp_candidates[-1])
            and pluck_delay < L
        ):
            raise ValueError("Require candidate delays with 1 <= dp < L.")
        return L, pluck_delay
