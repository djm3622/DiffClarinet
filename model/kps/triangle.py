"""Unit-impulse triangle excitation and positive-feedback KS resonator."""

import torch
import torchaudio
from torch import nn

from .dkps_fixed import KarplusStrongFixed


def triangle(n_samples: int, L: int | torch.Tensor,
             A: int | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Unit-height, zero-based triangle supported on [0, L]."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    n = torch.arange(n_samples, device=reference.device, dtype=reference.dtype)
    L = torch.as_tensor(L, device=n.device, dtype=n.dtype)
    A = torch.as_tensor(A, device=n.device, dtype=n.dtype)
    if not (bool((L >= 2).item()) and bool((A >= 1).item())
            and bool((A < L).item())):
        raise ValueError("Require L >= 2 and 1 <= A < L.")
    return torch.where(n <= A, n / A, torch.clamp((L - n) / (L - A), min=0))


class KarplusStrongTriangle(KarplusStrongFixed):
    """Categorical integer (L, A), with continuous loop gain K and allpass a."""

    def __init__(self, delay_len_min: int, delay_len_max: int,
                 A_min: int, A_max: int, **kwargs) -> None:
        if delay_len_min < 2 or delay_len_max < delay_len_min:
            raise ValueError("Invalid L candidate bounds.")
        if A_min < 1 or A_max < A_min or A_min >= delay_len_max:
            raise ValueError("Invalid A candidate bounds.")
        super().__init__(delay_len=delay_len_min, **kwargs)
        if not self.all_plus or self.n_fft <= delay_len_max:
            raise ValueError("Require allpass loop and n_fft > maximum L.")
        self.register_buffer("L_candidates", torch.arange(delay_len_min, delay_len_max + 1))
        self.register_buffer("A_candidates", torch.arange(A_min, A_max + 1))
        self.L_logits = nn.Parameter(torch.zeros(self.L_candidates.numel()))
        self.A_logits = nn.Parameter(torch.zeros(self.A_candidates.numel()))
        if self.valid_pair_count() == 0:
            raise ValueError("Candidate grid contains no valid (L, A) pair.")

    def _unit(self, unit_impulse: torch.Tensor) -> torch.Tensor:
        source = unit_impulse.flatten()
        if source.numel() != 1 or not bool(torch.isfinite(source).all()) or source.item() != 1:
            raise ValueError("Expected a one-sample unit impulse.")
        if source.device != self.L_logits.device or source.dtype != self.L_logits.dtype:
            raise ValueError("Unit impulse must match model device and dtype.")
        return source

    def _joint_logits(self) -> torch.Tensor:
        valid = self.A_candidates[None, :] < self.L_candidates[:, None]
        return (self.L_logits[:, None] + self.A_logits[None, :]).masked_fill(
            ~valid, -torch.inf).flatten()

    def pair_distribution(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self._joint_logits())

    def valid_pair_count(self) -> int:
        return int((self.A_candidates[None, :] < self.L_candidates[:, None]).sum())

    def all_pairs(self) -> list[tuple[int, int]]:
        return [(int(L), int(A)) for L in self.L_candidates.tolist()
                for A in self.A_candidates.tolist() if A < L]

    def _decode(self, index: int) -> tuple[int, int]:
        width = self.A_candidates.numel()
        return (int(self.L_candidates[index // width]),
                int(self.A_candidates[index % width]))

    def selected_delays(self) -> tuple[int, int]:
        return self._decode(int(self._joint_logits().argmax()))

    def sample_pairs(self, count: int) -> tuple[list[tuple[int, int]], torch.Tensor]:
        distribution = self.pair_distribution()
        indices = distribution.sample((count,))
        return [self._decode(int(i)) for i in indices], distribution.log_prob(indices)

    def select_pair(self, pair: tuple[int, int]) -> None:
        if pair not in self.all_pairs():
            raise ValueError("Pair is outside the valid candidate grid.")
        L, A = pair
        with torch.no_grad():
            self.L_logits.fill_(-20)
            self.A_logits.fill_(-20)
            self.L_logits[L - int(self.L_candidates[0])] = 20
            self.A_logits[A - int(self.A_candidates[0])] = 20

    def excitation(self, n_samples: int, unit_impulse: torch.Tensor,
                   L: int | torch.Tensor, A: int | torch.Tensor,
                   K: torch.Tensor | float | None = None) -> torch.Tensor:
        source = self._unit(unit_impulse)
        gain = self.scaled_gain() if K is None else K
        return -gain * triangle(n_samples, L, A, source)

    def spectral_response(self, unit_impulse: torch.Tensor,
                          L: int | torch.Tensor, A: int | torch.Tensor,
                          delay_gain: torch.Tensor | float | None = None,
                          allpass: torch.Tensor | float | None = None) -> torch.Tensor:
        if float(torch.as_tensor(L).detach()) >= self.n_fft:
            raise ValueError("n_fft must exceed L.")
        K = self.scaled_gain() if delay_gain is None else delay_gain
        a = self.scaled_allplus() if allpass is None else allpass
        excitation = self.excitation(self.n_fft, unit_impulse, L, A, K)
        z = self.z
        numerator = a / 2 + (a + 1) / 2 * z.pow(-1) + 0.5 * z.pow(-2)
        denominator = 1 + a * z.pow(-1) - K * numerator * z.pow(-L)
        return torch.fft.rfft(excitation) * numerator / denominator

    def time_domain_synth(self, n_samples: int, unit_impulse: torch.Tensor,
                          L: int | None = None, A: int | None = None) -> torch.Tensor:
        selected = self.selected_delays()
        L = selected[0] if L is None else int(L)
        A = selected[1] if A is None else int(A)
        if not (int(self.L_candidates[0]) <= L <= int(self.L_candidates[-1])
                and int(self.A_candidates[0]) <= A <= int(self.A_candidates[-1])
                and A < L):
            raise ValueError("Expected a valid candidate pair.")
        K = self.scaled_gain()
        excitation = self.excitation(n_samples, unit_impulse, L, A, K)
        a = torch.as_tensor(self.scaled_allplus(), device=excitation.device,
                            dtype=excitation.dtype)
        numerator = torch.stack((a / 2, (a + 1) / 2, a * 0 + 0.5))
        b = excitation.new_zeros(L + 3)
        b[:3] = numerator
        denominator = excitation.new_zeros(L + 3)
        denominator[0] = 1
        denominator[1] = a
        denominator[L:L + 3] -= K * numerator
        output = torchaudio.functional.lfilter(excitation, denominator, b, clamp=False)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("Nonfinite causal triangle output.")
        return output

    def forward(self, unit_impulse: torch.Tensor,
                L: int | None = None, A: int | None = None) -> torch.Tensor:
        pair = self.selected_delays()
        return self.spectral_response(unit_impulse,
                                      pair[0] if L is None else L,
                                      pair[1] if A is None else A)


class KarplusStrongTriangleRelaxation(KarplusStrongTriangle):
    """Continuous spectral L, categorical valid A; causal output floors L."""

    def __init__(self, *args, delay_len_init: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not float(self.L_candidates[0]) <= delay_len_init <= float(self.L_candidates[-1]):
            raise ValueError("Continuous L initialization outside candidate bounds.")
        self.L_logits.requires_grad_(False)
        self.delay_len_parameter = nn.Parameter(torch.tensor(delay_len_init))
        if int(self.A_candidates[0]) >= int(delay_len_init):
            raise ValueError("Initial continuous L has no valid A candidate.")

    def continuous_L(self) -> torch.Tensor:
        return self.delay_len_parameter.clamp(float(self.L_candidates[0]),
                                               float(self.L_candidates[-1]))

    def A_distribution(self) -> torch.distributions.Categorical:
        bound = int(self.continuous_L().detach().floor())
        logits = self.A_logits.masked_fill(self.A_candidates >= bound, -torch.inf)
        return torch.distributions.Categorical(logits=logits)

    def selected_delays(self) -> tuple[int, int]:
        L = int(self.continuous_L().detach().floor())
        A = int(self.A_candidates[self.A_distribution().logits.argmax()])
        return L, A
