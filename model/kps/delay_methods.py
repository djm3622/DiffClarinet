from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .dkps_fixed import KarplusStrongFixed


Objective = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class _CandidateDelayKarplusStrong(KarplusStrongFixed):
    """Shared candidate-grid behavior for discrete delay estimators."""

    def __init__(
        self,
        delay_len_min: int,
        delay_len_max: int,
        **kwargs,
    ) -> None:
        if delay_len_min >= delay_len_max:
            raise ValueError("delay_len_min must be less than delay_len_max.")
        super().__init__(delay_len=delay_len_min, **kwargs)
        self.delay_len_min = delay_len_min
        self.delay_len_max = delay_len_max
        self.register_buffer(
            "L_candidates",
            torch.arange(delay_len_min, delay_len_max + 1),
        )

    def _validate_candidate(self, delay_len: int) -> int:
        delay_len = int(delay_len)
        if not self.delay_len_min <= delay_len <= self.delay_len_max:
            raise ValueError(
                f"delay_len={delay_len} is outside the candidate range "
                f"[{self.delay_len_min}, {self.delay_len_max}]."
            )
        return delay_len

    def _resolve_delay_len(self, delay_len):
        if delay_len is None:
            return self.scaled_delay_len()
        return self._validate_candidate(delay_len)


class _CategoricalDelayKarplusStrong(_CandidateDelayKarplusStrong):
    """Shared categorical parameterization for learned delay estimators."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.L_logits = nn.Parameter(
            torch.zeros(
                self.L_candidates.numel(),
                dtype=torch.get_default_dtype(),
            )
        )

    def scaled_delay_len(self) -> int:
        delay_index = torch.argmax(self.L_logits)
        return int(self.L_candidates[delay_index].item())

    def delay_len_probabilities(self) -> torch.Tensor:
        return torch.softmax(self.L_logits, dim=0)

    def delay_len_logit_smoothness(self) -> torch.Tensor:
        adjacent_differences = self.L_logits[1:] - self.L_logits[:-1]
        return torch.mean(adjacent_differences.square())


class KarplusStrongReinforce(_CategoricalDelayKarplusStrong):
    """Categorical delay selection trained with a score-function estimator."""

    def delay_len_distribution(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.L_logits)

    def sample_delay_lengths(
        self,
        num_samples: int,
    ) -> tuple[list[int], torch.Tensor]:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")
        distribution = self.delay_len_distribution()
        delay_indices = distribution.sample((num_samples,))
        delay_lengths = [
            int(delay_len.item())
            for delay_len in self.L_candidates[delay_indices]
        ]
        return delay_lengths, distribution.log_prob(delay_indices)


class KarplusStrongGumbelSoftmax(_CategoricalDelayKarplusStrong):
    """Hard Gumbel--Softmax delay selection with a soft backward pass."""

    def __init__(
        self,
        *args,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.set_temperature(temperature)
        self.last_sampled_delay_len: int | None = None

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)

    def sample_relaxed_delay_len(self) -> torch.Tensor:
        selection = F.gumbel_softmax(
            self.L_logits,
            tau=self.temperature,
            hard=True,
            dim=0,
        )
        candidates = self.L_candidates.to(selection.dtype)
        return torch.sum(selection * candidates)

    def _relaxed_circular_forward(
        self,
        noise: torch.Tensor,
        delay_len: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one hard delay while retaining its straight-through gradient."""
        noise = noise.flatten()
        hard_delay_len = int(delay_len.detach().item())
        excitation = noise.new_zeros(self.n_fft)
        copied_samples = min(
            hard_delay_len,
            self.n_fft,
            noise.numel(),
        )
        excitation[:copied_samples] = noise[:copied_samples]
        excitation_fft = torch.fft.rfft(excitation)

        z = self.z
        delay_gain = self.scaled_gain()
        allpass = self.scaled_allplus()
        if not self.all_plus:
            numerator = 1 + z.pow(-1)
            denominator = (
                2
                - delay_gain * z.pow(-delay_len)
                - delay_gain * z.pow(-(delay_len + 1))
            )
        else:
            numerator = (
                0.5 * z.pow(-2)
                + (allpass + 1) / 2 * z.pow(-1)
                + allpass / 2
            )
            denominator = (
                -delay_gain / 2 * z.pow(-(delay_len + 2))
                - delay_gain * (allpass + 1) / 2
                * z.pow(-(delay_len + 1))
                - delay_gain * allpass / 2 * z.pow(-delay_len)
                + allpass * z.pow(-1)
                + 1
            )
        return excitation_fft * numerator / denominator

    def forward(
        self,
        noise: torch.Tensor,
        delay_len: int | None = None,
    ) -> torch.Tensor:
        if delay_len is not None:
            return super().forward(noise, delay_len=delay_len)

        if not self.training:
            selected_delay_len = self.scaled_delay_len()
            self.last_sampled_delay_len = selected_delay_len
            return super().forward(noise, delay_len=selected_delay_len)

        relaxed_delay_len = self.sample_relaxed_delay_len()
        self.last_sampled_delay_len = int(relaxed_delay_len.detach().item())
        return self._relaxed_circular_forward(noise, relaxed_delay_len)


class KarplusStrongExhaustive(_CandidateDelayKarplusStrong):
    """Non-gradient baseline that selects the best candidate by enumeration."""

    def select_delay_len(
        self,
        noise: torch.Tensor,
        target: torch.Tensor,
        objective: Objective,
    ) -> tuple[int, torch.Tensor]:
        with torch.no_grad():
            losses = torch.stack([
                objective(
                    self.forward(noise, delay_len=int(candidate)),
                    target,
                )
                for candidate in self.L_candidates
            ])
        best_index = int(torch.argmin(losses).item())
        self.delay_len = int(self.L_candidates[best_index].item())
        return self.delay_len, losses


class KarplusStrongPitch(_CandidateDelayKarplusStrong):
    """Non-gradient delay baseline based on waveform autocorrelation."""

    def estimate_delay_len(
        self,
        waveform: torch.Tensor,
        sample_rate: float,
    ) -> tuple[int, float]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        waveform = waveform.flatten()
        if waveform.numel() <= self.delay_len_max:
            raise ValueError(
                "waveform must be longer than delay_len_max for pitch "
                "estimation."
            )

        centered = waveform - waveform.mean()
        correlations = []
        for lag in range(self.delay_len_min, self.delay_len_max + 1):
            leading = centered[:-lag]
            lagging = centered[lag:]
            normalization = torch.sqrt(
                leading.square().sum() * lagging.square().sum()
            ).clamp_min(1e-12)
            correlations.append((leading * lagging).sum() / normalization)

        best_index = int(torch.argmax(torch.stack(correlations)).item())
        estimated_period = self.delay_len_min + best_index
        estimated_frequency = float(sample_rate) / estimated_period

        # The loop's two-point averager contributes 0.5 samples. The first-
        # order all-pass contributes its low-frequency group delay.
        loop_filter_delay = 0.5
        if self.all_plus:
            allpass_value = self.scaled_allplus()
            if isinstance(allpass_value, torch.Tensor):
                allpass = float(allpass_value.detach().item())
            else:
                allpass = float(allpass_value)
            loop_filter_delay += (1.0 - allpass) / (1.0 + allpass)
        estimated_delay_len = round(estimated_period - loop_filter_delay)
        self.delay_len = int(max(
            self.delay_len_min,
            min(self.delay_len_max, estimated_delay_len),
        ))
        return self.delay_len, estimated_frequency
