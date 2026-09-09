import torch
import torchaudio
from torch import nn

class KarplusStrongFixed(nn.Module):

    def __init__(
        self, delay_len, n_fft=2048, rescale=False, all_plus=True,
        all_plus_learnable=True, delay_gain_learnable=False, delay_len_learnable=False,
        delay_gain=0.99991, a=0.1, random_init=True,
        delay_len_min=40, delay_len_max=200,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.all_plus = all_plus
        self.rescale = rescale

        self.all_plus_learnable = all_plus_learnable
        self.delay_gain_learnable = delay_gain_learnable
        self.delay_len_learnable = delay_len_learnable
        # for frequency sampling
        self.z = torch.exp(1j * torch.linspace(0, torch.pi, n_fft // 2 + 1))  # vectory of possible frequencies

        if self.delay_gain_learnable:
            if random_init:
                gain_tensor = torch.rand(
                    (), dtype=torch.get_default_dtype()
                ).clamp_(1e-6, 1.0 - 1e-6)
            else:
                gain_tensor = torch.tensor(
                    delay_gain, dtype=torch.get_default_dtype()
                )
            self.delay_gain = nn.Parameter(torch.logit(gain_tensor))
        else:
            self.delay_gain = delay_gain

        if self.all_plus_learnable:
            if random_init:
                a_tensor = torch.rand(
                    (), dtype=torch.get_default_dtype()
                ).clamp_(1e-6, 1.0 - 1e-6)
            else:
                a_tensor = torch.tensor(a, dtype=torch.get_default_dtype())
            self.a = nn.Parameter(torch.logit(a_tensor))
        else:
            self.a = a

        if delay_len_learnable:
            if delay_len_min >= delay_len_max:
                raise ValueError(
                    "delay_len_min must be less than delay_len_max when "
                    "delay_len is learnable."
                )
            self.delay_len_min = delay_len_min
            self.delay_len_max = delay_len_max

            if random_init:
                initial_delay_len = torch.randint(
                    delay_len_min,
                    delay_len_max + 1,
                    (),
                    dtype=torch.int64,
                ).to(torch.get_default_dtype())
            else:
                if not delay_len_min <= delay_len <= delay_len_max:
                    raise ValueError(
                        f"delay_len={delay_len} is outside the candidate "
                        f"range [{delay_len_min}, {delay_len_max}]."
                    )
                initial_delay_len = torch.tensor(
                    delay_len,
                    dtype=torch.get_default_dtype(),
                )

            # Map integer bin centers to a finite unconstrained parameter.
            # The half-bin extension lets the endpoint delays remain reachable
            # without initializing a saturated logit.
            delay_span = delay_len_max - delay_len_min + 1
            initial_fraction = (
                initial_delay_len - delay_len_min + 0.5
            ) / delay_span
            self.delay_len_raw = nn.Parameter(torch.logit(initial_fraction))
        else:
            self.delay_len = delay_len

    def scaled_gain(self):
        if self.delay_gain_learnable:
            if self.rescale:
                return torch.sigmoid(self.delay_gain) * 0.1 + 0.9 # for to be positive, then scale. this init value is 0.95
            return torch.sigmoid(self.delay_gain)
        return self.delay_gain

    def scaled_allplus(self):
        if self.all_plus_learnable:
            if self.rescale:
                return torch.sigmoid(self.a) * 0.1 + 0.9 # for to be positive, then scale. this init value is 0.95
            return torch.sigmoid(self.a)
        return self.a

    def continuous_delay_len(self):
        if self.delay_len_learnable:
            delay_span = self.delay_len_max - self.delay_len_min + 1
            return (
                self.delay_len_min
                - 0.5
                + delay_span * torch.sigmoid(self.delay_len_raw)
            )
        return self.delay_len

    def straight_through_delay_len(self):
        delay_len_soft = self.continuous_delay_len()
        delay_len_hard = torch.round(delay_len_soft).clamp(
            self.delay_len_min,
            self.delay_len_max,
        )
        return delay_len_soft + (delay_len_hard - delay_len_soft).detach()

    def scaled_delay_len(self):
        if self.delay_len_learnable:
            delay_len = torch.round(self.continuous_delay_len()).clamp(
                self.delay_len_min,
                self.delay_len_max,
            )
            return int(delay_len.detach().item())
        return self.delay_len
    
    # forward pass: synthesis in the frequency domain
    def forward(self, noise):
        noise = noise.flatten()
        z = self.z

        if self.delay_len_learnable:
            delay_len = self.straight_through_delay_len()
            hard_delay_len = int(delay_len.detach().item())
            exc = noise.new_zeros(self.n_fft)
            copied_samples = min(
                hard_delay_len,
                self.n_fft,
                noise.numel(),
            )
            exc[:copied_samples] = noise[:copied_samples]

        else:
            delay_len = self.delay_len
            exc = noise.new_zeros(self.n_fft)
            copied_samples = min(delay_len, self.n_fft, noise.numel())
            exc[:copied_samples] = noise[:copied_samples]

        exc_fft = torch.fft.rfft(exc)
        
        delay_gain = self.scaled_gain()
        a = self.scaled_allplus()

        # transfer function implementation
        if not self.all_plus:
            numer = 1 + z**-1
            denom = 2 - delay_gain * (z**(-delay_len)) - delay_gain * (z**(-1 * (delay_len + 1)))
        else:
            numer = 1/2 * z**(-2) + (a+1)/2 * z**(-1) + a/2
            denom = -(delay_gain/2) * z**(-(delay_len+2)) - (delay_gain*(a+1))/2 * z**(-(delay_len+1)) - delay_gain*a/2 * z**(-delay_len) + a*z**(-1) + 1
        
        # filter excitation in frequency domain
        # apply filter to the input
        return exc_fft * numer / denom # circular convolution

    def time_domain_synth(self, n_samples, noise):
        noise = noise.flatten()
        delay_len = self.scaled_delay_len()
        delay_gain = self.scaled_gain()
        a = self.scaled_allplus()

        exc = noise.new_zeros(self.n_fft)
        copied_samples = min(delay_len, self.n_fft, noise.numel())
        exc[:copied_samples] = noise[:copied_samples]

        if not self.all_plus:
            a_coeffs = torch.zeros(delay_len + 2) # poles of delay line
            a_coeffs[0] = 2
            a_coeffs[delay_len] = -delay_gain
            a_coeffs[delay_len + 1] = -delay_gain

            b_coeffs = torch.zeros(delay_len + 2) # zeros of delay line
            b_coeffs[0] = 1
            b_coeffs[1] = 1
        else:
            a_coeffs = torch.zeros(delay_len + 3) # poles of delay line
            a_coeffs[0] = 1
            a_coeffs[1] = a
            a_coeffs[delay_len] = - delay_gain*a/2
            a_coeffs[delay_len + 1] = - (delay_gain*(a+1))/2
            a_coeffs[delay_len + 2] = - (delay_gain/2)

            b_coeffs = torch.zeros(delay_len + 3) # zeros of delay line
            b_coeffs[0] = a / 2
            b_coeffs[1] = (a + 1) / 2
            b_coeffs[2] = 1/2

        # pad or truncate exc to n_samples
        if exc.shape[0] < n_samples:
            audio = torch.cat([exc, torch.zeros(n_samples - exc.shape[0])])
        else:
            audio = exc[:n_samples]

        audio = torchaudio.functional.lfilter(audio, a_coeffs, b_coeffs, clamp=False)
        return audio
