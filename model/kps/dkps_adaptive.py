import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch import fft

from .encoders import base_cnn, dilated_cnn, res_cnn, spectral_2d_cnn

class KarplusStrongAdaptive(nn.Module):

    def __init__(
        self, delay_len, n_fft=2048, rescale=False, all_plus=False, a=0.1, 
        auraloss_package=True, all_pass_learnable=True, delay_len_learnable=True
    ):
        super().__init__()
        self.delay_len = delay_len
        self.n_fft = n_fft
        self.rescale = rescale
        self.all_plus = all_plus
        self.a = a
        self.auraloss_package = auraloss_package
        self.all_pass_learnable = all_pass_learnable
        self.delay_len_learnable = delay_len_learnable

        # for frequency sampling
        omega = torch.linspace(0.0, torch.pi, n_fft // 2 + 1)
        self.register_buffer("z", torch.exp(1j * omega)) 
        
        encoder_type = "dilated" # "base", "res", "dilated", "spectral"
        encoder_rank = "l1" # "s1", m1", "l1"

        if encoder_type == "base":
            if delay_len_learnable:
                self.delay_encoder = base_cnn.BaseCNN(encoder_rank)
            else:
                self.delay_encoder = None
            if all_pass_learnable:
                self.all_pass_encoder = base_cnn.BaseCNN(encoder_rank)
            else:
                self.all_pass_encoder = None
        elif encoder_type == "res":
            if delay_len_learnable:
                self.delay_encoder = res_cnn.ResCNN(encoder_rank)
            else:
                self.delay_encoder = None
            if all_pass_learnable:
                self.all_pass_encoder = res_cnn.ResCNN(encoder_rank)
            else:
                self.all_pass_encoder = None
        elif encoder_type == "dilated":
            if delay_len_learnable:
                self.delay_encoder = dilated_cnn.DilatedCNN(encoder_rank)
            else:
                self.delay_encoder = None
            if all_pass_learnable:
                self.all_pass_encoder = dilated_cnn.DilatedCNN(encoder_rank)
            else:
                self.all_pass_encoder = None
        elif encoder_type == "spectral":
            pass

        self.a = a

    def scaled_gain(self, x):

        # rms normalization
        # instead of peak amplitude, use the root mean square to give a slightly more stable estimate
        initial_rms = torch.sqrt(
            torch.mean(
                x[:, :self.delay_len] ** 2,
                dim=-1,
                keepdim=True
            )
            + 1e-8
        )

        x = x / initial_rms
        y = self.delay_encoder(x.unsqueeze(1))

        if self.rescale:
            return torch.sigmoid(y) * 0.1 + 0.9 # for to be positive, then scale. this init value is 0.95
        return torch.sigmoid(y)


    def scaled_allplus(self, x):
        # rms normalization
        # instead of peak amplitude, use the root mean square to give a slightly more stable estimate
        initial_rms = torch.sqrt(
            torch.mean(
                x[:, :self.delay_len] ** 2,
                dim=-1,
                keepdim=True
            )
            + 1e-8
        )
        
        x = x / initial_rms
        y = self.all_pass_encoder(x.unsqueeze(1))
        
        if self.rescale:
            return torch.sigmoid(y) * 0.1 + 0.9 # for to be positive, then scale. this init value is 0.95
        return torch.sigmoid(y)
    
    
    # forward pass: synthesis in the frequency domain
    def forward(self, x, noise):
        z = self.z
        exc = F.pad(noise, (0, self.n_fft - self.delay_len))
        exc_fft = fft.rfft(exc, n=self.n_fft, dim=-1)
        
        delay_gain = self.scaled_gain(x)
        
        # transfer function implementation
        if not self.all_plus:
            numer = 1 + z**-1
            denom = 2 - delay_gain * (z**(-self.delay_len)) - delay_gain * (z**(-1 * (self.delay_len + 1)))
        else:
            numer = 1/2 * z**(-2) + (self.a+1)/2 * z**(-1) + self.a/2
            denom = -(delay_gain/2) * z**(-(self.delay_len+2)) - (delay_gain*(self.a+1))/2 * z**(-(self.delay_len+1)) - delay_gain*self.a/2 * z**(-self.delay_len) + self.a*z**(-1) + 1
        
        # filter excitation in frequency domain
        # apply filter to the input
        y = exc_fft * numer / denom

        if self.auraloss_package:
            return torch.fft.irfft(y, n=self.n_fft, dim=-1)
        else:
            return y
    
    # also provide method for time domain synthesis
    def time_domain_synth(self, x, n_samples, noise):
        delay_gain = self.scaled_gain(x)

        if noise.ndim == 1:
            noise = noise.unsqueeze(0)
        if noise.shape[-1] != self.delay_len:
            raise ValueError(
                f"Expected excitation length {self.delay_len}, "
                f"but received {noise.shape[-1]}."
            )

        if n_samples >= self.delay_len:
            exc = F.pad(noise, (0, n_samples - self.delay_len))
        else:
            exc = noise[..., :n_samples]

        batch_size = exc.shape[0]
        delay_gain = delay_gain.reshape(-1)
        if delay_gain.numel() != batch_size:
            raise ValueError(
                "The number of predicted gains must match the excitation batch size."
            )

        if not self.all_plus:
            coefficient_count = self.delay_len + 2
            a_coeffs = noise.new_zeros(batch_size, coefficient_count)
            a_coeffs[:, 0] = 2
            a_coeffs[:, self.delay_len] = -delay_gain
            a_coeffs[:, self.delay_len + 1] = -delay_gain

            b_coeffs = noise.new_zeros(batch_size, coefficient_count)
            b_coeffs[:, 0] = 1
            b_coeffs[:, 1] = 1
        else:
            coefficient_count = self.delay_len + 3
            a_coeffs = noise.new_zeros(batch_size, coefficient_count)
            a_coeffs[:, 0] = 1
            a_coeffs[:, 1] = self.a
            a_coeffs[:, self.delay_len] = -delay_gain * self.a / 2
            a_coeffs[:, self.delay_len + 1] = -delay_gain * (self.a + 1) / 2
            a_coeffs[:, self.delay_len + 2] = -delay_gain / 2

            b_coeffs = noise.new_zeros(batch_size, coefficient_count)
            b_coeffs[:, 0] = self.a / 2
            b_coeffs[:, 1] = (self.a + 1) / 2
            b_coeffs[:, 2] = 1 / 2

        return torchaudio.functional.lfilter(
            exc,
            a_coeffs,
            b_coeffs,
            clamp=False,
            batching=True,
        )
