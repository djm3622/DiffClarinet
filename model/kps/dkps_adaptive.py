import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch import fft

class KarplusStrongAdaptive(nn.Module):

    def __init__(self, delay_len, n_fft=2048, rescale=False, all_plus=False, a=0.1, auraloss_package=True):
        super().__init__()
        self.delay_len = delay_len
        self.n_fft = n_fft
        self.rescale = rescale
        self.all_plus = all_plus
        self.a = a
        self.auraloss_package = auraloss_package

        # for frequency sampling
        omega = torch.linspace(0.0, torch.pi, n_fft // 2 + 1)
        self.register_buffer("z", torch.exp(1j * omega))  

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=33, stride=4, padding=16),
            nn.GELU(),

            nn.Conv1d(4, 8, kernel_size=15, stride=4, padding=7),
            nn.GELU(),

            nn.Conv1d(8, 16, kernel_size=9, stride=4, padding=4),
            nn.GELU(),

            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(8), # number of output points

            nn.Flatten(),
            nn.Linear(32 * 8, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

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
        y = self.encoder(x.unsqueeze(1))

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

        exc = torch.zeros(self.n_fft) 
        exc[:self.delay_len] = noise

        if not self.all_plus:
            a_coeffs = torch.zeros(self.delay_len + 2) # poles of delay line
            a_coeffs[0] = 2
            a_coeffs[self.delay_len] = -delay_gain
            a_coeffs[self.delay_len + 1] = -delay_gain

            b_coeffs = torch.zeros(self.delay_len + 2) # zeros of delay line
            b_coeffs[0] = 1
            b_coeffs[1] = 1
        else:
            a_coeffs = torch.zeros(self.delay_len + 3) # poles of delay line
            a_coeffs[0] = 1
            a_coeffs[1] = self.a
            a_coeffs[self.delay_len] = - delay_gain*self.a/2
            a_coeffs[self.delay_len + 1] = - (delay_gain*(self.a+1))/2
            a_coeffs[self.delay_len + 2] = - (delay_gain/2)

            b_coeffs = torch.zeros(self.delay_len + 3) # zeros of delay line
            b_coeffs[0] = self.a / 2
            b_coeffs[1] = (self.a + 1) / 2
            b_coeffs[2] = 1/2

        # pad or truncate exc to n_samples
        if exc.shape[0] < n_samples:
            audio = torch.cat([exc, torch.zeros(n_samples - exc.shape[0])])
        else:
            audio = exc[:n_samples]

        audio = torchaudio.functional.lfilter(audio, a_coeffs, b_coeffs, clamp=False)
        return audio
