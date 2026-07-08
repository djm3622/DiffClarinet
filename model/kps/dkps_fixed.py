import torch
import torchaudio
from torch import nn

class KarplusStrongFixed(nn.Module):

    def __init__(self, delay_len, n_fft=2048, rescale=False, all_plus=False, a=0.1):
        super().__init__()
        self.delay_gain = nn.Parameter(torch.tensor(0.0))
        self.delay_len = delay_len
        self.n_fft = n_fft
        self.all_plus = all_plus
        self.rescale = rescale

        # for frequency sampling
        self.z = torch.exp(1j * torch.linspace(0, torch.pi, n_fft // 2 + 1))  # vectory of possible frequencies        

        self.a = a

    def scaled_gain(self):
        if self.rescale:
            return torch.sigmoid(self.delay_gain) * 0.1 + 0.9 # for to be positive, then scale. this init value is 0.95
        return torch.sigmoid(self.delay_gain)
    
    # forward pass: synthesis in the frequency domain
    def forward(self, noise):
        z = self.z
        exc = torch.zeros(self.n_fft) 
        exc[:self.delay_len] = noise
        exec_fft = torch.fft.rfft(exc)
        
        delay_gain = self.scaled_gain()

        # transfer function implementation
        if not self.all_plus:
            numer = 1 + z**-1
            denom = 2 - delay_gain * (z**(-self.delay_len)) - delay_gain * (z**(-1 * (self.delay_len + 1)))
        else:
            numer = 1/2 * z**(-2) + (self.a+1)/2 * z**(-1) + self.a/2
            denom = -(delay_gain/2) * z**(-(self.delay_len+2)) - (delay_gain*(self.a+1))/2 * z**(-(self.delay_len+1)) - delay_gain*self.a/2 * z**(-self.delay_len) + self.a*z**(-1) + 1
        
        # filter excitation in frequency domain
        # apply filter to the input
        return exec_fft * numer / denom # circular convolution

    def time_domain_synth(self, n_samples, noise):
        delay_gain = self.scaled_gain()

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