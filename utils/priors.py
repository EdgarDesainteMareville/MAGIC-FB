import deepinv as dinv
import pywt
import torch


class WaveletPriorCustom(dinv.optim.WaveletPrior):
    '''
    Overwriting the prox of DeepInv's WaveletPrior to make sure it works as intended.
    '''

    def __init__(self, level=3, wv="db8", p=1, device="cpu"):
        super().__init__(level=level, wv=wv, p=p, device=device)

    def prox(self, x, gamma=1):
        prior_l1 = dinv.optim.L1Prior()

        x_np = x.cpu().numpy()
        coeffs = pywt.wavedec2(
            x_np, wavelet=self.wv, level=self.level, mode="periodization"
        )

        coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients unchanged
        for j in range(1, len(coeffs)):
            coeffs_thresh.append(
                tuple(
                    prior_l1.prox(x=torch.from_numpy(coeffs[j][c]), gamma=gamma)
                    .cpu()
                    .numpy()
                    for c in range(3)
                )
            )

        x_denoised = pywt.waverec2(coeffs_thresh, self.wv, mode="periodization")
        return torch.from_numpy(x_denoised).to(x.device, dtype=x.dtype)