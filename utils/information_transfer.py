import lazylinop
import torch
import deepinv as dinv

from lazylinop.signal import dwt


class InformationTransferMatrix():
    """
    Information transfer operator using matrix representation of the DWT.
    """
    def __init__(self, wv="haar", mode="circular", device="cpu"):
        self.wv = wv
        self.mode = mode
        self.device = device
        self.dtype = torch.float32

        self.W_row = None
        self.W_col = None

    def create_W_matrices(self, signal_size):
        """
        Create the matrix representation of the DWT operator, from the LazyLinop operators. Only for 1D signals.
        """
        W_row_op = dwt(signal_size, wavelet=self.wv, mode=self.mode, level=1, backend='lazylinop')
        W_row = torch.from_numpy(W_row_op.toarray()).to(dtype=self.dtype, device=self.device)[:signal_size // 2, :] # only keeping the low-pass part
        W_col_op = dwt(signal_size, wavelet=self.wv, mode=self.mode, level=1, backend='lazylinop')
        W_col = torch.from_numpy(W_col_op.toarray()).to(dtype=self.dtype, device=self.device)[:signal_size // 2, :] # only keeping the low-pass part

        self.W_row = W_row
        self.W_col = W_col

    def get_W_matrices(self, signal_size):
        if self.W_row is None or self.W_col is None or self.W_row.shape[1] != signal_size or self.W_col.shape[0] != signal_size:
            self.create_W_matrices(signal_size)
        return self.W_row, self.W_col

    def downsample(self, x):
        signal_size = x.shape[-1]
        if self.W_row is None or self.W_col is None or self.W_row.shape[1] != signal_size or self.W_col.shape[0] != signal_size:
            self.create_W_matrices(signal_size)
        return (self.W_row @ x @ self.W_col.T) #[:, :, :signal_size // 2, :signal_size // 2]

    def upsample(self, y):
        # y is the downsampled signal, so its size is half the original signal size
        signal_size = 2 * y.shape[-1]
        if self.W_row is None or self.W_col is None or self.W_row.shape[0] != signal_size or self.W_col.shape[1] != signal_size:
            self.create_W_matrices(signal_size)
        return self.W_row.T @ y @ self.W_col


class WaveletInformationTransferMatrices():
    def __init__(self, wv='haar', mode='periodization', level=1, device='cpu', dtype=torch.float32):
        self.wv = wv
        self.mode = mode
        self.level = level
        self.device = device
        self.dtype = dtype

        self.W_row = None
        self.W_col = None

    def compute_wavelet_matrices(self, N, wavelet="haar", mode="periodization", level=1):
        """Return the low-pass and high-pass wavelet matrices for 1D DWT."""
        W = torch.from_numpy(
            lazylinop.signal.dwt(
                N=N, wavelet=wavelet, mode=mode, level=1, backend="lazylinop"
            ).toarray()
        ).float()
        low = W[: N // 2, :].to(self.device)
        high = W[N // 2 :, :].to(self.device)
        return low, high

    def compute_Pi_operators(self, N, levels, wavelet="haar", mode="periodization"):
        """
        Constructs the Pi operators (projections on wavelet subspaces) for the 2D DWT up to the specified number of levels.
        Parameters:
        - N : Size of the original signal (assumed square)
        - levels : Number of decomposition levels
        - wavelet : Wavelet type
        - mode : Signal extension mode
        Returns a dictionary of Pi operators:
        - Pi["A{l}_row"], Pi["A{l}_col"] : Approximation at level l
        - Pi["V{l}_row"], Pi["V{l}_col"] : Vertical details at level l
        - Pi["H{l}_row"], Pi["H{l}_col"] : Horizontal details at level l
        - Pi["D{l}_row"], Pi["D{l}_col"] : Diagonal details at level l
        """
        Pi = {}
        low_prev = torch.eye(N).to(self.device) # base : identité sur le signal complet

        for l in range(1, levels + 1):
            low, high = self.compute_wavelet_matrices(N // (2 ** (l - 1)), wavelet, mode)

            Pi[f"A{l}_row"] = low @ low_prev
            Pi[f"A{l}_col"] = low @ low_prev

            Pi[f"V{l}_row"] = low @ low_prev
            Pi[f"V{l}_col"] = high @ low_prev

            Pi[f"H{l}_row"] = high @ low_prev
            Pi[f"H{l}_col"] = low @ low_prev

            Pi[f"D{l}_row"] = high @ low_prev
            Pi[f"D{l}_col"] = high @ low_prev

            low_prev = low @ low_prev

        return Pi

    def dwt(self, x):
        """Applies the DWT to the input image x and returns the wavelet coefficients."""
        L = self.level
        Pi_ops = self.compute_Pi_operators(N=x.shape[-1], levels=L, wavelet=self.wv, mode=self.mode)

        coeffs = {}
        # Approximation at the coarsest scale
        coeffs[f"A{L}"] = Pi_ops[f"A{L}_row"] @ x @ Pi_ops[f"A{L}_col"].T

        # DDetails at each level
        for l in range(L, 0, -1):
            coeffs[f"V{l}"] = Pi_ops[f"V{l}_row"] @ x @ Pi_ops[f"V{l}_col"].T
            coeffs[f"H{l}"] = Pi_ops[f"H{l}_row"] @ x @ Pi_ops[f"H{l}_col"].T
            coeffs[f"D{l}"] = Pi_ops[f"D{l}_row"] @ x @ Pi_ops[f"D{l}_col"].T

        return coeffs

    def idwt(self, Pi_ops, coeffs):
        """Reconstruct the image from the wavelet coefficients."""
        # Max level
        L = max(int(k[1:]) for k in coeffs if k[0] in "AVHD")

        # Approximation at the coarsest scale
        x_rec = (
            Pi_ops[f"A{L}_row"].T @ coeffs[f"A{L}"] @ Pi_ops[f"A{L}_col"]
        )

        # Add details from each level after projecting them in the fine space
        for l in range(L, 0, -1):
            for t in ("V", "H", "D"):
                k = f"{t}{l}"
                if k in coeffs:
                    x_rec += (
                        Pi_ops[f"{t}{l}_row"].T @ coeffs[k] @ Pi_ops[f"{t}{l}_col"]
                    )
        return x_rec

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage
    x = dinv.utils.load_example("butterfly.png", grayscale=False)
    it_op = InformationTransferMatrix(wv="db8", mode="periodization", device="cpu")
    W_row, W_col = it_op.get_W_matrices(x.shape[-1])

    y = W_row @ x @ W_col.T
    print(f"Downsampled shape: {y.shape}")

    x_recon = it_op.upsample(y)
    print(f"Reconstructed shape: {x_recon.shape}")

    dinv.utils.plot([x, y, x_recon], titles=["x", "x_coarse in the coarse space", "x_coarse in the fine space"])

    wv_info_transfer = WaveletInformationTransferMatrices(wv="haar", mode="periodization", level=2, device="cpu")

    coeffs = wv_info_transfer.dwt(x)

    Pi_ops = wv_info_transfer.compute_Pi_operators(N=x.shape[-1], levels=3, wavelet="haar", mode="periodization")

    print("Wavelet operators:")
    for k, v in Pi_ops.items():
        print(f"  {k}: {v.shape}")

    x_recon_wavelet = wv_info_transfer.idwt(Pi_ops, coeffs)

    dinv.utils.plot([x, x_recon_wavelet], titles=["Original", "Reconstructed from wavelets"])