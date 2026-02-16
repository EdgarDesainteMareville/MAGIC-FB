import deepinv as dinv
import torch

from deepinv.physics import Blur
from utils.matrix import create_conv_matrix_1d, create_gaussian_kernel_1d

class BlurMatrix(Blur):
    """
    Blur operator represented as a matrix.
    """

    def __init__(self, filter_row, filter_col, padding="circular", device="cpu", **kwargs):
        filter2d = torch.outer(filter_row, filter_col)
        super().__init__(filter2d, padding=padding, device=device, **kwargs)

        self.filter_row = filter_row
        self.filter_col = filter_col
        self.A_row = None
        self.A_col = None
        self.signal_size = None

        self.use_sparse_matrices = kwargs.get("use_sparse_matrices", False)

    def create_A_matrix(self, signal_size):
        """
        Create the matrix representation of the blur operator. Only for 1D filters
        """
        A_row = create_conv_matrix_1d(
            self.filter_row, signal_size, padding=self.padding, device=self.device, dtype=torch.float32
        )
        A_col = create_conv_matrix_1d(
            self.filter_col, signal_size, padding=self.padding, device=self.device, dtype=torch.float32
        )

        if self.use_sparse_matrices:
            A_row = A_row.to_sparse()
            A_col = A_col.to_sparse()

        return A_row, A_col

    def ensure_A_matrices(self, signal_size):
        """
        Ensure A_row and A_col exist and match the given signal size.
        Recreate them if necessary.
        """
        if (
            self.A_row is None
            or self.A_col is None
            or self.signal_size != signal_size
            or self.A_row.shape[0] != signal_size
            or self.A_col.shape[0] != signal_size
        ):
            self.A_row, self.A_col = self.create_A_matrix(signal_size)
            self.signal_size = signal_size

    def get_A_matrices(self, signal_size):
        self.ensure_A_matrices(signal_size)
        return self.A_row, self.A_col

    def A(self, x):
        signal_size = x.shape[-1]
        self.ensure_A_matrices(signal_size)
        return self.A_row @ x @ self.A_col.T


    def A_adjoint(self, y):
        signal_size = y.shape[-1]
        self.ensure_A_matrices(signal_size)
        return self.A_row.T @ y @ self.A_col

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage
    x = dinv.utils.load_example("butterfly.png", grayscale=False)

    filter_row = create_gaussian_kernel_1d(size=9, sigma=2.0, device='cpu', dtype=torch.float32)
    filter_col = create_gaussian_kernel_1d(size=9, sigma=2.0, device='cpu', dtype=torch.float32)
    physics = BlurMatrix(filter_row, filter_col, padding="circular", use_sparse_matrices=True)

    A_row, A_col = physics.get_A_matrices(signal_size=x.shape[-1])
    print("A_row shape:", A_row.shape)
    print("A_col shape:", A_col.shape)

    y = physics.A(x)

    dinv.utils.plot([x, y])