import torch
import torch.nn.functional as F

def create_conv_matrix_1d(kernel, signal_size, padding="circular", device='cpu', dtype=None):
    """
    Constructs the convolution matrix for a given 1D kernel and signal size.
    """
    kernel = kernel.to(device=device, dtype=dtype)
    k = kernel.numel()

    # Create identity matrix of size signal_size
    I = torch.eye(signal_size, device=device, dtype=dtype)

    # Apply convolution to each basis vector e_i
    # This is equivalent to explicitly constructing the operator matrix
    A_rows = []
    pad_left = (k - 1) // 2
    pad_right = (k - 1) // 2

    for i in range(signal_size):
        e_i = I[i].view(1, 1, -1)  # batch=1, channel=1

        if padding == "circular":
            e_padded = torch.cat([e_i[:, :, -pad_left:], e_i, e_i[:, :, :pad_right]], dim=-1)
            y_i = F.conv1d(e_padded, kernel.view(1, 1, -1))
        elif padding == "reflect":
            y_i = F.conv1d(F.pad(e_i, (pad_left, pad_right), mode="reflect"), kernel.view(1, 1, -1))
        else:
            # zero padding
            y_i = F.conv1d(F.pad(e_i, (pad_left, pad_right)), kernel.view(1, 1, -1))

        A_rows.append(y_i.view(-1)[:signal_size])

    A = torch.stack(A_rows, dim=0)
    return A


def create_gaussian_kernel_1d(size=None, sigma=1.0, device='cpu', dtype=None):
    """
    Generates a normalized 1D Gaussian kernel.
    If 'size' is not specified, it is calculated automatically from sigma.
    """
    if size is None:
        c = int(sigma/0.3 + 1)
        size = 2 * c + 1
    else:
        c = (size - 1) // 2

    # Centered coordinates
    coords = torch.arange(size, device=device, dtype=dtype) - c

    # Gaussian computation
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)

    # Normalization
    kernel /= kernel.sum()

    return kernel