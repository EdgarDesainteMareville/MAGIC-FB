import numpy as np
import matplotlib.pyplot as plt

def plot_block_activation_matrix(
    updated_blocks_history,
    J,
    ax=None,
    title="Updated blocks per iteration",
    xlabel="Iteration",
    ylabel="Block",
    show_colorbar=True,
    savepath=None,
):
    """
    Plot a binary activation matrix of updated blocks over iterations,
    with details (D*) at the top and approximation (A) at the bottom.
    """

    n_iters = len(updated_blocks_history)
    n_blocks = J + 1  # 1 approx + J detail levels
    update_matrix = np.zeros((n_iters, n_blocks), dtype=int)

    for it, update_list in enumerate(updated_blocks_history):
        active_blocks = set()

        # update_list: list of "groups" of blocks, each group is a list of (scale, mode)
        for group in update_list:
            for scale, mode in group:
                if mode == "approx":
                    idx = 0
                elif mode == "details":
                    idx = scale + 1  # scale=0 -> details level J, ..., scale=J-1 -> level 1
                else:
                    continue
                active_blocks.add(idx)

        for idx in active_blocks:
            update_matrix[it, idx] = 1

    # Reorder rows for display: D_J, D_{J-1}, ..., D_1, A
    # Current columns: [A, D_J, D_{J-1}, ..., D_1] after transpose
    display_order = list(range(n_blocks - 1, -1, -1))  # reverse order: [J, J-1, ..., 1, 0]
    mat_display = update_matrix[:, display_order].T

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    im = ax.imshow(mat_display, aspect="auto", interpolation="nearest")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    yticks = list(range(n_blocks))
    yticklabels = [f"D{s}" for s in range(1, J+1)] + ["A"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Updated (1) / Not updated (0)")

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return update_matrix
