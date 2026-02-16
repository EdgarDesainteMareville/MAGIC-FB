import sys
import deepinv as dinv
import torch
import torch.nn.functional as F
import pywt
import time
import copy
import numpy as np
from tqdm import tqdm

PSNR = dinv.metric.PSNR()

from utils.information_transfer import (
    WaveletInformationTransferMatrices,
)
from utils.priors import WaveletPriorCustom


class BlockCoordinateDescent:
    def __init__(
        self,
        img_size,
        wv_type,
        physics,
        data_fidelity,
        prior,
        max_levels,
        stepsize=1e-3,
        device="cpu",
        denoiser_list=None,
    ):
        self.img_size = img_size
        self.wv_type = wv_type
        self.physics = physics
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.max_levels = max_levels
        self.device = device

        self.denoiser_list = denoiser_list

        self.stepsize = stepsize
        self.reg_weight = None
        self.y = None

        self.exact = True  # Whether to use exact gradient updates or partial ones

        self.grad_matrices = {}  # Matrices for the gradient computation
        self.grad_wavelet_terms = {}  # Intermediate terms for the gradient computation
        self.grad_wavelet = {}  # Gradient in the wavelet domain
        self.proj_ATy = {}  # Projections of A^T y onto the wavelet subspaces

        self.grad_g_wavelet = (
            {}
        )  # Gradient of the smoothed regularization term in the wavelet domain

        self.full_gradient = (
            {}
        )  # Full gradient (fidelity + smoothed regularization) in the wavelet domain

        self.grad_f_norm_approx = (
            []
        )  # Partial gradient norms for the fidelity term (approximation coefficients)
        self.grad_f_norm_details = (
            {}
        )  # Partial gradient norms for the fidelity term (detail coefficients)
        self.grad_g_norm_approx = (
            []
        )  # Partial gradient norms for the smoothed regularization term (approximation coefficients)
        self.grad_g_norm_details = (
            {}
        )  # Partial gradient norms for the smoothed regularization term (detail coefficients)

        self.full_grad_norm_approx = (
            []
        )  # Partial gradient norms for the full gradient (approximation coefficients)
        self.full_grad_norm_details = (
            {}
        )  # Partial gradient norms for the full gradient (detail coefficients)

        self.step_lengths = {}  # Step lengths for each block
        self.step_lengths_norm_approx = (
            []
        )  # Step length norms for the approximation coefficients
        self.step_lengths_norm_details = (
            {}
        )  # Step length norms for the detail coefficients

        self.updated_blocks_history = []  # History of updated blocks for visualization

        # self.wavelet_prior = dinv.optim.WaveletPrior(level=self.max_levels, wv=self.wv_type, device=self.device)
        self.wavelet_prior = WaveletPriorCustom(
            level=self.max_levels, wv=self.wv_type, device=self.device
        )
        self.information_transfer = WaveletInformationTransferMatrices(
            wv=self.wv_type,
            mode="periodization",
            level=self.max_levels,
            device=self.device,
        )
        self.Pi_ops = self.information_transfer.compute_Pi_operators(
            N=self.img_size[-1],
            levels=self.max_levels,
            wavelet=self.wv_type,
            mode="periodization",
        )

    def compute_gradient_f_norms(self):
        # Compute partial gradient norms of fidelity term for each block

        # Initialize dictionaries
        for level in range(self.max_levels):
            if level not in self.grad_f_norm_details:
                self.grad_f_norm_details[level] = []

        # Compute partial gradient norm of approximation block
        grad_norm_approx = torch.norm(self.grad_wavelet[f"A{self.max_levels}"]).item()

        # Normalize according to subimage size
        grad_norm_approx /= self.grad_wavelet[f"A{self.max_levels}"].numel() ** 0.5
        self.grad_f_norm_approx.append(grad_norm_approx)

        # Compute partial gradient norms of detail blocks
        for scale in range(self.max_levels):
            level = self.max_levels - scale
            grad_norm_H = torch.norm(self.grad_wavelet[f"H{level}"]).item()
            grad_norm_H /= self.grad_wavelet[f"H{level}"].numel() ** 0.5

            grad_norm_V = torch.norm(self.grad_wavelet[f"V{level}"]).item()
            grad_norm_V /= self.grad_wavelet[f"V{level}"].numel() ** 0.5

            grad_norm_D = torch.norm(self.grad_wavelet[f"D{level}"]).item()
            grad_norm_D /= self.grad_wavelet[f"D{level}"].numel() ** 0.5

            grad_norm_det = (grad_norm_H**2 + grad_norm_V**2 + grad_norm_D**2) ** 0.5
            self.grad_f_norm_details[scale].append(grad_norm_det)

    def compute_gradient_regularization(self, x_wavelet):
        # Compute the gradient of the Moreau envelope of the regularization term for every block

        approx = x_wavelet[f"A{self.max_levels}"]
        grad_g_approx = 0 * approx
        self.grad_g_wavelet[f"A{self.max_levels}"] = grad_g_approx

        for level in range(self.max_levels, 0, -1):
            details_H = x_wavelet[f"H{level}"]
            details_V = x_wavelet[f"V{level}"]
            details_D = x_wavelet[f"D{level}"]
            grad_g_H = (1 / (self.stepsize)) * (
                details_H
                - self.prior.prox(details_H, gamma=self.stepsize * self.reg_weight)
            )
            grad_g_V = (1 / (self.stepsize)) * (
                details_V
                - self.prior.prox(details_V, gamma=self.stepsize * self.reg_weight)
            )
            grad_g_D = (1 / (self.stepsize)) * (
                details_D
                - self.prior.prox(details_D, gamma=self.stepsize * self.reg_weight)
            )
            self.grad_g_wavelet[f"H{level}"] = grad_g_H
            self.grad_g_wavelet[f"V{level}"] = grad_g_V
            self.grad_g_wavelet[f"D{level}"] = grad_g_D

    def compute_full_gradient(self, x_wavelet):
        # Compute the partial full gradients (fidelity + smoothed regularization) in the wavelet domain
        self.compute_gradient_regularization(x_wavelet)
        for Xchar in ["A", "V", "H", "D"]:
            Xlevel = (
                [self.max_levels] if Xchar == "A" else range(self.max_levels, 0, -1)
            )
            for lX in Xlevel:
                keyX = f"{Xchar}{lX}"
                grad_f_X = self.grad_wavelet[keyX]
                grad_g_X = self.grad_g_wavelet[keyX]
                # print(f'Full gradient block {keyX}: fidelity norm {torch.norm(grad_f_X).item():.4f}, regularization norm {torch.norm(grad_g_X).item():.4f}')
                self.full_gradient[keyX] = grad_f_X + grad_g_X

    def compute_full_gradient_norms(self):
        # Compute partial full gradient norms for each block

        # Initialize dictionaries
        for level in range(self.max_levels):
            if level not in self.full_grad_norm_details:
                self.full_grad_norm_details[level] = []

        # Compute partial full gradient norm of approximation block
        full_grad_approx = self.full_gradient[f"A{self.max_levels}"]
        grad_norm_approx = torch.norm(full_grad_approx).item()
        grad_norm_approx /= full_grad_approx.numel() ** 0.5
        self.full_grad_norm_approx.append(grad_norm_approx)

        # Compute partial full gradient norms of detail blocks
        for scale in range(self.max_levels):
            level = self.max_levels - scale
            full_grad_H = self.full_gradient[f"H{level}"]
            full_grad_V = self.full_gradient[f"V{level}"]
            full_grad_D = self.full_gradient[f"D{level}"]

            norm_H = torch.norm(full_grad_H).item() / full_grad_H.numel() ** 0.5
            norm_V = torch.norm(full_grad_V).item() / full_grad_V.numel() ** 0.5
            norm_D = torch.norm(full_grad_D).item() / full_grad_D.numel() ** 0.5

            full_grad_norm = (norm_H**2 + norm_V**2 + norm_D**2) ** 0.5
            self.full_grad_norm_details[scale].append(full_grad_norm)

    def compute_step_lengths(self, x_wavelet):
        # Compute step lengths for each block: xk^i - prox_{\gamma g_i}(xk^i - \gamma \nabla_i f(xk))
        self.step_lengths = {}
        for Xchar in ["A", "V", "H", "D"]:
            Xlevel = (
                [self.max_levels] if Xchar == "A" else range(self.max_levels, 0, -1)
            )
            for lX in Xlevel:
                keyX = f"{Xchar}{lX}"
                grad_f_X = self.grad_wavelet[keyX]
                xk_X = x_wavelet[keyX]
                x_temp = xk_X - self.stepsize * grad_f_X

                if Xchar == "A":
                    prox_X = x_temp  # No regularization on approximation
                else:
                    prox_X = self.prior.prox(
                        x_temp, gamma=self.stepsize * self.reg_weight
                    )

                step_length = xk_X - prox_X
                self.step_lengths[keyX] = step_length

    def compute_step_length_norms(self):
        # Initialize dictionaries for detail blocks
        for scale in range(self.max_levels):
            if scale not in self.step_lengths_norm_details:
                self.step_lengths_norm_details[scale] = []

        # Approx block
        step_length_approx = self.step_lengths[f"A{self.max_levels}"]
        norm_approx = torch.norm(step_length_approx).item() / (
            step_length_approx.numel() ** 0.5
        )
        self.step_lengths_norm_approx.append(norm_approx)

        # Detail blocks (scale=0 -> level=max_levels, ..., scale=max_levels-1 -> level=1)
        for scale in range(self.max_levels):
            level = self.max_levels - scale

            step_length_H = self.step_lengths[f"H{level}"]
            step_length_V = self.step_lengths[f"V{level}"]
            step_length_D = self.step_lengths[f"D{level}"]

            norm_H = torch.norm(step_length_H).item() / (step_length_H.numel() ** 0.5)
            norm_V = torch.norm(step_length_V).item() / (step_length_V.numel() ** 0.5)
            norm_D = torch.norm(step_length_D).item() / (step_length_D.numel() ** 0.5)

            step_length_norm = (norm_H**2 + norm_V**2 + norm_D**2) ** 0.5
            self.step_lengths_norm_details[scale].append(step_length_norm)

    def update_blocks(
        self,
        x_wavelet,
        n_iter_coarse,
        updated_blocks,
        reg_weight,
        use_conditional_thresholding=False,
    ):
        # Update list is a list of tuples (level, mode) where mode is 'approx' or 'details'

        for scale, mode in updated_blocks:
            level = self.max_levels - scale
            if mode == "approx":
                key = f"A{self.max_levels}"
                self.compute_gradient_f_norms()
                coeff = x_wavelet[f"A{self.max_levels}"]

                for i in range(n_iter_coarse):
                    # No regularization on approximation, so simple gradient step
                    coeff = (
                        coeff - self.stepsize * self.grad_wavelet[f"A{self.max_levels}"]
                    )

                    x_wavelet[key] = coeff
                    self.current_iter += 1
                    self.compute_metrics(x_wavelet)

            elif mode == "details":
                for char in ["V", "H", "D"]:
                    key = f"{char}{level}"
                    self.compute_gradient_f_norms()
                    coeff = x_wavelet[f"{char}{level}"]

                    for i in range(n_iter_coarse):
                        coeff = (
                            coeff - self.stepsize * self.grad_wavelet[f"{char}{level}"]
                        )

                        coeff = self.prior.prox(
                            coeff, gamma=self.stepsize * reg_weight
                        )

                        x_wavelet[key] = coeff
                        self.current_iter += 1
                        self.compute_metrics(x_wavelet)

            else:
                raise ValueError("Invalid mode. Choose 'details' or 'approx'.")

        # print('Updating gradient...\n')
        self.update_gradient(updated_blocks, x_wavelet)
        """if self.exact:
            self.update_gradient(updated_blocks, x_wavelet)
        else:
            for block in updated_blocks:
                self.update_gradient_partial(block, x_wavelet)"""

        return x_wavelet

    def compute_gradient_matrices(self):
        A_row_TA_row = self.physics.A_row.T @ self.physics.A_row
        A_col_TA_col = self.physics.A_col.T @ self.physics.A_col
        types = ["A", "V", "H", "D"]

        for X in types:
            # Niveau max pour ce type
            if X == "A":
                level_X = [self.max_levels]  # A uniquement au niveau max
            else:
                level_X = range(
                    self.max_levels, 0, -1
                )  # V,H,D aux niveaux max, max-1, max-2

            for lX in level_X:
                for Y in types:
                    # Niveau cible : tous les niveaux possibles pour V,H,D, mais seulement max pour A
                    if Y == "A":
                        level_Y = [self.max_levels]
                    else:
                        level_Y = range(self.max_levels, 0, -1)

                    for lY in level_Y:
                        key_row = f"{X}{lX}_{Y}{lY}_row"
                        key_col = f"{X}{lX}_{Y}{lY}_col"

                        PiX_row = self.Pi_ops[f"{X}{lX}_row"]
                        PiY_row = self.Pi_ops[f"{Y}{lY}_row"]
                        PiX_col = self.Pi_ops[f"{X}{lX}_col"]
                        PiY_col = self.Pi_ops[f"{Y}{lY}_col"]

                        self.grad_matrices[key_row] = PiX_row @ A_row_TA_row @ PiY_row.T
                        self.grad_matrices[key_col] = PiX_col @ A_col_TA_col @ PiY_col.T

        # Projections A^T y
        ATy = self.physics.A_row.T @ self.y @ self.physics.A_col
        for X in types:
            if X == "A":
                levels = [self.max_levels]
            else:
                levels = range(self.max_levels, 0, -1)

            for l in levels:
                PiX_row = self.Pi_ops[f"{X}{l}_row"]
                PiX_col = self.Pi_ops[f"{X}{l}_col"]
                self.proj_ATy[f"{X}{l}"] = PiX_row @ ATy @ PiX_col.T

    def compute_gradient_terms(self, x_wavelet, Xchar, Xlevel, Ychar, Ylevel):
        # Compute intermediate terms for gradient computation
        # i.e. computes \Pi_X A^T A \Pi_Y^T x_wavelet[Y]

        # print(f"Computing gradient term for blocks {Xchar}{Xlevel} and {Ychar}{Ylevel}...")

        key_row = f"{Xchar}{Xlevel}_{Ychar}{Ylevel}_row"
        key_col = f"{Xchar}{Xlevel}_{Ychar}{Ylevel}_col"

        term = (
            self.grad_matrices[key_row]
            @ x_wavelet[f"{Ychar}{Ylevel}"]
            @ self.grad_matrices[key_col].T
        )
        self.grad_wavelet_terms[f"{Xchar}{Xlevel}_{Ychar}{Ylevel}"] = term

    def compute_gradient(self, x_wavelet, Xchar, Xlevel):
        # Compute the gradient for a given block
        # i.e. computes \sum_Y ( \Pi_X A^T A \Pi_Y^T x_wavelet[Y] ) - \Pi_X A^T y
        keyX = f"{Xchar}{Xlevel}"
        # print(f"Computing gradient for block {keyX}...")

        grad_X = -self.proj_ATy[keyX]

        for Ychar in ["A", "V", "H", "D"]:
            keyY_levels = (
                [self.max_levels] if Ychar == "A" else range(self.max_levels, 0, -1)
            )
            for lY in keyY_levels:
                key_row = f"{Xchar}{Xlevel}_{Ychar}{lY}_row"
                key_col = f"{Xchar}{Xlevel}_{Ychar}{lY}_col"

                self.compute_gradient_terms(x_wavelet, Xchar, Xlevel, Ychar, lY)
                grad_X += self.grad_wavelet_terms[f"{Xchar}{Xlevel}_{Ychar}{lY}"]

        self.grad_wavelet[keyX] = grad_X

    def update_gradient(self, updated_blocks, x_wavelet):
        for scale, mode in updated_blocks:
            level = self.max_levels - scale

            if mode == "approx":
                # If we just updated the approximation
                for Xchar in ["A", "V", "H", "D"]:
                    Xlevel = (
                        [self.max_levels]
                        if Xchar == "A"
                        else range(self.max_levels, 0, -1)
                    )
                    for lX in Xlevel:
                        # Re-compute the terms \Pi_X A^T A \Pi_A^T x_wavelet[A] for every block X
                        # (the only thing that changed is x_wavelet[A])
                        self.compute_gradient_terms(
                            x_wavelet, Xchar, lX, "A", self.max_levels
                        )
                        self.compute_gradient(x_wavelet, Xchar, lX)

            elif mode == "details":
                # If we just updated the details at level 'level'
                for t in ["V", "H", "D"]:
                    Ychar = t
                    Ylevel = level
                    for Xchar in ["A", "V", "H", "D"]:
                        Xlevel = (
                            [self.max_levels]
                            if Xchar == "A"
                            else range(self.max_levels, 0, -1)
                        )
                        for lX in Xlevel:
                            # Re-compute the terms \Pi_X A^T A \Pi_Y^T x_wavelet[Y] for every block X
                            # (the only thing that changed is x_wavelet[Y] for Y=(t,level))
                            self.compute_gradient_terms(
                                x_wavelet, Xchar, lX, Ychar, Ylevel
                            )
                            self.compute_gradient(x_wavelet, Xchar, lX)

            else:
                raise ValueError("Invalid mode. Choose 'details' or 'approx'.")

    def update_gradient_partial(self, block, x_wavelet):
        scale, mode = block
        level = self.max_levels - scale

        # Re-compute only the term corresponding to the updated block
        if mode == "approx":
            self.compute_gradient_terms(
                x_wavelet, "A", self.max_levels, "A", self.max_levels
            )
            # Sum to get the gradient associated to selected block
            keyX = f"A{self.max_levels}"
            grad_X = -self.proj_ATy[keyX]
            for Ychar in ["A", "V", "H", "D"]:
                for lY in (
                    [self.max_levels] if Ychar == "A" else range(self.max_levels, 0, -1)
                ):
                    grad_X += self.grad_wavelet_terms[f"{keyX}_{Ychar}{lY}"]
            self.grad_wavelet[keyX] = grad_X

        elif mode == "details":
            for t in ["V", "H", "D"]:
                self.compute_gradient_terms(x_wavelet, t, level, t, level)
                # Sum to get the gradient associated to selected block
                keyX = f"{t}{level}"
                grad_X = -self.proj_ATy[keyX]
                for Ychar in ["A", "V", "H", "D"]:
                    for lY in (
                        [self.max_levels]
                        if Ychar == "A"
                        else range(self.max_levels, 0, -1)
                    ):
                        grad_X += self.grad_wavelet_terms[f"{keyX}_{Ychar}{lY}"]
                self.grad_wavelet[keyX] = grad_X

    def run(
        self,
        y,
        x0,
        x_true,
        n_iter,
        n_iter_coarse,
        reg_weight,
        distribution="gauss_southwell",
        update_mode="MLFB",
        use_conditional_thresholding=False,
        metrics=False,
        t_max=None,
    ):

        # Set attributes
        self.reg_weight = reg_weight
        self.y = y
        self.x_true = x_true

        self.ATy = self.physics.A_adjoint(self.y)

        self.losses = []
        self.times = []
        self.psnrs = []
        self.current_iter = 0
        self.cycles = []
        self.update_probabilities = []
        self.full_grad_norm_approx = []
        self.full_grad_norm_details = {}
        self.step_lengths_norm_approx = []
        self.step_lengths_norm_details = {}

        self.t_gs_step_total = 0.0
        self.t_gs_step_calls = 0
        self.t_gs_step_history = []
        self.step_lengths_norms_history = []

        self.updated_blocks_history = []

        self.last_recon_under_tmax = None
        self.last_time_under_tmax = None
        self.last_psnr_under_tmax = None

        self.t_max = t_max
        self.start_time = None  # will be set when metrics start
        self._stop_due_to_time = False

        self.compute_gradient_matrices()

        # Initial wavelet coefficients
        xk_wavelet = self.information_transfer.dwt(x0)

        # Initial gradient computation
        for Xchar in ["A", "V", "H", "D"]:
            Xlevel = (
                [self.max_levels] if Xchar == "A" else range(self.max_levels, 0, -1)
            )
            for lX in Xlevel:
                self.compute_gradient(x_wavelet=xk_wavelet, Xchar=Xchar, Xlevel=lX)

        # Update list
        updatelist_inst = UpdateList(self.max_levels)
        if update_mode != "stochastic":
            update_list = updatelist_inst.get_list(type=update_mode)
            self.updated_blocks_history.append(copy.deepcopy(update_list))

        if metrics:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            self.start_time = start

        # Compute initial metrics
        self.compute_metrics(x_wavelet=None, x_img=x0)

        with torch.no_grad():
            for it in range(n_iter):
                if metrics:
                    x_recon = self.information_transfer.idwt(self.Pi_ops, xk_wavelet)

                # If the update mode is stochastic, we need to generate a new update list at each iteration
                if update_mode == "stochastic":

                    if distribution == "uniform":
                        uniform = 0.5 * np.ones(self.max_levels + 1)
                        update_list = updatelist_inst.get_list(
                            type="stochastic", probabilities=uniform
                        )
                        self.updated_blocks_history.append(copy.deepcopy(update_list))

                    elif distribution == "deterministic_gauss_southwell":
                        self.compute_step_lengths(xk_wavelet)
                        self.compute_step_length_norms()

                        step_length_norms = [self.step_lengths_norm_approx[-1]]
                        for level in range(self.max_levels):
                            step_length_norms += [
                                self.step_lengths_norm_details[level][-1]
                            ]

                        step_length_norms = np.array(step_length_norms, dtype=float)
                        self.step_lengths_norms_history.append(step_length_norms.copy())

                        # One for the max, zero for the others
                        update_probabilities = np.zeros_like(step_length_norms)
                        max_index = int(np.argmax(step_length_norms))
                        update_probabilities[max_index] = 1.0

                        self.update_probabilities.append(update_probabilities.copy())

                        # Generate new update list based on updated probabilities
                        update_list = updatelist_inst.get_list(
                            type="stochastic", probabilities=update_probabilities
                        )
                        self.updated_blocks_history.append(copy.deepcopy(update_list))

                    elif distribution == "stochastic_gauss_southwell":
                        self.compute_step_lengths(xk_wavelet)
                        self.compute_step_length_norms()

                        step_length_norms = [self.step_lengths_norm_approx[-1]]
                        for level in range(self.max_levels):
                            step_length_norms += [
                                self.step_lengths_norm_details[level][-1]
                            ]

                        step_length_norms = np.array(step_length_norms, dtype=float)
                        update_probabilities = step_length_norms / np.linalg.norm(
                            step_length_norms
                        )
                        self.update_probabilities.append(
                            update_probabilities.copy()
                        )

                        # Generate new update list based on updated probabilities
                        update_list = updatelist_inst.get_list(
                            type="stochastic", probabilities=update_probabilities
                        )
                        self.updated_blocks_history.append(
                            copy.deepcopy(update_list)
                        )

                    else:
                        raise ValueError(f"Unknown distribution: {distribution}")
                else:
                    self.updated_blocks_history.append(copy.deepcopy(update_list))

                # Update coefficients
                for updated_blocks in update_list:
                    xk_wavelet = self.update_blocks(
                        xk_wavelet,
                        n_iter_coarse=n_iter_coarse,
                        updated_blocks=updated_blocks,
                        reg_weight=reg_weight,
                        use_conditional_thresholding=use_conditional_thresholding,
                    )

                    # If we exceeded t_max inside compute_metrics, break out
                    if metrics and self._stop_due_to_time:
                        break

                self.cycles.append(self.current_iter)

                if metrics and self._stop_due_to_time:
                    break

        x_recon = self.information_transfer.idwt(self.Pi_ops, xk_wavelet)

        if metrics:
            self.times = [t - start for t in self.times]  # shift to start at 0

        if t_max is not None:
            return (
                x_recon,
                self.losses,
                self.times,
                self.cycles,
                self.psnrs,
                self.last_recon_under_tmax,
                self.last_time_under_tmax,
                self.last_psnr_under_tmax,
            )

        return x_recon, self.losses, self.times, self.cycles, self.psnrs

    def compute_metrics(self, x_wavelet, x_img=None):
        if x_img is None:
            x_img = self.information_transfer.idwt(self.Pi_ops, x_wavelet)
        crit = (
            self.data_fidelity.fn(x_img, self.y, self.physics).item()
            + self.reg_weight * self.wavelet_prior.fn(x_img).item()
        )
        self.losses.append(crit)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.times.append(time.perf_counter())
        else:
            self.times.append(time.process_time())
        self.psnrs.append(PSNR(x_img, self.x_true).item())

        # ---- keep only the last reconstruction under the time budget ----
        if getattr(self, "t_max", None) is not None:
            # times stores raw perf_counter (or process_time) values for now; run() will shift later
            elapsed = self.times[-1] - self.start_time
            if elapsed <= self.t_max:
                self.last_recon_under_tmax = x_img.detach().cpu()
                self.last_time_under_tmax = float(elapsed)
                self.last_psnr_under_tmax = float(self.psnrs[-1])
            else:
                # signal to break out of the main loop
                self._stop_due_to_time = True


class UpdateList:
    def __init__(self, max_levels):
        self.max_levels = max_levels

    def get_list(self, type="MLFB", probabilities=None):
        if type == "MLFB":
            return self.create_update_list_MLFB()
        elif type == "FB":
            return self.create_update_list_FB()
        elif type == "MLFBdetails":
            return self.create_update_list_MLFBdetails()
        elif type == "cyclic":
            return self.create_update_list_cyclic()
        elif type == "approx":
            return self.create_update_list_approx()
        elif type == "details":
            return self.create_update_list_details()
        elif type == "stochastic":
            return self.create_update_list_stochastic(probabilities=probabilities)
        else:
            raise ValueError(
                "Invalid type. Choose 'MLFB', 'FB', 'MLFBdetails' or 'cyclic'."
            )

    def create_update_list_MLFB(self, n_iter_coarse=1):
        update_list = [[(0, "approx")]] * n_iter_coarse
        updated_blocks = [(0, "approx")]
        for i in range(self.max_levels):
            updated_blocks.append((i, "details"))
            for _ in range(n_iter_coarse):
                update_list.append(copy.deepcopy(updated_blocks))
        return update_list

    def create_update_list_FB(self):
        update_list = []
        updated_blocks = [(0, "approx")] + [
            (level, "details") for level in range(self.max_levels)
        ]
        update_list = [copy.deepcopy(updated_blocks)]
        return update_list

    def create_update_list_MLFBdetails(self):
        update_list = [[(0, "approx")]]
        updated_blocks = [(0, "approx")]
        for i in range(self.max_levels):
            update_list.append([(i, "details")])
            updated_blocks.append((i, "details"))
            update_list.append(copy.deepcopy(updated_blocks))
        return update_list

    def create_update_list_cyclic(self):
        update_list = [[(0, "approx")]]
        for i in range(self.max_levels):
            update_list.append([(i, "details")])
        return update_list

    def create_update_list_approx(self):
        update_list = [[(0, "approx")]]
        return update_list

    def create_update_list_details(self):
        update_list = []
        for i in range(self.max_levels):
            update_list.append([(i, "details")])
        return update_list

    def create_update_list_stochastic(self, probabilities=None):
        """
        Stochastic update list :
        - Probabilities per level: probabilities[0] for approximation,
          probabilities[1] for details level 0, ...,
          probabilities[max_levels + 1] for details level max_levels
        - If no block is selected, select the one with highest probability
        """
        if probabilities is None:
            raise ValueError("Probabilities must be provided.")

        update_list = []
        current = []

        # Independant Bernoulli for each level
        if np.random.rand() < probabilities[0]:
            current.append((0, "approx"))

        for level in range(1, self.max_levels + 1):
            if np.random.rand() < probabilities[level]:
                current.append((level - 1, "details"))
        update_list.append(copy.deepcopy(current))

        # If no block selected, force the one with highest probability
        if not any(len(lst) > 0 for lst in update_list):
            highest_proba_index = int(np.argmax(probabilities))

            if highest_proba_index == 0:
                forced = [(0, "approx")]
            else:
                forced = [(highest_proba_index - 1, "details")]

            # Choose only this block to update
            update_list = [forced]

        return update_list


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.physics import BlurMatrix
    from utils.matrix import create_gaussian_kernel_1d

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    plt.style.use("tableau-colorblind10")

    x_true = dinv.utils.load_image(
        "images/corals.png", img_size=1024, device=device
    )

    # Wavelet parameters
    J = 5
    wv_type = "db8"

    # Physics
    filter_row = create_gaussian_kernel_1d(sigma=10, device=device, dtype=torch.float32)
    filter_col = create_gaussian_kernel_1d(sigma=10, device=device, dtype=torch.float32)
    physics = BlurMatrix(filter_row, filter_col, padding="circular", device=device)
    sigma = 0.01

    physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)
    # Observation
    y = physics(x_true)

    # Objective function
    data_fidelity = dinv.optim.L2()
    prior = dinv.optim.L1Prior()
    reg_weight = 1e-5

    # Parameters
    n_iter = 10
    n_iter_coarse = 1  # Number of coarse iterations for MLFB
    Anorm2 = physics.compute_norm(x_true).item()
    stepsize = 1.9 / Anorm2
    print(f"Using device: {device}")
    print(f"Stepsize: {stepsize}")

    bcd = BlockCoordinateDescent(
        x_true.shape,
        wv_type=wv_type,
        physics=physics,
        data_fidelity=data_fidelity,
        prior=prior,
        max_levels=J,
        stepsize=stepsize,
        device=device,
    )

    update_mode = "stochastic_gauss_southwell"
    print(f"\nRunning BCD {update_mode} ...")

    x0 = y.clone()
    x_recon_1, loss_1, times_1, cycles_1, psnr_1 = bcd.run(
        y,
        x0,
        x_true=x_true,
        n_iter=n_iter,
        n_iter_coarse=n_iter_coarse,
        reg_weight=reg_weight,
        update_mode='stochastic',
        distribution=update_mode,
        metrics=True,
        use_conditional_thresholding=False,
    )

    updated_blocks_history_1 = bcd.updated_blocks_history

    plt.figure()
    plt.plot(loss_1, label=f"BCD {update_mode}")
    plt.xlabel("Iterations")
    plt.ylabel("Objective function")
    plt.legend()
    plt.savefig("bcd_objective_iterations.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(times_1, loss_1, label=f"BCD {update_mode}")
    plt.xlabel("CPU time (s)")
    plt.ylabel("Objective function")
    plt.legend()
    plt.savefig("bcd_objective_time.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(psnr_1, label=f"BCD {update_mode}", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("PSNR")
    plt.legend()
    plt.savefig("bcd_psnr_iterations.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(times_1, psnr_1, label=f"BCD {update_mode}", color="blue")
    plt.xlabel("CPU time (s)")
    plt.ylabel("PSNR")
    plt.legend()
    plt.savefig("bcd_psnr_time.png", dpi=300)
    plt.show()

    dinv.utils.plot(
        [x_true, y, x_recon_1],
        titles=[
            "Original",
            "Observation",
            f"BCD {update_mode}",
        ],
        subtitles=[
            "PSNR:",
            f"{PSNR(y, x_true).item():.2f}",
            f"{PSNR(x_recon_1, x_true).item():.2f}",
        ],
        save_fn="reconstruction.png",
    )