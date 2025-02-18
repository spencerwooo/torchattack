import math
import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from torchattack.attack import Attack, register_attack
from torchattack.attack_model import AttackModel


def dct(x: int, y: int, v: int, u: int, n: int) -> float:
    # Normalisation
    def alpha(a: int) -> float:
        if a == 0:
            return math.sqrt(1.0 / n)
        else:
            return math.sqrt(2.0 / n)

    return (
        alpha(u)
        * alpha(v)
        * math.cos(((2 * x + 1) * (u * math.pi)) / (2 * n))
        * math.cos(((2 * y + 1) * (v * math.pi)) / (2 * n))
    )


def gen_2d_dct_sub_basis(sub_dim: int, n: int, path: str) -> np.ndarray:
    # Assume square images, so we don't have different xres and yres. We can get
    # different frequencies by setting u and v.

    # Here, we have a max u and v to loop over and display.
    max_u = sub_dim
    max_v = sub_dim

    dct_basis_list = []

    for u in range(0, max_u):
        if u % (max_u // 5) == 0:
            print(f'GeoDA: generating DCT subspace basis {u + 1}/{max_u}')
        for v in range(0, max_v):
            basis_img = np.zeros((n, n))
            for y in range(0, n):
                for x in range(0, n):
                    basis_img[y, x] = dct(x, y, v, u, max(n, max_v))

            dct_basis_list.append(basis_img)

    dct_basis = np.reshape(np.asarray(dct_basis_list), (max_v * max_u, n * n))
    dct_basis = np.asmatrix(dct_basis).transpose()

    # Cache the generated DCT subspace basis
    np.save(path, dct_basis)

    return dct_basis


class SubNoise(nn.Module):
    """Given subspace x and the number of noises, generate sub noises."""

    def __init__(self, num_noises: int, sub_basis: torch.Tensor) -> None:
        super(SubNoise, self).__init__()

        self.sub_basis = sub_basis
        self.num_noises = num_noises
        self.size = int(self.sub_basis.shape[0] ** 0.5)

    def forward(self) -> torch.Tensor:
        noise = torch.randn([self.sub_basis.shape[1], 3 * self.num_noises])
        noise = noise.to(self.sub_basis.device)
        sub_noise = torch.transpose(torch.mm(self.sub_basis, noise), 0, 1)
        return sub_noise.view([self.num_noises, 3, self.size, self.size])


@register_attack(category='NON_EPS')
class GeoDA(Attack):
    """The Geometric Decision-based Attack (GeoDA).

    Note:
        This attack does not fully support batch inputs. Batch size of more than 1 will
        generate adversarial perturbations with incorrect magnitude.

    > From the paper: [GeoDA: a geometric framework for black-box adversarial
    attacks](https://arxiv.org/abs/2003.06468).
    """

    def __init__(
        self,
        model: nn.Module | AttackModel,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
        input_shape: tuple = (3, 224, 224),
        epsilon: int = 5,
        p: str = 'l2',
        max_queries: int | float = 4e3,
        sub_dim: int = 75,
        tol: float = 1e-4,
        sigma: float = 2e-4,
        mu: float = 0.6,
        grad_estimator_batch_size: int = 40,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(model, normalize, device)

        self.epsilon = epsilon
        self.p = p
        self.max_queries = max_queries
        self.sub_dim = sub_dim
        self.tol = tol
        self.sigma = sigma
        self.mu = mu
        self.grad_estimator_batch_size = grad_estimator_batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Prepare the DCT basis
        self.x_size = input_shape[1]
        self.sub_basis_path = os.path.join(
            os.path.dirname(__file__),
            'output',
            f'2d_dct_basis_{self.sub_dim}_{self.x_size}.npy',
        )

    def opt_query_iteration(self, nq: int, t: int, eta: float) -> tuple[list[int], int]:
        """Determine optimal distribution of number of queries."""

        coefs = [eta ** (-2 * i / 3) for i in range(0, t)]
        coefs[0] = 1 * coefs[0]
        sum_coefs = sum(coefs)
        opt_q = [round(nq * coefs[i] / sum_coefs) for i in range(0, t)]

        if opt_q[0] > 80:
            t = t + 1
            opt_q, t = self.opt_query_iteration(nq, t, eta)
        elif opt_q[0] < 50:
            t = t - 1
            opt_q, t = self.opt_query_iteration(nq, t, eta)

        return opt_q, t

    def find_random_adversarial(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Find an adversarial example by random search."""

        nb_calls = 1
        step_size = 0.02
        perturbed = x

        while not self.is_adv(perturbed, y).all():
            pert = torch.randn(x.shape).to(self.device)

            perturbed = x + nb_calls * step_size * pert
            perturbed = torch.clamp(perturbed, 0, 1)
            nb_calls += 1

        return perturbed, nb_calls

    def bin_search(
        self, x: torch.Tensor, y: torch.Tensor, x_random: torch.Tensor, tol: float
    ) -> tuple[torch.Tensor, int]:
        """
        Find an example on the model's decision boundary between input x and random
        sample x_rand by binary search.
        """

        num_calls = 0
        adv = x_random
        cln = x

        while torch.norm(adv - cln) >= tol:
            mid = (cln + adv) / 2.0
            num_calls += 1

            if self.is_adv(mid, y).all():
                adv = mid
            else:
                cln = mid

        return adv, num_calls

    def go_to_boundary(
        self, x: torch.Tensor, y: torch.Tensor, grad: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Move x towards the model's decision boundary."""

        epsilon = 1
        nb_calls = 1
        perturbed = x

        if self.p == 'l2':
            grads = grad
        elif self.p == 'linf':
            grads = torch.sign(grad) / torch.norm(grad)
        else:
            raise ValueError(f'Unknown p-norm {self.p}')

        while not self.is_adv(perturbed, y).all():
            perturbed = x + (nb_calls * epsilon * grads[0].to(self.device))
            perturbed = torch.clamp(perturbed, 0, 1)

            nb_calls += 1

            if nb_calls > 100:
                print('Failed to project sample to boundary (too many iters)')
                break

        return perturbed, nb_calls

    def black_grad_batch(
        self,
        x_boundary: torch.Tensor,
        q_max: int,
        y_0: torch.Tensor,
        sub_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Calculate gradient towards decision boundary."""

        estimated_grad = []  # estimated gradients in each estimate_batch
        z = []  # sign of estimated_grad
        outs = []
        batch_size = self.grad_estimator_batch_size

        num_batchs = math.ceil(q_max / batch_size)
        last_batch = q_max - (num_batchs - 1) * batch_size
        est_noise = SubNoise(batch_size, sub_basis)
        all_noises = []

        for j in range(num_batchs):
            if j == num_batchs - 1:
                est_noise_last = SubNoise(last_batch, sub_basis)
                current_batch = est_noise_last()
                current_batch_np = current_batch.cpu().numpy()
                noisy_boundary = [
                    x_boundary[0, :, :, :].cpu().numpy()
                ] * last_batch + self.sigma * current_batch.cpu().numpy()

            else:
                current_batch = est_noise()
                current_batch_np = current_batch.cpu().numpy()
                noisy_boundary = [
                    x_boundary[0, :, :, :].cpu().numpy()
                ] * batch_size + self.sigma * current_batch.cpu().numpy()

            all_noises.append(current_batch_np)
            noisy_boundary_tensor = torch.from_numpy(noisy_boundary)
            predict_labels = self.predict(noisy_boundary_tensor)
            outs.append(predict_labels.cpu().numpy())

        all_noise = np.concatenate(all_noises, axis=0)
        outs = np.concatenate(outs, axis=0)

        for i, predict in enumerate(outs):
            if predict == y_0:
                z.append(1)
                estimated_grad.append(all_noise[i])
            else:
                z.append(-1)
                estimated_grad.append(-all_noise[i])

        grad = -(1 / q_max) * sum(estimated_grad)
        grad_f = torch.from_numpy(grad)[None, :, :, :]

        return grad_f.to(self.device), sum(z)

    def geoda(
        self,
        x_0: torch.Tensor,
        y_0: torch.Tensor,
        x_b: torch.Tensor,
        iteration: int,
        q_opt: list[int],
        sub_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        q_num = 0
        grad = torch.zeros_like(x_0).to(self.device)

        for i in range(iteration):
            grad_oi, _ = self.black_grad_batch(x_b, q_opt[i], y_0[0], sub_basis)
            q_num = q_num + q_opt[i]

            grad = grad_oi + grad
            x_adv, qs = self.go_to_boundary(x_0, y_0, grad)
            q_num = q_num + qs

            x_adv, bin_query = self.bin_search(x_0, y_0, x_adv, self.tol)
            q_num = q_num + bin_query

            x_b = x_adv

            # norm = self.distance(x_adv, x_0)
            # if norm < self.epsilon or q_num > self.max_queries:
            #     break
            if q_num > self.max_queries:
                break

        x_adv = torch.clamp(x_adv, 0, 1).detach()
        return x_adv, q_num, grad

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform GeoDA on a batch of images.

        N.B., although batch size > 1 is supported (attack can be executed), the
        perturbation magnitude will be incorrect.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        # Load the DCT subspaces basis
        if not hasattr(self, 'sub_basis'):
            if os.path.exists(self.sub_basis_path):
                sub_basis = np.load(self.sub_basis_path).astype(np.float32)
            else:
                os.makedirs(os.path.dirname(self.sub_basis_path), exist_ok=True)
                sub_basis = gen_2d_dct_sub_basis(
                    self.sub_dim, self.x_size, self.sub_basis_path
                ).astype(np.float32)
                print(f'Generated DCT sub-basis to `{self.sub_basis_path}`')
            self.sub_basis = torch.from_numpy(sub_basis).to(self.device)

        # Determine ys with the actual model prediction
        ys = self.predict(x)

        # Starting with a randomized perturbation
        x_random, query_random_1 = self.find_random_adversarial(x, ys)

        # Binary search
        x_boundary, query_binsearch_2 = self.bin_search(x, ys, x_random, self.tol)
        x_b = x_boundary

        query_rnd = query_binsearch_2 + query_random_1

        # Determine the optimal query iteration
        iteration = round(self.max_queries / 500)
        q_opt_it = int(self.max_queries - (iteration) * 25)
        q_opt_iter, iterate = self.opt_query_iteration(q_opt_it, iteration, self.mu)
        q_opt_it = int(self.max_queries - (iterate) * 25)
        q_opt_iter, iterate = self.opt_query_iteration(q_opt_it, iteration, self.mu)

        # Perform the GeoDA attack
        adv, query_o, _ = self.geoda(x, ys, x_b, iterate, q_opt_iter, self.sub_basis)

        # Number of queries
        nb_queries = query_o + query_rnd  # noqa: F841

        return adv

    def predict(self, xs: torch.Tensor) -> torch.Tensor:
        xs = xs.to(self.device)
        out: torch.Tensor = self.model(self.normalize(xs))
        return out.argmax(dim=1).detach()

    def distance(
        self, x_adv: torch.Tensor, x: torch.Tensor | None = None
    ) -> int | float:
        diff = (
            x_adv.reshape(x_adv.size(0), -1)
            if x is None
            else (x_adv - x).reshape(x.size(0), -1)
        )

        if self.p == 'l2':
            out = torch.sqrt(torch.sum(diff * diff)).item()
        elif self.p == 'linf':
            out = torch.sum(torch.max(torch.abs(diff), 1)[0]).item()

        return out

    def is_adv(self, x_adv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = self.predict(x_adv)
        return y_pred != y


if __name__ == '__main__':
    from torchattack.evaluate import run_attack

    run_attack(GeoDA, {'epsilon': 4, 'p': 'l2', 'max_queries': 4000}, batch_size=2)
