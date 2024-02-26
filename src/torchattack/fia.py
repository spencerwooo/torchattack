from typing import Callable

import torch
import torch.nn as nn

from torchattack.base import Attack


class FIA(Attack):
    """The FIA (Feature Importance-aware) Attack.

    From the paper 'Feature Importance-aware Transferable Adversarial Attacks'.
    https://arxiv.org/abs/2107.14185

    Args:
        model: The model to attack.
        normalize: A transform to normalize images.
        device: Device to use for tensors. Defaults to cuda if available.
        eps: The maximum perturbation. Defaults to 8/255.
        steps: Number of steps. Defaults to 10.
        alpha: Step size, `eps / steps` if None. Defaults to None.
        decay: Decay factor for the momentum term. Defaults to 1.0.
        num_ens: Number of aggregate gradients. Defaults to 30.
        feature_layer: The layer to compute feature importance. Defaults to "layer4".
        drop_rate: Dropout rate for random pixels. Defaults to 0.3.
        clip_min: Minimum value for clipping. Defaults to 0.0.
        clip_max: Maximum value for clipping. Defaults to 1.0.
        targeted: Targeted attack if True. Defaults to False.
    """

    def __init__(
        self,
        model: nn.Module,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None,
        device: torch.device | None = None,
        eps: float = 8 / 255,
        steps: int = 10,
        alpha: float | None = None,
        decay: float = 1.0,
        num_ens: int = 30,
        feature_layer: str = 'layer4',
        drop_rate: float = 0.3,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ) -> None:
        super().__init__(normalize, device)
        self.model = model
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.decay = decay
        self.num_ens = num_ens
        self.feature_layer = self.find_layer(feature_layer)
        self.drop_rate = drop_rate
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        # self.lossfn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform FIA on a batch of images.

        Args:
            x: A batch of images. Shape: (N, C, H, W).
            y: A batch of labels. Shape: (N).

        Returns:
            The perturbed images if successful. Shape: (N, C, H, W).
        """

        delta = torch.zeros_like(x, requires_grad=True)

        # If alpha is not given, set to eps / steps
        if self.alpha is None:
            self.alpha = self.eps / self.steps

        h = self.feature_layer.register_forward_hook(self.__forward_hook)
        h2 = self.feature_layer.register_full_backward_hook(self.__backward_hook)

        # Gradient aggregation on ensembles
        agg_grad = 0
        for _ in range(self.num_ens):
            x_dropped = self.drop_pixels(x)
            output_random = self.model(self.normalize(x_dropped))
            output_random = torch.softmax(output_random, dim=1)
            loss = 0
            for batch_i in range(x.shape[0]):
                loss += output_random[batch_i][y[batch_i]]
            self.model.zero_grad()
            loss.backward()
            agg_grad += self.mid_grad[0].detach()
        for batch_i in range(x.shape[0]):
            agg_grad[batch_i] /= agg_grad[batch_i].norm(p=2)
        h2.remove()

        # Perform FIA
        for _ in range(self.steps):
            # Pass through the model
            _ = self.model(self.normalize(x + delta))

            # Hooks are updated during forward pass
            loss = (self.mid_output * agg_grad).sum()

            self.model.zero_grad()
            grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]

            # Update delta
            delta.data = delta.data - self.alpha * grad.sign()
            delta.data = torch.clamp(delta.data, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta.data, self.clip_min, self.clip_max) - x

        h.remove()
        return x + delta

    def drop_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop pixels from the input image.

        Args:
            x: A batch of images. Shape: (N, C, H, W).

        Returns:
            A batch of images with randomly dropped pixels.
        """

        x_dropped = torch.zeros_like(x)
        x_dropped.copy_(x).detach()
        x_dropped.requires_grad = True

        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.drop_rate))
        x_dropped = x_dropped * mask

        return x_dropped

    def find_layer(self, feature_layer_name) -> nn.Module:
        """Find the layer to compute feature importance.

        Returns:
            The layer to compute feature importance.
        """

        parser = feature_layer_name.split(' ')
        m = self.model

        for layer in parser:
            if layer in m._modules:
                m = m._modules[layer]
                break
            else:
                raise ValueError(f'Layer {layer} not found in the model.')

        return m

    def __forward_hook(self, m: nn.Module, i: torch.Tensor, o: torch.Tensor):
        self.mid_output = o

    def __backward_hook(self, m: nn.Module, i: torch.Tensor, o: torch.Tensor):
        self.mid_grad = o


if __name__ == '__main__':
    from torchattack.utils import run_attack

    run_attack(FIA, {'eps': 8 / 255, 'steps': 10, 'feature_layer': 'layer2'})
