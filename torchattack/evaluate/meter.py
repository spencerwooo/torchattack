import torch


class FoolingRateMeter:
    """Fooling rate metric tracker.

    Attributes:
        all_count: Total number of samples.
        cln_count: Number of correctly predicted clean samples.
        adv_count: Number of correctly predicted adversarial examples.
        targeted_count: Number of successfully attacked targeted adversarial examples.

    Args:
        targeted: Whether the current attack is targeted or not. Defaults to False.
    """

    def __init__(self, targeted: bool = False) -> None:
        self.targeted = targeted
        self.all_count = torch.tensor(0)
        self.cln_count = torch.tensor(0)
        self.adv_count = torch.tensor(0)
        self.targeted_count = torch.tensor(0)

    def update(
        self, labels: torch.Tensor, cln_logits: torch.Tensor, adv_logits: torch.Tensor
    ) -> None:
        """Update metric tracker during attack progress.

        Args:
            labels: Ground truth labels for non-targeted attacks, or a tuple of (ground
                truth labels, target labels) for targeted attacks.
            cln_logits: Prediction logits for clean samples.
            adv_logits: Prediction logits for adversarial examples.
        """

        if self.targeted:
            self.targeted_count += (adv_logits.argmax(dim=1) == labels[1]).sum().item()
            labels = labels[0]

        self.all_count += labels.numel()
        self.cln_count += (cln_logits.argmax(dim=1) == labels).sum().item()
        self.adv_count += (adv_logits.argmax(dim=1) == labels).sum().item()

    def compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the fooling rate and related metrics.

        Returns:
            A tuple of torch.Tensors containing the clean sample accuracy, adversarial
            example accuracy, and fooling rate (either non-targeted or targeted,
            depending on the attack) computed, respectively.
        """

        return (
            self.cln_count / self.all_count,
            self.adv_count / self.all_count,
            self.targeted_count / self.all_count
            if self.targeted
            else (self.cln_count - self.adv_count) / self.cln_count,
        )

    def reset(self) -> None:
        """Reset the metric tracker to initial state."""

        self.all_count = torch.tensor(0)
        self.cln_count = torch.tensor(0)
        self.adv_count = torch.tensor(0)
        self.targeted_count = torch.tensor(0)
