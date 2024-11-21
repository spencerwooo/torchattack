import torch


class FoolingRateMetric:
    """Fooling rate metric tracker."""

    def __init__(self) -> None:
        self.total_count = torch.tensor(0)
        self.clean_count = torch.tensor(0)
        self.adv_count = torch.tensor(0)

    def update(
        self, labels: torch.Tensor, clean_logits: torch.Tensor, adv_logits: torch.Tensor
    ) -> None:
        """Update metric tracker during attack progress.

        Args:
            labels: Ground truth labels.
            clean_logits: Prediction logits for clean samples.
            adv_logits: Prediction logits for adversarial samples.
        """

        self.total_count += labels.numel()
        self.clean_count += (clean_logits.argmax(dim=1) == labels).sum().item()
        self.adv_count += (adv_logits.argmax(dim=1) == labels).sum().item()

    def compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the fooling rate and related metrics.

        Returns:
            A tuple of torch.Tensors containing the clean accuracy, adversarial
            accuracy, and fooling rate computed, respectively.
        """
        return (
            self.clean_count / self.total_count,
            self.adv_count / self.total_count,
            (self.clean_count - self.adv_count) / self.clean_count,
        )
