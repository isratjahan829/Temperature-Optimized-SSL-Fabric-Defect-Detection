"""Loss functions: NT-Xent, BYOL cosine loss, SwAV swapped prediction, focal loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    r"""Normalised temperature-scaled cross entropy (SimCLR).

    For a batch of ``N`` images the two views are stacked into ``2N``
    embeddings; each sample's positive is its counterpart view and the
    remaining ``2N - 2`` embeddings act as negatives:

    .. math::
        \ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/T)}
                                {\sum_{k \neq i}\exp(\mathrm{sim}(z_i, z_k)/T)}

    The temperature ``T`` is the quantity the paper optimises (0.1/0.5/0.8).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if z_i.shape != z_j.shape:
            raise ValueError(f"view shapes differ: {tuple(z_i.shape)} vs {tuple(z_j.shape)}")
        batch_size = z_i.size(0)
        if batch_size < 2:
            raise ValueError("NT-Xent needs a batch size of at least 2")

        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        similarity = (z @ z.t()) / self.temperature

        # Mask self-similarity before the log-sum-exp over negatives.
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity = similarity.masked_fill(self_mask, float("-inf"))

        targets = torch.arange(2 * batch_size, device=z.device)
        targets = (targets + batch_size) % (2 * batch_size)
        return F.cross_entropy(similarity, targets)


class BYOLLoss(nn.Module):
    """Symmetric negative-cosine loss between online prediction and target."""

    def forward(
        self,
        p_online: torch.Tensor,
        z_target: torch.Tensor,
        p_online_2: torch.Tensor | None = None,
        z_target_2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self._single(p_online, z_target)
        if p_online_2 is not None and z_target_2 is not None:
            loss = 0.5 * (loss + self._single(p_online_2, z_target_2))
        return loss

    @staticmethod
    def _single(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target.detach(), dim=-1)
        return (2.0 - 2.0 * (prediction * target).sum(dim=-1)).mean()


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, epsilon: float = 0.05, iterations: int = 3) -> torch.Tensor:
    """Sinkhorn-Knopp normalisation producing equipartitioned soft assignments.

    ``scores`` has shape ``(batch, prototypes)``; the result is a doubly
    normalised assignment matrix of the same shape whose rows sum to 1.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    q = torch.exp(scores / epsilon).t()  # prototypes x batch
    q = q / torch.clamp(q.sum(), min=1e-12)
    num_prototypes, batch = q.shape
    for _ in range(max(1, iterations)):
        q = q / torch.clamp(q.sum(dim=1, keepdim=True), min=1e-12)
        q = q / num_prototypes
        q = q / torch.clamp(q.sum(dim=0, keepdim=True), min=1e-12)
        q = q / batch
    q = q * batch  # columns sum to 1
    return q.t()


class SwAVLoss(nn.Module):
    """Swapped-prediction loss: view ``a``'s codes supervise view ``b``'s scores."""

    def __init__(self, temperature: float = 0.1, epsilon: float = 0.05, sinkhorn_iterations: int = 3) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)

    def forward(self, scores: list[torch.Tensor]) -> torch.Tensor:
        """``scores`` holds prototype logits per crop; the first two are global."""
        if len(scores) < 2:
            raise ValueError("SwAV needs at least two crops")
        num_codes = min(2, len(scores))
        total = scores[0].new_zeros(())
        pairs = 0
        for code_view in range(num_codes):
            with torch.no_grad():
                code = sinkhorn(
                    scores[code_view].detach().float(),
                    epsilon=self.epsilon,
                    iterations=self.sinkhorn_iterations,
                )
            for other in range(len(scores)):
                if other == code_view:
                    continue
                log_p = F.log_softmax(scores[other] / self.temperature, dim=1)
                total = total - torch.mean(torch.sum(code * log_p, dim=1))
                pairs += 1
        return total / max(1, pairs)


class FocalLoss(nn.Module):
    r"""Multi-class focal loss, :math:`-\alpha_t (1 - p_t)^\gamma \log p_t`.

    Used in the explainability study to sharpen attention on hard,
    under-represented defect regions.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).gather(0, targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_ssl_loss(method: str, temperature: float, **kwargs) -> nn.Module:
    """Factory mapping a method name to its pretext loss."""
    method = method.lower()
    if method == "simclr":
        return NTXentLoss(temperature=temperature)
    if method == "byol":
        return BYOLLoss()
    if method == "swav":
        return SwAVLoss(
            temperature=temperature,
            epsilon=kwargs.get("epsilon", 0.05),
            sinkhorn_iterations=kwargs.get("sinkhorn_iterations", 3),
        )
    raise ValueError(f"unknown SSL method: {method!r}")
