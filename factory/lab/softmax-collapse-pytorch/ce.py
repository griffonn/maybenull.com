"""Standard vs fixed cross-entropy in PyTorch.

fixed_ce is a direct port of the JAX cross_entropy in Barron et al., arXiv 2610.31642, Sec. 2:
exclude the target from the log-sum-exp, take the margin m, return softplus(-m).
FixedCE is the same math as a custom autograd.Function, so the backward pass is one fused expression.
"""
import torch
import torch.nn.functional as F


def margin(z, t):
    """Logit margin m = z_t - logsumexp_{j != t} z_j, computed in fp64."""
    z = z.double()
    zt = z.gather(-1, t[:, None])
    others = z.scatter(-1, t[:, None], float("-inf"))
    return (zt.squeeze(-1) - torch.logsumexp(others, -1))


def fixed_ce(z, t, reduction="mean"):
    zt = z.gather(-1, t[:, None])
    is_t = F.one_hot(t, z.shape[-1]).bool()
    others = torch.where(is_t, float("-inf"), z - zt)
    m = -torch.logsumexp(others, -1)
    loss = F.softplus(-m)
    return loss.mean() if reduction == "mean" else loss


class _FixedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, t):
        zt = z.gather(-1, t[:, None])
        d = z - zt                                   # z_j - z_t, exact 0 at target
        d.scatter_(-1, t[:, None], float("-inf"))    # drop the target from the sum
        lse = torch.logsumexp(d, -1, keepdim=True)  # = -m
        loss = F.softplus(lse)                       # log(1 + e^-m)
        ctx.save_for_backward(d, loss, t)
        return loss.mean()

    @staticmethod
    def backward(ctx, go):
        d, loss, t = ctx.saved_tensors
        g = torch.exp(d - loss)                      # non-target grads, target slot exp(-inf)=0
        g.scatter_(-1, t[:, None], -g.sum(-1, keepdim=True))  # target grad = -sum(others)
        return g * (go / d.shape[0]), None


def fixed_ce_fn(z, t):
    return _FixedCE.apply(z, t)
