from __future__ import annotations

import torch
import torch.nn as nn


class _CondEmbeddingFlow(nn.Module):
    """Wrapper linking an embedding network and a conditional flow.

    The embedding_net takes the conditioning data and returns a fixed-size
    representation used as context for the flow.

    ``amp_encoder`` (default False, set by build_model from config.ml_perf.amp) runs the
    embedding_net (the big map CNN) under bf16 autocast for tensor-core speed + ~½ activation
    memory, then casts its output back to fp32 so the flow's rational-quadratic spline stays
    fp32. Whole-forward bf16 (Lightning ``precision='bf16-mixed'``) instead CRASHES the nflows
    spline at its ``index_put`` (BFloat16 vs Float), so the autocast MUST be scoped here.
    """

    amp_encoder: bool = False

    def __init__(self, embedding_net: nn.Module, flow: nn.Module):
        super().__init__()
        self.embedding_net = embedding_net if embedding_net is not None else nn.Identity()
        self.flow = flow

    def _embed(self, y):
        if self.amp_encoder:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.embedding_net(y)
            if isinstance(out, torch.Tensor):
                return out.float()
            if isinstance(out, (tuple, list)):
                return type(out)(o.float() if isinstance(o, torch.Tensor) else o for o in out)
            return out
        return self.embedding_net(y)

    def encode(self, y):
        return self._embed(y)

    def get_representation(self, y):
        return self.embedding_net.get_representation(y)

    def log_prob(self, x, y):
        y_emb = self._embed(y)
        x = x.unsqueeze(0)
        return self.flow.log_prob(x, y_emb)

    def latent_log_prob(self, x, y_emb):
        x = x.unsqueeze(0)
        return self.flow.log_prob(x, y_emb)

    def sample(self, shape, y, **kwargs):
        y_emb = self._embed(y)
        return self.flow.sample(shape, y_emb, **kwargs)

    def sample_batched(self, shape, y, **kwargs):
        y_emb = self._embed(y)
        return self.flow.sample_batched(shape, y_emb, **kwargs)


class MultipleFlow(nn.Module):
    """Container for an ensemble of flow models with a single-flow-like API."""

    def __init__(self, flows: list[nn.Module]):
        super().__init__()
        if len(flows) == 0:
            raise ValueError("MultipleFlow requires at least one flow.")
        self.flows = nn.ModuleList(flows)

    def log_prob(self, x, y, **kwargs):
        log_probs = [flow.log_prob(x, y, **kwargs) for flow in self.flows]
        stacked = torch.stack(log_probs, dim=0)
        return stacked.mean(dim=0)

    def sample(self, shape, y, **kwargs):
        samples = [flow.sample(shape, y, **kwargs) for flow in self.flows]
        samples = torch.stack(samples, dim=0)
        return samples.mean(dim=0)

    def sample_batched(self, shape, y, **kwargs):
        samples = []
        for flow in self.flows:
            if hasattr(flow, "sample_batched"):
                samples.append(flow.sample_batched(shape, y, **kwargs))
            else:
                samples.append(flow.sample(shape, y, **kwargs))
        samples = torch.stack(samples, dim=0)
        return samples.mean(dim=0)
