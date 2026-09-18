from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn


def _sanitize_omp_num_threads() -> None:
    value = os.environ.get("OMP_NUM_THREADS")
    if value is None:
        return
    try:
        if int(value) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        os.environ["OMP_NUM_THREADS"] = "1"


_sanitize_omp_num_threads()


def _broadcast_module_state(module: nn.Module, src: int = 0) -> None:
    for tensor in module.state_dict().values():
        dist.broadcast(tensor, src=src)


class _GradientReducer:
    def __init__(self, params: Iterable[nn.Parameter]):
        self.world_size = dist.get_world_size()
        self.params: list[nn.Parameter] = []

        seen = set()
        for param in params:
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            self.params.append(param)

    def finish(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(self.world_size)


class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        _broadcast_module_state(self.module)
        self._reducer = _GradientReducer(self.module.parameters())

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        self._reducer.finish()


def _apply_mixed_precision_hooks(module: nn.Module, compute_dtype: torch.dtype) -> None:
    from cs336_basics.model import Embedding, Linear

    for mod in module.modules():
        if not isinstance(mod, (Linear, Embedding)):
            continue

        def make_fwd_pre(dt):
            def hook(m, _inp):
                m._saved_fp32 = m.weight.data
                m.weight.data = m.weight.data.to(dt)

            return hook

        def make_fwd_post():
            def hook(m, _inp, _out):
                m.weight.data = m._saved_fp32
                del m._saved_fp32
                m.weight.grad = None

            return hook

        mod.register_forward_pre_hook(make_fwd_pre(compute_dtype))
        mod.register_forward_hook(make_fwd_post())

        if isinstance(mod, Linear):

            def make_bwd_pre(dt):
                def hook(m, _grad_output):
                    m._saved_fp32_bwd = m.weight.data
                    m.weight.data = m.weight.data.to(dt)
                    m.weight.grad = None

                return hook

            mod.register_full_backward_pre_hook(make_bwd_pre(compute_dtype))

        def make_grad_hook(m, is_linear):
            def hook(param):
                if is_linear and hasattr(m, "_saved_fp32_bwd"):
                    m.weight.data = m._saved_fp32_bwd
                    del m._saved_fp32_bwd
                if param.grad is not None:
                    param.grad = param.grad.to(torch.float32)

            return hook

        mod.weight.register_post_accumulate_grad_hook(make_grad_hook(mod, isinstance(mod, Linear)))


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        _broadcast_module_state(self.module)
        if self.compute_dtype is not None:
            _apply_mixed_precision_hooks(self.module, self.compute_dtype)
        self._reducer = _GradientReducer(self.module.parameters())

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        self._reducer.finish()

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in self.module.named_parameters()}


def get_sharded_optimizer(
    params,
    optimizer_cls: type[torch.optim.Optimizer],
    **kwargs,
) -> torch.optim.Optimizer:
    return optimizer_cls(params, **kwargs)
