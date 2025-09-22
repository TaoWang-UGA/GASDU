###############################################################################
# gasdu.py — Sparse-gradient linear layer for GASDU (delta-param mode)
#
# Behavior (current-step mask):
#   • 'weight' is a frozen dense matrix (requires_grad=False)
#   • A trainable vector 'delta_vals' (capacity ≥ k) holds per-step updates
#   • Forward: y = x W^T + Σ_j (delta_j * x[:, col_j]) added into out[:, row_j]
#   • Backward:
#       - Computes grads for input, bias, and delta_vals (NO grad for weight)
#       - If refresh is due, computes Top-K from *current* (grad_out, x) and
#         INSTALLS the mask immediately; grad_delta uses this fresh mask
#   • After optimizer.step(), call commit_and_swap_mask() to fold delta into
#     the frozen base and zero delta (no mask swap needed)
#
# Notes:
#   - Uses chunked index_add_ to avoid materializing [N,K] temporaries
#   - Keeps bf16 where safe; small reductions in fp32 for stability
#   - Optional cross-module D2H bucketing to coalesce small CPU logs
###############################################################################

from __future__ import annotations
import math
import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────── helpers ──────────────────────────────────── #
def _low_priority_stream() -> Optional[torch.cuda.Stream]:
    """Create a valid low-priority CUDA stream (or None if CUDA not available)."""
    if not torch.cuda.is_available():
        return None
    try:
        least, _greatest = torch.cuda.get_stream_priority_range()
        return torch.cuda.Stream(priority=least)  # numerically larger = lower priority
    except Exception:
        return torch.cuda.Stream()


# ────────────────────────────── Bucket Manager ───────────────────────────── #
class _GradBucket:
    """Global bucket to coalesce D2H copies across modules.

    Enable by setting env var GASDU_BUCKET_BYTES (e.g., 4000000).
    Call gasdu_flush_buckets() once per batch to force a flush at safe points.
    """
    def __init__(self, threshold_bytes: int = 0):
        self.threshold_bytes = int(threshold_bytes)
        self._gpu_tensors: List[torch.Tensor] = []
        self._cpu_targets: List[torch.Tensor] = []
        self._total_bytes: int = 0
        self._d2h_stream = _low_priority_stream()

    def enqueue(self, src_gpu_vec: torch.Tensor, dst_cpu_vec: torch.Tensor):
        assert src_gpu_vec.is_cuda and not dst_cpu_vec.is_cuda
        self._gpu_tensors.append(src_gpu_vec)
        self._cpu_targets.append(dst_cpu_vec)
        self._total_bytes += src_gpu_vec.numel() * src_gpu_vec.element_size()
        if self.threshold_bytes > 0 and self._total_bytes >= self.threshold_bytes:
            self.flush()

    def flush(self):
        if not self._gpu_tensors:
            return
        stream = self._d2h_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            big_gpu = torch.cat(self._gpu_tensors, dim=0).contiguous()
            big_cpu = torch.empty(
                big_gpu.numel(),
                dtype=self._cpu_targets[0].dtype,
                device="cpu",
                pin_memory=True
            )
            big_cpu.copy_(big_gpu, non_blocking=True)
        if torch.cuda.is_available():
            stream.synchronize()

        offset = 0
        for dst in self._cpu_targets:
            n = dst.numel()
            dst.copy_(big_cpu[offset:offset + n], non_blocking=False)
            offset += n

        self._gpu_tensors.clear()
        self._cpu_targets.clear()
        self._total_bytes = 0


_BUCKET = _GradBucket(int(os.getenv("GASDU_BUCKET_BYTES", "0"))) if torch.cuda.is_available() else _GradBucket(0)

def gasdu_flush_buckets():
    if _BUCKET is not None:
        _BUCKET.flush()


# ───────────────────────────── Streaming Top-K (exact) ───────────────────── #
def _topk_indices_streaming(grad_out_2d: torch.Tensor,
                            x_2d: torch.Tensor,
                            k: int,
                            tile_o: Optional[int] = None,
                            tile_i: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute indices of the global Top-K by |grad_out^T @ x_2d| WITHOUT materializing [O,I].

    Shapes:
      grad_out_2d: [N, O]
      x_2d:        [N, I]
    Returns:
      (row_idx[k], col_idx[k]) on the same device as inputs.
    """
    assert grad_out_2d.dim() == 2 and x_2d.dim() == 2
    device = grad_out_2d.device
    N, O = grad_out_2d.shape
    _, I = x_2d.shape
    k = int(min(max(0, k), O * I))
    if k == 0:
        z = torch.empty(0, dtype=torch.long, device=device)
        return z, z

    tile_o = int(os.getenv("GASDU_TILE_O", "1024")) if tile_o is None else int(tile_o)
    tile_i = int(os.getenv("GASDU_TILE_I", "1024")) if tile_i is None else int(tile_i)
    tile_o = max(1, min(tile_o, O))
    tile_i = max(1, min(tile_i, I))

    cand_vals = torch.empty(0, dtype=torch.float32, device=device)
    cand_rows = torch.empty(0, dtype=torch.long,    device=device)
    cand_cols = torch.empty(0, dtype=torch.long,    device=device)

    GO = grad_out_2d.to(torch.bfloat16)  # [N,O]
    X  = x_2d.to(torch.bfloat16)         # [N,I]

    POOL_MULT = int(os.getenv("GASDU_TOPK_POOL_MULT", "8"))
    POOL_CAP  = max(k, POOL_MULT * k)

    for o0 in range(0, O, tile_o):
        To = min(tile_o, O - o0)
        go = GO[:, o0:o0+To]                            # [N, To]
        for i0 in range(0, I, tile_i):
            Ti = min(tile_i, I - i0)
            xx = X[:, i0:i0+Ti]                         # [N, Ti]
            block = go.t().mm(xx)                       # [To, Ti], bf16
            block = block.abs().to(torch.float32)

            local_k = min(k, block.numel())
            v, idx = torch.topk(block.view(-1), local_k, largest=True, sorted=False)
            r = idx // Ti
            c = idx %  Ti

            cand_vals = torch.cat([cand_vals, v], dim=0)
            cand_rows = torch.cat([cand_rows, r.to(torch.long) + o0], dim=0)
            cand_cols = torch.cat([cand_cols, c.to(torch.long) + i0], dim=0)

            if cand_vals.numel() > POOL_CAP:
                vv, ii = torch.topk(cand_vals, k, largest=True, sorted=False)
                cand_vals, cand_rows, cand_cols = vv, cand_rows[ii], cand_cols[ii]

    if cand_vals.numel() > k:
        vv, ii = torch.topk(cand_vals, k, largest=True, sorted=True)
        cand_vals, cand_rows, cand_cols = vv, cand_rows[ii], cand_cols[ii]

    return cand_rows, cand_cols


# ───────────────────── Custom autograd Function ─────────────────────────── #
class _SparseUpdateLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor],
                delta_vals: torch.Tensor,
                module_ref: "SparseUpdateLinear"):
        """
        out = input @ weight^T + Σ_j (delta_j * input[:, col_j]) added into out[:, row_j]
        This is done in chunks to limit peak memory (no N×K indices).
        """
        mod: "SparseUpdateLinear" = module_ref
        ctx.save_for_backward(input.to(torch.bfloat16))  # halve saved-act memory
        ctx.has_bias = bias is not None
        ctx.module = mod
        ctx.dtype = input.dtype

        # 2-D or 3-D input
        if input.dim() == 2:
            N, I = input.shape
            O = weight.size(0)
            out = F.linear(input, weight, None if bias is None else bias)
        elif input.dim() == 3:
            B, S, I = input.shape
            N = B * S
            O = weight.size(0)
            out = F.linear(input, weight, None if bias is None else bias)
        else:
            raise ValueError("SparseUpdateLinear expects 2- or 3-D inputs.")

        # Delta-path contribution (chunked) — SKIP if deltas are zero
        k_act = min(int(mod.k), mod.row_idx.numel(), mod.col_idx.numel(), delta_vals.numel())
        if k_act > 0 and torch.any(delta_vals[:k_act] != 0):
            x2d = input.reshape(N, I)
            out2d = out.reshape(N, O)

            CH = int(os.getenv("GASDU_K_CHUNK", "4096"))
            for j0 in range(0, k_act, CH):
                j1 = min(k_act, j0 + CH)
                cols = mod.col_idx[j0:j1]                         # [j]
                rows = mod.row_idx[j0:j1]                         # [j]
                vals_scale = delta_vals[j0:j1].to(x2d.dtype)      # [j]
                x_sel = x2d.index_select(1, cols)                 # [N, j]
                out2d.index_add_(1, rows, x_sel * vals_scale)     # add into selected output cols

            out = out2d.view_as(out)

        return out.to(ctx.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        mod: "SparseUpdateLinear" = ctx.module
        weight = mod.weight
        bias   = mod.bias

        (x_saved_bf16,) = ctx.saved_tensors

        if grad_output.dim() == 3:
            B, S, O = grad_output.shape
            grad_out = grad_output.reshape(-1, O).to(torch.bfloat16)     # [N, O]
            x_2d     = x_saved_bf16.reshape(-1, x_saved_bf16.size(-1))  # [N, I]
        else:
            grad_out = grad_output.to(torch.bfloat16)                    # [N, O]
            x_2d     = x_saved_bf16                                     # [N, I]

        N, I = x_2d.shape

        # dense base path gradient wrt input
        grad_input = grad_out.mm(weight.to(torch.bfloat16)).to(ctx.dtype)

        full_l2 = None
        if getattr(mod, "track_grad_norm_prop", False):
            _full_grad = grad_out.t().mm(x_2d).to(torch.float32)  # [O, I]
            full_l2 = torch.norm(_full_grad, p=2)

        # refresh policy
        if mod.row_idx.numel() == 0:
            refresh = True
        elif mod.mask_mode == "fixed_topk":
            refresh = False
        elif mod.mask_mode == "topk_refresh_each_step":
            refresh = True
        else:  # "topk" periodic
            refresh = (mod._step % max(1, mod.full_grad_every) == 0)

        if refresh:
            next_row, next_col = _topk_indices_streaming(grad_out, x_2d, int(mod.k))
            mod.row_idx = next_row.to(weight.device, non_blocking=True)
            mod.col_idx = next_col.to(weight.device, non_blocking=True)
            if mod.row_idx_cpu is not None:
                mod.row_idx_cpu.resize_(int(mod.k))
                mod.col_idx_cpu.resize_(int(mod.k))
                mod.row_idx_cpu.copy_(mod.row_idx.detach().to(mod.row_idx_cpu.device), non_blocking=True)
                mod.col_idx_cpu.copy_(mod.col_idx.detach().to(mod.col_idx_cpu.device), non_blocking=True)

        eff_rows = mod.row_idx
        eff_cols = mod.col_idx
        k_act = min(int(mod.k), eff_rows.numel(), eff_cols.numel(), mod.delta_vals.numel())

        if k_act > 0 and torch.any(mod.delta_vals[:k_act] != 0):
            CH = int(os.getenv("GASDU_K_CHUNK", "4096"))
            for j0 in range(0, k_act, CH):
                j1 = min(k_act, j0 + CH)
                rows = eff_rows[j0:j1]
                cols = eff_cols[j0:j1]
                scales = mod.delta_vals[j0:j1].to(grad_out.dtype)        # [j]
                gi_add = grad_out[:, rows] * scales                      # [N, j]
                grad_input.index_add_(1, cols, gi_add.to(grad_input.dtype))

        if grad_output.dim() == 3:
            grad_input = grad_input.view(B, S, -1)

        if k_act > 0:
            part_grad = torch.empty(k_act, dtype=torch.float32, device=grad_out.device)
            CH = int(os.getenv("GASDU_K_CHUNK", "4096"))
            for j0 in range(0, k_act, CH):
                j1 = min(k_act, j0 + CH)
                go_sel = grad_out[:, eff_rows[j0:j1]]     # [N, j]
                x_sel  = x_2d[:,  eff_cols[j0:j1]]        # [N, j]
                part_grad[j0:j1] = (go_sel * x_sel).sum(dim=0, dtype=torch.float32)

            grad_delta = torch.zeros_like(mod.delta_vals)
            grad_delta[:k_act] = part_grad.to(grad_delta.dtype)

            if full_l2 is not None:
                masked_l2 = torch.norm(part_grad, p=2)
                mod.latest_ratio = (masked_l2 / (full_l2 + 1e-12)).item()
        else:
            grad_delta = torch.zeros_like(mod.delta_vals)
            if full_l2 is not None:
                mod.latest_ratio = 0.0

        grad_bias = (grad_out.sum(dim=0).to(bias.dtype) if bias is not None else None)

        if getattr(mod, "vals_cpu", None) is not None and k_act > 0:
            vals_gpu_cast = part_grad.to(torch.bfloat16).contiguous()
            if mod.vals_cpu.numel() != vals_gpu_cast.numel():
                mod.vals_cpu = torch.empty(vals_gpu_cast.numel(), dtype=torch.bfloat16, device="cpu").pin_memory()
            if mod.bucket_bytes > 0 and _BUCKET is not None and _BUCKET.threshold_bytes > 0:
                _BUCKET.enqueue(vals_gpu_cast, mod.vals_cpu)
            else:
                s = mod.d2h_stream or torch.cuda.current_stream()
                with torch.cuda.stream(s):
                    mod.vals_cpu.copy_(vals_gpu_cast, non_blocking=True)

        mod._step += 1

        return (grad_input, None, grad_bias, grad_delta, None)


# ─────────────────────── nn.Module wrapper ──────────────────────────────── #
class SparseUpdateLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with sparse-gradient updates (delta-param mode).

    • 'weight' is a frozen dense matrix (requires_grad=False).
    • A trainable vector 'delta_vals' (capacity ≥ k) holds per-step updates at active indices.
    • Forward adds delta contribution without building dense updates.
    • Backward computes grads for input, bias, and delta_vals only.
    • Mask (row_idx/col_idx) uses the **current step’s gradients**:
        - We *compute and install* a fresh mask during this step’s backward if refresh rules allow,
          and we **use that mask immediately** to assemble grad_delta for this step.
    • Optional cross-module D2H bucketing remains available for logging small k-vectors.

    Env tunables:
      GASDU_TILE_O, GASDU_TILE_I      – Top-K tile sizes (default 1024)
      GASDU_TOPK_POOL_MULT            – candidate pool multiplier (default 8)
      GASDU_K_CHUNK                   – per-chunk K used in chunked index_add_ (default 4096)
      GASDU_BUCKET_BYTES              – D2H bucket threshold (0 disables)
      GASDU_LOG_VALUES                – if "1", allocate pinned value buffer for logging
    """

    def __init__(self, *,
                 in_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 k: int = 100,
                 old_linear: Optional[nn.Linear] = None,
                 use_dynamic_k: bool = False,
                 k_initial: Optional[int] = None,
                 k_final: Optional[int] = None,
                 num_decay_steps: int = 10_000,
                 mask_mode: str = "topk",
                 track_grad_norm_prop: bool = False,
                 full_grad_every: int = 50,
                 bucket_bytes: Optional[int] = None):
        super().__init__()

        # infer shape from old layer if provided
        if old_linear is not None:
            out_features, in_features = old_linear.weight.shape
        if in_features is None or out_features is None:
            raise ValueError("Provide in/out features or pass old_linear.")

        if mask_mode not in ("topk", "fixed_topk", "topk_refresh_each_step"):
            raise ValueError("mask_mode must be 'topk', 'fixed_topk', or 'topk_refresh_each_step'.")

        self.in_features  = int(in_features)
        self.out_features = int(out_features)
        self.k            = int(k)
        self.mask_mode    = mask_mode
        self.track_grad_norm_prop = bool(track_grad_norm_prop)
        self.full_grad_every      = int(full_grad_every)

        # bucket threshold
        if bucket_bytes is not None:
            self.bucket_bytes = int(bucket_bytes)
        else:
            self.bucket_bytes = int(os.getenv("GASDU_BUCKET_BYTES", "0")) if torch.cuda.is_available() else 0

        # dynamic-k schedule
        self.use_dynamic_k = bool(use_dynamic_k)
        if self.use_dynamic_k:
            if k_initial is None or k_final is None:
                raise ValueError("k_initial and k_final required when use_dynamic_k=True.")
            self.k_initial, self.k_final = int(k_initial), int(k_final)
            self.num_decay_steps = int(num_decay_steps)
            self.decay_rate = (self.k_final / max(1, self.k_initial)) ** (1.0 / max(1, self.num_decay_steps))
            self.current_step = 0
            self._delta_capacity = max(self.k_initial, self.k_final)  # capacity stays fixed
        else:
            self.k_initial, self.k_final = int(k), int(k)
            self.num_decay_steps = 0
            self.decay_rate = 1.0
            self.current_step = 0
            self._delta_capacity = int(k)

        # parameters --------------------------------------------------------
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight.requires_grad_(False)

        if old_linear is not None and getattr(old_linear, "bias", None) is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(self.out_features))

        if old_linear is not None:
            with torch.no_grad():
                self.weight.copy_(old_linear.weight)
                if getattr(old_linear, "bias", None) is not None:
                    self.bias.copy_(old_linear.bias)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        # Trainable delta vector (capacity ≥ k); dtype matches weight
        self.delta_vals = nn.Parameter(torch.zeros(self._delta_capacity, dtype=self.weight.dtype))

        # GPU mask indices (initialized empty; filled on first refresh)
        self.register_buffer("row_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("col_idx", torch.empty(0, dtype=torch.long), persistent=False)

        # Optional CPU mirrors for logging (allocated by user if desired)
        self.row_idx_cpu: Optional[torch.Tensor] = None
        self.col_idx_cpu: Optional[torch.Tensor] = None

        # Host value buffer in bfloat16 (pinned, sized lazily to k)
        self.vals_cpu: Optional[torch.Tensor] = None

        # Low-priority stream for D2H copies
        self.d2h_stream = _low_priority_stream()

        # step counter
        self._step = 0

        # diagnostics
        self.latest_ratio: Optional[float] = None  # masked/full L2 on current step

    # ───────── helper for dynamic-k ───────── #
    def _update_k(self):
        if self.use_dynamic_k and self.current_step < self.num_decay_steps:
            new_k = max(self.k_final, int(round(self.k_initial * (self.decay_rate ** self.current_step))))
            self.k = int(new_k)
            self.current_step += 1
        self.k = min(self.k, self.delta_vals.numel())

    # ───────────────── forward ────────────── #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_k()
        if self.vals_cpu is None and int(self.k) > 0 and os.getenv("GASDU_LOG_VALUES", "0") == "1":
            self.vals_cpu = torch.empty(int(self.k), dtype=torch.bfloat16, device="cpu").pin_memory()
        return _SparseUpdateLinearFn.apply(x, self.weight, self.bias, self.delta_vals, self)

    @torch.no_grad()
    def commit_and_swap_mask(self):
        """
        Fold the current delta into the base weight and zero delta.
        (No mask swap is needed; the mask for the next step will be refreshed
         during that step’s backward if rules say so.)
        """
        k_act = min(int(self.k), self.row_idx.numel(), self.col_idx.numel(), self.delta_vals.numel())
        if k_act > 0:
            r = self.row_idx[:k_act]
            c = self.col_idx[:k_act]
            self.weight.data[r, c] += self.delta_vals.data[:k_act].to(self.weight.dtype)
            self.delta_vals.data.zero_()


__all__ = ["SparseUpdateLinear", "gasdu_flush_buckets"]
