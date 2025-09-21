"""
local_model_utilities.py — Lightning module & token-probe accuracy helpers.

SFT-aware:
  • Uses SftAdamW or SftSM3 when sft_enabled=True
  • Initializes SftSelector on train start (version-proof)
  • Calls selector.step() after each real optimizer step
  • Keeps GOLU commit-and-swap behavior

Also provides make_forced_labels_scheme() for GODLU_train_SpIEL.py.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple

import re
import torch
import torch.nn as nn
import lightning as L

# Optional GPU mem reporting (safe if NVML unavailable)
try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

# Tokenizer typing (not strictly required at runtime)
try:
    from transformers import PreTrainedTokenizer
except Exception:  # pragma: no cover
    PreTrainedTokenizer = object  # type: ignore

# GOLU utilities (commit after step; harmless for other methods)
from godlu import golu_flush_buckets, SparseUpdateLinear

# ---- Optimizer selection: DeepSpeed FusedAdam -> Apex FusedAdam -> torch.Adam
_OPTIM_BACKEND = "torch"
try:
    from deepspeed.ops.adam import FusedAdam as _FusedAdam  # type: ignore
    _OPTIM_BACKEND = "deepspeed"
except Exception:
    try:
        from apex.optimizers import FusedAdam as _FusedAdam  # type: ignore
        _OPTIM_BACKEND = "apex"
    except Exception:
        from torch.optim import Adam as _FusedAdam  # type: ignore
        _OPTIM_BACKEND = "torch"

# NEW: SFT optimizer + selector (AlanAnsell/peft)
try:
    from peft import SftAdamW, SftSM3, SftSelector, PeftModel, PeftConfig
    _HAS_SFT = True
except Exception:
    SftAdamW = SftSM3 = SftSelector = PeftModel = PeftConfig = None
    _HAS_SFT = False


# ───────────────────────────── GPU helpers ───────────────────────────── #
def report_gpu_usage(stage: str) -> None:
    if not _HAS_NVML:
        return
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        print(f"[GPU] {stage} — {m.used/1024**2:.1f} MB / {m.total/1024**2:.1f} MB")
    except Exception as exc:
        print(f"[GPU] Failed to query memory: {exc}")


# ───────────────────────────── Scheme helpers ───────────────────────────── #
@torch.no_grad()
def top1_choice(
    logits: torch.Tensor,            # [B, T, V]
    ctx_len: torch.Tensor,           # [B] index of FIRST answer token
    choice_ids: torch.Tensor         # [M] token ids of candidates
) -> torch.Tensor:                   # returns [B] index into 0..M-1
    idx = (ctx_len - 1).clamp_min(0)
    sel = logits[torch.arange(logits.size(0), device=logits.device), idx]   # [B, V]
    probs = sel.log_softmax(dim=-1)
    return probs[:, choice_ids].argmax(dim=-1)

def _first_id(tokenizer, s: str) -> Optional[int]:
    ids = tokenizer.encode(s, add_special_tokens=False)
    return ids[0] if ids else None

def _ids_union(tok, variants: List[str]) -> List[int]:
    out = []; seen = set()
    for s in variants:
        tid = _first_id(tok, s)
        if tid is not None and tid not in seen:
            seen.add(tid); out.append(tid)
    return out

def _scheme_letters(tok, K=5):
    id2cls = {}
    for i, L in enumerate(list("ABCDE")[:K]):
        variants = [f" {L}", f" {L.lower()}", L, L.lower()]
        bucket = _ids_union(tok, variants)
        for tid in bucket:
            id2cls.setdefault(tid, i)
    ids = list(id2cls.keys())
    l2c = [id2cls[tid] for tid in ids]
    return f"letters{K}", ids, l2c

def _scheme_numbers(tok, K=5):
    id2cls = {}
    for i in range(1, K+1):
        s = str(i)
        bucket = _ids_union(tok, [f" {s}", s])
        for tid in bucket:
            id2cls.setdefault(tid, i-1)
    ids = list(id2cls.keys())
    l2c = [id2cls[tid] for tid in ids]
    return f"numbers{K}", ids, l2c

def _scheme_prefixed(tok, prefix: str, K: int):
    id2cls = {}
    for i in range(1, K+1):
        variants = [
            f" {prefix}{i}", f" {prefix}{i}".lower(), f" {prefix.capitalize()}{i}",
            f"{prefix}{i}",  f"{prefix}{i}".lower(),   f"{prefix.capitalize()}{i}",
        ]
        bucket = _ids_union(tok, variants)
        for tid in bucket:
            id2cls.setdefault(tid, i-1)
    ids = list(id2cls.keys())
    l2c = [id2cls[tid] for tid in ids]
    return f"{prefix}{K}", ids, l2c

class _Scheme:
    def __init__(self, name: str, choice_ids: List[int], list_to_class: List[int], offset: int = 0):
        self.name = name
        self.choice_ids = list(choice_ids)
        self.list_to_class = list(list_to_class)
        self.offset = int(offset)
    def as_tensors(self, device):
        return (torch.tensor(self.choice_ids, dtype=torch.long, device=device),
                torch.tensor(self.list_to_class, dtype=torch.long, device=device))

def make_forced_labels_scheme(tok, labels: List[str]) -> _Scheme:
    tok_seqs: List[List[int]] = [tok.encode(lab, add_special_tokens=False) or [] for lab in labels]
    if any(len(s) == 0 for s in tok_seqs):
        return _Scheme("forced_labels", [], [], offset=0)
    K = len(tok_seqs)
    max_len = max(len(s) for s in tok_seqs)
    chosen_offset, chosen_ids = None, []
    for off in range(max_len):
        ids_at_off = []
        ok = True
        for s in tok_seqs:
            if len(s) <= off:
                ok = False; break
            ids_at_off.append(s[off])
        if not ok: continue
        if len(set(ids_at_off)) == K:
            chosen_offset = off; chosen_ids = ids_at_off; break
    if chosen_offset is None:
        chosen_offset = 0
        chosen_ids = [s[0] for s in tok_seqs]
    id2cls = {}
    for cls_idx, tid in enumerate(chosen_ids):
        if tid not in id2cls:
            id2cls[tid] = cls_idx
    choice_ids = list(id2cls.keys())
    list_to_class = [id2cls[tid] for tid in choice_ids]
    return _Scheme("forced_labels", choice_ids, list_to_class, offset=chosen_offset)

def _schemes_for_task(tok, task_name: str, expected_classes: Optional[int]) -> List[_Scheme]:
    if expected_classes is None:
        return [_Scheme("open-ended", [], [])]
    K = expected_classes
    out: List[_Scheme] = []
    for name, ids, l2c in [_scheme_letters(tok, K), _scheme_numbers(tok, K)]:
        out.append(_Scheme(name, ids, l2c))
    preferred_prefix = {
        "piqa": "solution",
        "siqa": "answer",
        "hellaswag": "ending",
        "winogrande": "option",
        "arc_e": "answer",
        "arc_c": "answer",
        "obqa": "answer",
        "aqua": "answer",
    }.get(task_name)
    prefixes = []
    if preferred_prefix: prefixes.append(preferred_prefix)
    for p in ["answer","solution","ending","option"]:
        if p not in prefixes: prefixes.append(p)
    for p in prefixes:
        name, ids, l2c = _scheme_prefixed(tok, p, K)
        out.append(_Scheme(name, ids, l2c))
    return out

def _detect_scheme_by_coverage(tokenizer, labels: torch.Tensor, task_name: str, expected_classes: Optional[int]) -> _Scheme:
    if expected_classes is None:
        return _Scheme("open-ended", [], [])
    device = labels.device
    B, T = labels.shape
    ctx_len = (labels != -100).float().argmax(dim=1).clamp_max(T - 1)
    row = torch.arange(B, device=device)
    gold_tok = labels[row, ctx_len]
    gold_tok = gold_tok[gold_tok != -100]
    if gold_tok.numel() == 0:
        return _Scheme("undetected", [], [])
    gold_set = set(int(x) for x in gold_tok.tolist())

    best: Tuple[int,int,_Scheme] = (0,0,_Scheme("undetected", [], []))
    for sc in _schemes_for_task(tokenizer, task_name, expected_classes):
        cov = sum(1 for g in gold_set if g in sc.choice_ids)
        classes_seen = set()
        for g in gold_set:
            if g in sc.choice_ids:
                idx = sc.choice_ids.index(g)
                classes_seen.add(sc.list_to_class[idx])
        ncls = len(classes_seen)
        if (cov, ncls) > (best[0], best[1]):
            best = (cov, ncls, sc)
        if ncls == expected_classes and cov >= max(2, expected_classes):
            return sc

    sc = best[2]
    if expected_classes is not None:
        distinct_classes = len(set(sc.list_to_class))
        if distinct_classes < 2 or distinct_classes < expected_classes:
            return _Scheme("undetected", [], [])
    return sc if sc.name != "undetected" else _Scheme("undetected", [], [])


# ───────────────────────────── Lightning Module ───────────────────────────── #
DATASET_CLASS_INFO = {
    # commonsense
    "boolq": 2, "piqa": 2, "siqa": 3, "hellaswag": 4, "winogrande": 2,
    "arc_e": 4, "arc_c": 4, "obqa": 4,
    # math
    "aqua": 5, "gsm8k": None, "svamp": None, "addsub": None, "singleeq": None, "multiarith": None,
    "mathqa": None, "mawps": None,
}

class CustomLightningModule(L.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 5e-5,
        task_name: Optional[str] = None,
        log_every_n: int = 50,
        log_grad_norm: bool = False,
        forced_scheme: Optional[_Scheme] = None,
        # NEW (SFT):
        sft_enabled: bool = False,
        sft_use_sm3: bool = False,
        sft_beta: float = 0.0,
        sft_total_train_steps: Optional[int] = None,
        sft_grad_accum: int = 1,
        sft_peft_config: Optional[object] = None,
        # NEW (math numeric EM):
        math_numeric_eval: bool = False,
        max_gen_new_tokens: int = 32,
    ):
        super().__init__()
        self.model         = model
        self.tokenizer     = tokenizer
        self.learning_rate = learning_rate
        self.task_name     = (task_name or "").lower().strip()
        self._log_every_n  = log_every_n
        self._log_grad_norm= log_grad_norm

        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "task_name": self.task_name,
            "log_every_n": log_every_n,
            "log_grad_norm": log_grad_norm,
            "math_numeric_eval": bool(math_numeric_eval),
            "max_gen_new_tokens": int(max_gen_new_tokens),
        })

        self.register_buffer("loss_ema", torch.tensor(0.0), persistent=False)
        self.ema_beta = 0.98

        self._scheme: Optional[_Scheme] = forced_scheme
        self.register_buffer("choice_id_list", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("list_to_class",  torch.empty(0, dtype=torch.long), persistent=False)

        self.register_buffer("epoch_correct", torch.tensor(0.0), persistent=False)
        self.register_buffer("epoch_total",   torch.tensor(0.0), persistent=False)

        # SFT
        self.sft_enabled = bool(sft_enabled) and _HAS_SFT
        self.sft_use_sm3 = bool(sft_use_sm3)
        self.sft_beta = float(sft_beta)
        self.sft_total_train_steps = int(sft_total_train_steps or 0)
        self.sft_grad_accum = int(max(1, sft_grad_accum))
        self.sft_peft_config = sft_peft_config
        self._sft_selector = None

        # Math numeric EM
        self.math_numeric_eval = bool(math_numeric_eval)
        self.max_gen_new_tokens = int(max_gen_new_tokens)
        self._num_re = re.compile(r'[-+]?\d+(?:\.\d+)?')
        self._em_eps = 1e-6

    # ------------- core forward -------------
    def on_fit_start(self):
        # Keep minimal; do NOT create the selector here
        if hasattr(self, "model"):
            self.model.train()
    
    def on_train_start(self):
        if hasattr(self, "model"):
            self.model.train()
    
        if self.sft_enabled and self._sft_selector is None and SftSelector is not None:
            # Prefer explicit total steps if provided
            if self.sft_total_train_steps and self.sft_total_train_steps > 0:
                est_steps = int(self.sft_total_train_steps)
            else:
                est_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
                if est_steps <= 0:
                    est_steps = int(getattr(self.trainer, "num_training_batches", 0)) * max(1, int(self.trainer.max_epochs or 1))
                if est_steps <= 0:
                    est_steps = 1  # last resort
    
            # ✅ Put it here, before SftSelector()
            self.sft_grad_accum = int(getattr(self.trainer, "accumulate_grad_batches", 1) or 1)
    
            opt = self.optimizers() if callable(getattr(self, "optimizers", None)) else self.trainer.optimizers[0]
            self._sft_selector = SftSelector(
                self.model,
                opt,
                self.sft_peft_config,
                est_steps,
                self.sft_grad_accum,            # now correct
                completed_steps=self.trainer.global_step,
            )
    
            if self.global_rank == 0:
                print(f"[INFO] SFT selector initialized: steps={est_steps}, "
                      f"accum={self.sft_grad_accum}, completed={self.trainer.global_step}")

    def on_before_optimizer_step(self, optimizer):
        # keep GOLU flushing
        try:
            golu_flush_buckets()
        except Exception:
            pass
    
        # IMPORTANT: trigger SFT reselection here (grads are populated, step not yet applied)
        if self.sft_enabled and self._sft_selector is not None:
            try:
                self._sft_selector.step()
            except Exception as e:
                if self.global_rank == 0:
                    print(f"[WARN] SFT selector pre-step failed: {e}")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # ------------- scheme management -------------
    def _ensure_choice_mapping(self, labels: torch.Tensor):
        device = self.device
        expected = DATASET_CLASS_INFO.get(self.task_name, None)
        if self._scheme is not None and self._scheme.name not in ("undetected",):
            if self.choice_id_list.numel() == 0:
                self.choice_id_list, self.list_to_class = self._scheme.as_tensors(device)
            return
        if expected is None:
            if self._scheme is None:
                self._scheme = _Scheme("open-ended", [], [])
                self.choice_id_list = torch.empty(0, dtype=torch.long, device=device)
                self.list_to_class  = torch.empty(0, dtype=torch.long, device=device)
            return
        if self._scheme is not None and self._scheme.name not in ("undetected","open-ended"):
            if self.choice_id_list.numel() == 0:
                self.choice_id_list, self.list_to_class = self._scheme.as_tensors(device)
            return
        sc = _detect_scheme_by_coverage(self.tokenizer, labels, self.task_name, expected)
        if sc.name != "undetected":
            self._scheme = sc
            self.choice_id_list, self.list_to_class = self._scheme.as_tensors(device)
            if self.global_rank == 0:
                ncls = len(set(self.list_to_class.tolist()))
                print(f"[INFO] Detected scheme for {self.task_name} = {self._scheme.name} "
                      f"• classes={ncls} • candidates={len(self.choice_id_list)}")
        else:
            if self._scheme is None:
                self._scheme = _Scheme("undetected", [], [])
                self.choice_id_list = torch.empty(0, dtype=torch.long, device=device)
                self.list_to_class  = torch.empty(0, dtype=torch.long, device=device)

    # ------------- math numeric EM helpers -------------
    def _extract_number_from_text(self, s: Optional[str]) -> Optional[float]:
        if s is None:
            return None
        s = s.replace(",", "")
        m = self._num_re.search(s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    @torch.no_grad()
    def _math_numeric_batch_eval(self, batch):
        """
        Greedy-generate from the prompt (before the first supervised label token).
        Compute numeric exact-match accuracy across the batch.
        """
        model, tok = self.model, self.tokenizer
        ids = batch["input_ids"]
        attn = batch["attention_mask"]
        labels = batch["labels"]
        B, T = ids.shape

        correct = 0
        pred_in_format = 0
        gold_in_allowed = 0

        for i in range(B):
            labels_i = labels[i]
            pos = torch.nonzero(labels_i != -100, as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                prompt_len = T
            else:
                prompt_len = int(pos[0].item())

            in_ids = ids[i, :prompt_len].unsqueeze(0)
            in_attn = attn[i, :prompt_len].unsqueeze(0)

            gen = model.generate(
                input_ids=in_ids,
                attention_mask=in_attn,
                max_new_tokens=self.max_gen_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )

            new_tokens = gen[0, in_ids.size(1):]
            pred_text = tok.decode(new_tokens.tolist(), skip_special_tokens=True)
            pred_num = self._extract_number_from_text(pred_text)
            if pred_num is not None:
                pred_in_format += 1

            gold_tokens = labels_i[labels_i != -100]
            gold_text = tok.decode(gold_tokens.tolist(), skip_special_tokens=True)
            gold_num = self._extract_number_from_text(gold_text)
            if gold_num is not None:
                gold_in_allowed += 1

            if (pred_num is not None) and (gold_num is not None):
                if abs(pred_num - gold_num) <= self._em_eps:
                    correct += 1

        denom = float(B) if B > 0 else 1.0
        return (
            correct / denom,
            pred_in_format / denom,
            gold_in_allowed / denom,
        )

    # ------------- train/eval steps -------------
    def on_train_epoch_start(self):
        self.epoch_correct.zero_(); self.epoch_total.zero_()
        if self.global_rank == 0 and self.current_epoch == 0 and self.trainer.global_step == 0:
            print(f"[INFO] Optimizer backend: {_OPTIM_BACKEND}")

    def _compute_batch_acc(self, logits: torch.Tensor, labels: torch.Tensor):
        device = labels.device
        if self.choice_id_list.numel() == 0 or self.list_to_class.numel() == 0:
            z = torch.tensor(0.0, device=device); return z, z, z, z, z
        if len(torch.unique(self.list_to_class)) < 2:
            z = torch.tensor(0.0, device=device); return z, z, z, z, z

        B, T, V = logits.shape
        ctx_len = (labels != -100).float().argmax(dim=1).clamp_max(T - 1)
        row = torch.arange(B, device=device)

        offset = getattr(self._scheme, "offset", 0) if (self._scheme is not None) else 0
        pos = (ctx_len - 1 + offset).clamp_min(0).clamp_max(T - 1)
        sel = logits[row, pos]
        pred_vocab_id = sel.argmax(dim=-1)
        allowed = self.choice_id_list

        pred_in_allowed = (pred_vocab_id.unsqueeze(1) == allowed.unsqueeze(0)).any(dim=1).float().mean()

        gold_pos = (ctx_len + offset).clamp_min(0).clamp_max(T - 1)
        gold_tok = labels[row, gold_pos]
        has_gold = (gold_tok.unsqueeze(1) == allowed.unsqueeze(0)).any(dim=1)
        gold_in_allowed = has_gold.float().mean()

        if has_gold.any():
            probs_over_allowed = sel.log_softmax(dim=-1)[:, allowed]
            pred_col = probs_over_allowed.argmax(dim=-1)
            pred_cls = self.list_to_class[pred_col]
            gold_col = (gold_tok.unsqueeze(1) == allowed.unsqueeze(0)).float().argmax(dim=1)
            gold_cls = self.list_to_class[gold_col]
            mask = has_gold
            correct = (pred_cls[mask] == gold_cls[mask]).sum()
            total   = mask.sum()
            acc = correct.float() / total.float()
            return acc, correct, total, pred_in_allowed, gold_in_allowed
        else:
            z = torch.tensor(0.0, device=device)
            return z, z, torch.tensor(0.0, device=device), pred_in_allowed, gold_in_allowed

    def training_step(self, batch, batch_idx):
        self._ensure_choice_mapping(batch["labels"])
        outs = self(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outs.loss
        with torch.no_grad():
            acc, correct, total, pred_in_allowed, gold_in_allowed = self._compute_batch_acc(
                outs.logits.detach(), batch["labels"]
            )
        self.epoch_correct += correct
        self.epoch_total   += total
        # EMA loss
        self.loss_ema = self.ema_beta * self.loss_ema + (1 - self.ema_beta) * loss.detach()
        if batch_idx % self._log_every_n == 0:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            acc_str = f"{(acc.item() if total.item() > 0 else 0.0):.3f}" if torch.is_tensor(total) else "—"
            print(f"[TRAIN] step={self.global_step:>6} loss={loss.item():.4f}  "
                  f"ema={self.loss_ema.item():.4f}  acc={acc_str}  lr={lr:.2e}")
            print(f"        pred_in_format={pred_in_allowed.item():.3f}  gold_in_allowed={gold_in_allowed.item():.3f}")
            report_gpu_usage("train_step")
        self.log("train_loss", loss, on_step=True, logger=True)
        self.log("train_pred_in_format", pred_in_allowed, on_step=True, logger=True)
        self.log("train_gold_in_allowed", gold_in_allowed, on_step=True, logger=True)
        return loss

    @torch.no_grad()
    def _eval_batch_cls(self, batch, stage: str):
        """Classification-style eval (BOOL/MC)."""
        self._ensure_choice_mapping(batch["labels"])
        outs = self(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        acc, _, _, pred_in_allowed, gold_in_allowed = self._compute_batch_acc(outs.logits, batch["labels"])
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_pred_in_format", pred_in_allowed, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_gold_in_allowed", gold_in_allowed, prog_bar=False, on_epoch=True, sync_dist=True)

    def _should_use_numeric_eval(self) -> bool:
        """Use numeric EM when requested and scheme is open/undetected (i.e., math open-ended)."""
        if not self.math_numeric_eval:
            return False
        if self._scheme is None:
            return True
        return self._scheme.name in ("open-ended", "undetected")

    def validation_step(self, batch, _):
        if self._should_use_numeric_eval():
            acc, pif, gia = self._math_numeric_batch_eval(batch)
            self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_pred_in_format", pif, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val_gold_in_allowed", gia, on_step=False, on_epoch=True, sync_dist=True)
        else:
            self._eval_batch_cls(batch, "val")

    def test_step(self, batch, _):
        if self._should_use_numeric_eval():
            acc, pif, gia = self._math_numeric_batch_eval(batch)
            self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_pred_in_format", pif, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test_gold_in_allowed", gia, on_step=False, on_epoch=True, sync_dist=True)
        else:
            self._eval_batch_cls(batch, "test")

    def on_validation_epoch_end(self):
        cm = self.trainer.callback_metrics
        def _grab(name, default="n/a"):
            x = cm.get(name, None)
            try:
                return f"{float(x):.4f}" if x is not None else default
            except Exception:
                return default
        print(
            f"[VAL][epoch={self.current_epoch}] "
            f"acc={_grab('val_acc')}  "
            f"pred_in_format={_grab('val_pred_in_format')}  "
            f"gold_in_allowed={_grab('val_gold_in_allowed')}"
        )

    # ------------- optimizer / hooks -------------
    def configure_optimizers(self):
        # Prefer params from the wrapped model (PEFT lives there)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            # Fallback to module-level in the unlikely event nothing was marked trainable yet
            params = [p for p in self.parameters() if p.requires_grad]
    
        if not params:
            raise RuntimeError("No trainable parameters found. Did you wrap the model before creating the LightningModule?")
    
        if self.sft_enabled and _HAS_SFT and (SftAdamW is not None or SftSM3 is not None):
            optim_groups = [{"params": params, "weight_decay": 0.0}]
    
            if self.sft_use_sm3:
                if SftSM3 is None:
                    raise RuntimeError("SFT/SM3 requested but SftSM3 is unavailable.")
                try:
                    # Map of parameter tensors -> delta objects, as required by SftSM3
                    deltas = {delta.values: delta for _m, _n, delta in self.model.active_deltas()}
                except Exception as e:
                    raise RuntimeError(f"SFT/SM3 requires model.active_deltas(); got: {e}")
                opt = SftSM3(
                    optim_groups,
                    deltas=deltas,
                    beta=self.sft_beta,
                    lr=self.learning_rate,
                )
                if self.global_rank == 0:
                    print("[INFO] Optimizer: SFT-SM3")
                return opt
    
            # Default SFT optimizer: AdamW with fp32 momentum for stability
            if SftAdamW is None:
                raise RuntimeError("SFT requested but SftAdamW is unavailable.")
            opt = SftAdamW(
                optim_groups,
                lr=self.learning_rate,
                momentum_dtype=torch.float32,
            )
            if self.global_rank == 0:
                print("[INFO] Optimizer: SFT-AdamW")
            return opt
    
        # Non-SFT path (GOLU/LoRA/FULL)
        optim_kwargs: Dict[str, object] = dict(lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        opt = _FusedAdam(params, **optim_kwargs)
        if self.global_rank == 0:
            print(f"[INFO] Optimizer: FusedAdam backend = {_OPTIM_BACKEND}")
        return opt

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        # Commit GOLU deltas & swap next mask
        def _commit(m):
            if isinstance(m, SparseUpdateLinear):
                m.commit_and_swap_mask()
        self.model.apply(_commit)

    def optimizer_zero_grad(self, *args, **kwargs):
        opt = kwargs.get("optimizer", None)
        if opt is None and len(args) >= 3:
            opt = args[2]
        if opt is not None:
            opt.zero_grad(set_to_none=True)
