# train.py — GASDU-only fine-tuning on local datasets
# - No internet calls. Supports .json / .jsonl (optionally .gz).
# - Commonsense tasks now use train.json (train) and test.json (validation),
#   mirroring the NumGLUE behavior.
# - Math tasks keep the original behavior:
#     *_*_1.json + test.json collected and split 90/10 deterministically.
#
# Assumed instruction-style records (for commonsense & many math/NumGLUE):
#   { "instruction": str, "input": str, "output": str, "answer": str }
# We parse "Answer format:" and normalize to a discriminative label set so the
# FIRST supervised token is unique per class (important for accuracy).

from lightning.pytorch.strategies import SingleDeviceStrategy
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import argparse, os, sys, time, random, gzip, json, io, re, glob
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- GASDU layer ---
from gasdu import SparseUpdateLinear

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────── Registries ─────────────────────────────── #
# "kind": "bool" | "mc" | "open"
COMMONSENSE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "boolq":      {"classes": 2, "kind": "bool"},
    "piqa":       {"classes": 2, "kind": "mc", "default_prefix": "solution"},
    "siqa":       {"classes": 3, "kind": "mc", "default_prefix": "answer"},
    "hellaswag":  {"classes": 4, "kind": "mc", "default_prefix": "ending"},
    "winogrande": {"classes": 2, "kind": "mc", "default_prefix": "option"},
    "arc_e":      {"classes": 4, "kind": "mc", "default_prefix": "answer"},
    "arc_c":      {"classes": 4, "kind": "mc", "default_prefix": "answer"},
    "obqa":       {"classes": 4, "kind": "mc", "default_prefix": "answer"},
}

MATH_REGISTRY: Dict[str, Dict[str, Any]] = {
    "multiarith": {"classes": None, "kind": "open"},
    "gsm8k":      {"classes": None, "kind": "open"},
    "addsub":     {"classes": None, "kind": "open"},
    "aqua":       {"classes": 5,    "kind": "mc", "default_prefix": "answer"},  # 5-way
    "singleeq":   {"classes": None, "kind": "open"},
    "svamp":      {"classes": None, "kind": "open"},
    # add others if you need them
}

# Map CLI task_name → folder name under ./dataset
TASK_FOLDER_MAP: Dict[str, str] = {
    # commonsense
    "boolq": "boolq",
    "piqa": "piqa",
    "siqa": "social_i_qa",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "arc_e": "ARC-Easy",
    "arc_c": "ARC-Challenge",
    "obqa": "openbookqa",
    # math
    "aqua": "AQuA",
    "gsm8k": "gsm8k",
    "addsub": "AddSub",
    "multiarith": "MultiArith",
    "singleeq": "SingleEq",
    "svamp": "SVAMP",
    # optional extras
    "mathqa": "mathqa",
    "mawps": "mawps",
    # NumGLUE (Types 1–8 live directly under dataset/Type_1 ... Type_8)
    "numglue_type1": "Type_1",
    "numglue_type2": "Type_2",
    "numglue_type3": "Type_3",
    "numglue_type4": "Type_4",
    "numglue_type5": "Type_5",
    "numglue_type6": "Type_6",
    "numglue_type7": "Type_7",
    "numglue_type8": "Type_8",
}

# ─────────────────────────────── CLI ─────────────────────────────── #
parser = argparse.ArgumentParser(description="GASDU fine-tuning on a single local dataset")
parser.add_argument("task_name", type=str, help="Dataset name (commonsense, math subtask, or NumGLUE type).")
parser.add_argument("--dataset_root", type=str, default="dataset", help="Root folder containing task subfolders.")
parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B", help="HF model path or id.")
parser.add_argument("--max_epoch", type=int, default=3)
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--update_percent", type=float, default=0.1, help="x%% of total params updated each iter (e.g., 0.71 means 0.71%%).")
parser.add_argument("--use_dynamic_k", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--stage1_only", action="store_true")
parser.add_argument("--stage2_only", action="store_true")
parser.add_argument("--mask_mode", default="topk", choices=["topk", "bernoulli", "fixed_topk", "topk_refresh_each_step"])
parser.add_argument("--track_grad_norm_prop", action="store_true")
parser.add_argument("--full_grad_every", type=int, default=50)
parser.add_argument("--stage2_seeds", type=str, default="",
                    help="Optional comma-separated list of seeds for Stage-2 (e.g., '42' or '42,43,233'). "
                         "If empty, use default [42, 43, 233].")
args = parser.parse_args()

task_name_raw = args.task_name
task_name = task_name_raw.lower().strip()
model_name = args.model_name

# ───────────────────────────── NumGLUE helpers ───────────────────────────── #
_NUMGLUE_ALIASES = {
    # accept many user spellings
    "numglue_type1": 1, "numglue_t1": 1, "type1": 1, "t1": 1,
    "numglue_type2": 2, "numglue_t2": 2, "type2": 2, "t2": 2,
    "numglue_type3": 3, "numglue_t3": 3, "type3": 3, "t3": 3,
    "numglue_type4": 4, "numglue_t4": 4, "type4": 4, "t4": 4,
    "numglue_type5": 5, "numglue_t5": 5, "type5": 5, "t5": 5,
    "numglue_type6": 6, "numglue_t6": 6, "type6": 6, "t6": 6,
    "numglue_type7": 7, "numglue_t7": 7, "type7": 7, "t7": 7,
    "numglue_type8": 8, "numglue_t8": 8, "type8": 8, "t8": 8,
}

def is_numglue_task(name: str) -> bool:
    if name.startswith("numglue"):
        return True
    return name in _NUMGLUE_ALIASES

def _resolve_numglue_folder(dataset_root: str, name: str) -> str:
    """
    Resolve NumGLUE Type folders to the flat layout:
      • dataset/Type_1, dataset/Type_2, ..., dataset/Type_8
    """
    direct = os.path.join(dataset_root, TASK_FOLDER_MAP.get(name, name))
    if os.path.isdir(direct):
        return direct

    tnum = None
    if name in _NUMGLUE_ALIASES:
        tnum = _NUMGLUE_ALIASES[name]
    else:
        m = re.match(r"(?:numglue[_\- ]*type|type|t)\s*(\d+)$", name, flags=re.I)
        if m:
            tnum = int(m.group(1))

    if tnum is not None:
        c = os.path.join(dataset_root, f"Type_{tnum}")
        if os.path.isdir(c):
            return c

    fallback = os.path.join(dataset_root, name)
    if os.path.isdir(fallback):
        return fallback

    raise FileNotFoundError(
        f"[NumGLUE] Could not resolve folder for '{name}'. "
        f"Expected: {os.path.join(dataset_root, 'Type_1')} ... {os.path.join(dataset_root, 'Type_8')} "
        f"or an existing folder named '{name}' under {dataset_root}."
    )

is_commonsense = task_name in COMMONSENSE_REGISTRY
is_math = task_name in MATH_REGISTRY
is_numglue = is_numglue_task(task_name)

if not (is_commonsense or is_math or is_numglue):
    raise ValueError(
        f"Unknown dataset '{task_name_raw}'. Valid commonsense: {list(COMMONSENSE_REGISTRY)}; "
        f"math: {list(MATH_REGISTRY)}; or NumGLUE types: Type1..Type8 (aliases accepted)."
    )

# ─────────────────────────── Output dirs ─────────────────────────── #
method_tag = "gasdu"
out_basename = os.path.basename(model_name).replace("/", "-")
_numglue_suffix = "_numglue" if is_numglue else ""
out_dir = f"./GASDU_{task_name}{_numglue_suffix}_{out_basename}_{method_tag}_{args.mask_mode}"
os.makedirs(out_dir, exist_ok=True)
logs_dir = os.path.join(out_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# ───────────────────────────── Seeding ───────────────────────────── #
def set_random_seed(sd: int):
    random.seed(sd); np.random.seed(sd)
    torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
_SPLIT_SEED = 137  # fixed seed for deterministic 90/10 train/val splits (for math)

# ─────────────────────── GPU pick (simple) ─────────────────────── #
def _pick_gpu_with_lowest_mem():
    """
    Return a **local** CUDA ordinal (0..device_count-1), respecting CUDA_VISIBLE_DEVICES.
    - GASDU_FORCE_LOCAL_DEVICE: hard override (e.g., "0")
    - LOCAL_RANK: respected if present (e.g., DDP)
    - Otherwise: pick local 0 (safe when each worker masks to a single GPU)
    """
    if not torch.cuda.is_available():
        return None

    # Hard override
    force = os.environ.get("GASDU_FORCE_LOCAL_DEVICE")
    if force is not None:
        try:
            return int(force)
        except Exception:
            return 0

    # Respect LOCAL_RANK (if using DDP later)
    lr = os.environ.get("LOCAL_RANK")
    if lr is not None:
        try:
            return int(lr)
        except Exception:
            return 0

    try:
        n = torch.cuda.device_count()
        if n >= 1:
            return 0
    except Exception:
        pass
    return None

SELECTED_GPU = _pick_gpu_with_lowest_mem()
print(f"► Using GPU {SELECTED_GPU}" if SELECTED_GPU is not None else "► No suitable GPU — CPU mode.")
device = torch.device(f"cuda:{SELECTED_GPU}" if SELECTED_GPU is not None else "cpu")

# ── Helpers to capture peak GPU memory (GB) ──────────────────────── #
def _reset_gpu_peak():
    try:
        if SELECTED_GPU is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass

def _peak_gpu_gb() -> float:
    try:
        if SELECTED_GPU is None or not torch.cuda.is_available():
            return 0.0
        alloc = torch.cuda.max_memory_allocated(device)
        reserv = torch.cuda.max_memory_reserved(device)
        return max(alloc, reserv) / (1024.0**3)
    except Exception:
        return 0.0

# ───────────────────────────── Local data utils ───────────────────────────── #
def _open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def _load_single_json_like(path: str) -> List[Dict[str, Any]]:
    """Load a single .json or .jsonl (optionally .gz). Returns list of dict records."""
    with _open_maybe_gz(path) as f:
        sniff = f.read(2048)
        if isinstance(sniff, bytes): sniff = sniff.decode("utf-8", errors="replace")
        sniff = sniff.lstrip("\ufeff").lstrip()
        f.seek(0)
        if sniff.startswith("{") or sniff.startswith("["):
            text = f.read()
            if isinstance(text, bytes): text = text.decode("utf-8", errors="replace")
            text = text.lstrip("\ufeff")
            obj = json.loads(text)
            if isinstance(obj, list): return obj
            if isinstance(obj, dict):
                for k in ("data","examples","records","items","train","validation","test"):
                    if isinstance(obj.get(k), list): return obj[k]
                return [obj]
        # JSONL
        f.seek(0)
        out = []
        for ln in f:
            if isinstance(ln, bytes): ln = ln.decode("utf-8", errors="replace")
            ln = ln.strip()
            if not ln or ln.startswith("//") or ln.startswith("#"): continue
            if ln.endswith(","): ln = ln[:-1].rstrip()
            try: obj = json.loads(ln)
            except Exception: continue
            if isinstance(obj, dict): out.append(obj)
        return out

# ───────────────────── Math/NumGLUE normalization helpers ─────────────────── #
def _coerce_answer_to_str(x: Any) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        from math import isclose
        return str(int(round(x))) if isclose(x, round(x), abs_tol=1e-12) else str(x)
    return str(x).strip()

def _normalize_math_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Already instruction-style?
    if isinstance(rec.get("instruction", None), str):
        ins = rec["instruction"].strip()
        if not ins: return None
        ans = rec.get("answer", None)
        if ans is None: return None
        out = rec.get("output", "")
        return {
            "instruction": ins,
            "input": "" if rec.get("input") is None else str(rec.get("input")),
            "output": "" if out is None else str(out),
            "answer": _coerce_answer_to_str(ans),
        }

    q = rec.get("question", None)
    a = rec.get("answer", None)
    if not isinstance(q, str) or a is None:
        return None

    cot = rec.get("chain-of-thought", rec.get("rationale", rec.get("solution", "")))
    return {
        "instruction": q.strip(),
        "input": "",
        "output": "" if cot is None else str(cot),
        "answer": _coerce_answer_to_str(a),
    }

def _list_math_candidate_files(root: str) -> List[str]:
    pats = [
        "**/*_1.json", "**/*_1.json.gz", "**/*_1.jsonl", "**/*_1.jsonl.gz",
        "**/test.json", "**/test.json.gz", "**/test.jsonl", "**/test.jsonl.gz",
    ]
    files: List[str] = []
    for pat in pats:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return sorted(set(files))

def _load_task_records_from_dir(dataset_root: str, task: str) -> List[Dict[str, Any]]:
    folder = TASK_FOLDER_MAP.get(task, task)
    root = os.path.join(dataset_root, folder)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Task folder not found: {root}")

    is_math_task = task in MATH_REGISTRY
    if is_math_task:
        files = _list_math_candidate_files(root)
    else:
        # not used for commonsense anymore
        files = []

    if is_math_task and not files:
        raise RuntimeError(
            f"No candidate files (*_1.json / test.json) found for math task '{task}' under {root}"
        )

    records: List[Dict[str, Any]] = []
    bad_files = 0
    for fp in files:
        try:
            recs = _load_single_json_like(fp)
            for r in recs:
                if is_math_task:
                    nr = _normalize_math_record(r)
                    if nr is not None and isinstance(nr.get("instruction", ""), str):
                        records.append(nr)
        except Exception:
            bad_files += 1
            continue

    if bad_files:
        print(f"[DATA] Warning: {bad_files} files failed to load under {root}")
    if is_math_task and not records:
        raise RuntimeError(f"No usable records for math task '{task}' from {root}")

    return records

# ─────────────────── Generic train/test loader (commonsense) ───────────────── #
def _load_commonsense_train_test(dataset_root: str, task: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For commonsense tasks:
      - Directly load train.json and test.json inside the task folder.
      - Expect instruction-style rows; skip malformed rows.
    """
    folder = TASK_FOLDER_MAP.get(task, task)
    base = os.path.join(dataset_root, folder)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"[Commonsense] Folder not found: {base}")

    def _first_existing(paths: List[str]) -> Optional[str]:
        for p in paths:
            if os.path.isfile(p):
                return p
        return None

    train_fp_candidates = [
        os.path.join(base, "train.json"),
        os.path.join(base, "train.jsonl"),
        os.path.join(base, "train.json.gz"),
        os.path.join(base, "train.jsonl.gz"),
    ]
    test_fp_candidates = [
        os.path.join(base, "test.json"),
        os.path.join(base, "test.jsonl"),
        os.path.join(base, "test.json.gz"),
        os.path.join(base, "test.jsonl.gz"),
    ]
    tr_path = _first_existing(train_fp_candidates)
    te_path = _first_existing(test_fp_candidates)
    if tr_path is None or te_path is None:
        raise FileNotFoundError(
            f"[Commonsense] Missing required files. train.json present? {tr_path is not None}; test.json present? {te_path is not None}. "
            f"Looked under: {base}"
        )

    tr_raw = _load_single_json_like(tr_path)
    te_raw = _load_single_json_like(te_path)

    def _keep_instruction_style(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            if isinstance(r, dict) and isinstance(r.get("instruction", ""), str):
                out.append({
                    "instruction": r["instruction"],
                    "input": "" if r.get("input") is None else str(r.get("input")),
                    "output": "" if r.get("output") is None else str(r.get("output")),
                    "answer": _coerce_answer_to_str(r.get("answer", "")),
                })
        return out

    train_records = _keep_instruction_style(tr_raw)
    test_records  = _keep_instruction_style(te_raw)
    if not train_records: raise RuntimeError("[Commonsense] No usable rows in train.json")
    if not test_records:  raise RuntimeError("[Commonsense] No usable rows in test.json")
    return train_records, test_records

# ──────────────────────── NumGLUE: direct train/test I/O ─────────────────────── #
def _load_numglue_train_test(dataset_root: str, task: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    folder = _resolve_numglue_folder(dataset_root, task)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"[NumGLUE] Could not resolve dataset folder for '{task}'. Tried '{folder}'.")

    def _first_existing(paths: List[str]) -> Optional[str]:
        for p in paths:
            if os.path.isfile(p):
                return p
        return None

    train_fp_candidates = [
        os.path.join(folder, "train.json"),
        os.path.join(folder, "train.jsonl"),
        os.path.join(folder, "train.json.gz"),
        os.path.join(folder, "train.jsonl.gz"),
    ]
    test_fp_candidates = [
        os.path.join(folder, "test.json"),
        os.path.join(folder, "test.jsonl"),
        os.path.join(folder, "test.json.gz"),
        os.path.join(folder, "test.jsonl.gz"),
    ]
    tr_path = _first_existing(train_fp_candidates)
    te_path = _first_existing(test_fp_candidates)
    if tr_path is None or te_path is None:
        raise FileNotFoundError(
            f"[NumGLUE] Missing required files. train.json present? {tr_path is not None}; test.json present? {te_path is not None}. "
            f"Looked under: {folder}"
        )

    tr_raw = _load_single_json_like(tr_path)
    te_raw = _load_single_json_like(te_path)

    train_records: List[Dict[str, Any]] = []
    test_records:  List[Dict[str, Any]] = []

    bad_tr, bad_te = 0, 0
    for r in tr_raw:
        try:
            nr = _normalize_math_record(r)
            if nr is not None and isinstance(nr.get("instruction",""), str):
                train_records.append(nr)
            else:
                bad_tr += 1
        except Exception:
            bad_tr += 1
    for r in te_raw:
        try:
            nr = _normalize_math_record(r)
            if nr is not None and isinstance(nr.get("instruction",""), str):
                test_records.append(nr)
            else:
                bad_te += 1
        except Exception:
            bad_te += 1

    if bad_tr:
        print(f"[NumGLUE] Warning: {bad_tr} malformed rows skipped in train.json")
    if bad_te:
        print(f"[NumGLUE] Warning: {bad_te} malformed rows skipped in test.json")
    if not train_records:
        raise RuntimeError("[NumGLUE] No usable rows in train.json")
    if not test_records:
        raise RuntimeError("[NumGLUE] No usable rows in test.json")

    return train_records, test_records

def _split_train_val(records: List[Dict[str, Any]], val_ratio: float = 0.1, seed: int = _SPLIT_SEED) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    cut = max(1, int(len(idx) * (1.0 - val_ratio)))
    train_idx, val_idx = idx[:cut], idx[cut:]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    return train, val

# ───────────────────────── Formatting helpers ───────────────────────── #
_BOOL_TRUE = {"true","1","yes","y"}
_BOOL_FALSE = {"false","0","no","n"}

def _prefix_and_K_from_records_or_defaults(records, dsname) -> Tuple[str, Optional[int]]:
    reg = COMMONSENSE_REGISTRY.get(dsname, {})
    if reg.get("kind") == "bool":
        return "bool", 2

    for rec in records:
        instr = rec.get("instruction", "")
        fmt = _parse_answer_format(instr)
        if fmt:
            if fmt[0] == "bool":
                return "bool", 2
            prefix, K = fmt
            if isinstance(K, int) and K >= 2:
                return prefix, K

    mc_prefixes = ["answer", "solution", "ending", "option"]
    max_idx = 0
    chosen_prefix = None
    for rec in records:
        ans = rec.get("answer", "")
        if not isinstance(ans, str):
            continue
        for p in mc_prefixes:
            m = re.search(rf"\b{re.escape(p)}\s*([1-9][0-9]*)\b", ans.lower())
            if m:
                idx = int(m.group(1)); 
                if idx > max_idx:
                    max_idx = idx; chosen_prefix = p
    if chosen_prefix and max_idx >= 2:
        return chosen_prefix, max_idx
    return "open", None

def _extract_bool_answer(ex: Dict[str, Any]) -> Optional[bool]:
    a = ex.get("answer")
    if isinstance(a, str):
        s = a.strip().lower()
        if s in _BOOL_TRUE: return True
        if s in _BOOL_FALSE: return False
    elif isinstance(a, bool):
        return bool(a)
    out = ex.get("output")
    if isinstance(out, str):
        last = None
        for m in re.finditer(r"\b(true|false)\b", out.lower()):
            last = m.group(1)
        if last is not None:
            return last == "true"
    return None

def _extract_indexed_answer(ex: Dict[str, Any], prefixes: List[str]) -> Optional[Tuple[str,int]]:
    def find_pair(s: str) -> Optional[Tuple[str,int]]:
        s = s.strip().lower()
        for p in prefixes:
            m = re.search(rf"\b{re.escape(p)}\s*([1-9][0-9]*)\b", s)
            if m:
                return p, int(m.group(1))
        return None

    a = ex.get("answer")
    if isinstance(a, str):
        hit = find_pair(a)
        if hit: return hit
    out = ex.get("output")
    if isinstance(out, str):
        last = None
        for p in prefixes:
            for m in re.finditer(rf"\b{re.escape(p)}\s*([1-9][0-9]*)\b", out.lower()):
                last = (p, int(m.group(1)))
        if last: return last
    return None

def _parse_answer_format(instr: str) -> Optional[Tuple[str,int]]:
    if not isinstance(instr, str): return None
    m = re.search(r"answer\s*format\s*:\s*(.+)", instr, re.I|re.S)
    if not m: return None
    tail = m.group(1).strip()
    tail = tail.splitlines()[0]

    if re.search(r"\b(?:a\s+)?(?:number|numeric|integer|float|decimal)\b", tail, re.I):
        return ("numeric", 0)

    raw_parts = re.split(r"[\/,;]", tail)
    parts: List[str] = []
    for x in raw_parts:
        x = x.strip()
        if x in {"...", "…", "etc", "etc."}:
            continue
        if x:
            parts.append(x)
    if not parts:
        return None

    if all(re.fullmatch(r"(true|false)", p, re.I) for p in parts) and len(parts) == 2:
        return ("bool", 2)

    pref = None
    idxs: List[int] = []
    for p in parts:
        mm = re.match(r"([a-zA-Z]+)\s*([1-9][0-9]*)$", p)
        if not mm:
            return None
        pf, num = mm.group(1).lower(), int(mm.group(2))
        pref = pf if pref is None else pref
        if pf != pref:
            return None
        idxs.append(num)
    if not idxs:
        return None
    K = max(idxs)
    return (pref, K)

def _task_defaults(dsname: str) -> Tuple[str, Optional[int]]:
    reg = COMMONSENSE_REGISTRY.get(dsname) or MATH_REGISTRY.get(dsname) or {}
    kind = reg.get("kind")
    if kind == "bool":
        return ("bool", 2)
    if kind == "mc":
        return (reg.get("default_prefix","answer"), reg.get("classes"))
    return ("open", None)

def _first_token_id(tok, text: str) -> Optional[int]:
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if ids else None

def _build_candidate_label_sets(tok, K: int, preferred_prefix: Optional[str]) -> List[List[str]]:
    sets: List[List[str]] = []
    prefixes = []
    if preferred_prefix: prefixes.append(preferred_prefix)
    for p in ["answer","solution","ending","option"]:
        if p not in prefixes: prefixes.append(p)
    for p in prefixes:
        sets.append([f" {p}{i}" for i in range(1, K+1)])
    letters = list("ABCDE")[:K]
    sets.append([f" {L}" for L in letters])
    sets.append([f" {i}" for i in range(1, K+1)])
    return sets

def _pick_discriminative_labels(tok, K: int, preferred_prefix: Optional[str]) -> List[str]:
    for cand in _build_candidate_label_sets(tok, K, preferred_prefix):
        first_ids = [ _first_token_id(tok, lab) for lab in cand ]
        if any(fid is None for fid in first_ids):
            continue
        if len(set(first_ids)) == K:
            return cand
    return [f" {L}" for L in list("ABCDE")[:K]]

def _ensure_prompt_matches_labels(instr: str, labels: List[str]) -> str:
    canonical = " / ".join(l.strip() for l in labels)
    if re.search(r"(?im)^.*\banswer\s*format\s*:\s*.*$", instr):
        instr = re.sub(r"(?im)^.*\banswer\s*format\s*:\s*.*$",
                       f"Answer format: {canonical}", instr)
    else:
        instr = instr.rstrip() + f"\nAnswer format: {canonical}"
    return instr

def _append_answer_stub(instr: str) -> str:
    s = instr.rstrip()
    return s + " " if s.endswith("Answer:") else s + "\nAnswer: "

def _extract_enumerated_labels(instr: str) -> Optional[List[str]]:
    if not isinstance(instr, str):
        return None
    m = re.search(r"answer\s*format\s*:\s*(.+)", instr, re.I | re.S)
    if not m:
        return None
    tail = m.group(1).strip().splitlines()[0]
    if re.search(r"\b(?:a\s+)?(?:number|numeric|integer|float|decimal)\b", tail, re.I):
        return None
    if re.search(r"\btrue\b", tail, re.I) and re.search(r"\bfalse\b", tail, re.I):
        return None
    parts = [p.strip() for p in re.split(r"[\/,;]", tail) if p.strip()]
    parts = [p for p in parts if p not in {"...", "…", "etc", "etc."}]
    if not (2 <= len(parts) <= 10):
        return None
    clean = []
    for p in parts:
        if re.fullmatch(r"[A-Za-z][A-Za-z _\-]*", p):
            clean.append(p.lower().strip())
        else:
            return None
    return clean if clean else None

def _ensure_discriminative_from_list(tok, labels: List[str]) -> List[str]:
    labs = [" " + l.strip() for l in labels]
    fids = [_first_token_id(tok, lab) for lab in labs]
    if None not in fids and len(set(fids)) == len(labs):
        return labs
    K = len(labels)
    return [f" {L}" for L in list("ABCDE")[:K]]

def _extract_enum_answer(ex: Dict[str, Any], allowed: List[str]) -> Optional[int]:
    def find(s: str) -> Optional[int]:
        s_low = s.lower()
        hit = None
        for i, lab in enumerate(allowed):
            if lab in s_low:
                hit = i
        return hit

    a = ex.get("answer")
    if isinstance(a, str):
        idx = find(a)
        if idx is not None:
            return idx
    o = ex.get("output")
    if isinstance(o, str):
        idx = find(o)
        if idx is not None:
            return idx
    return None

def _build_prompt_and_answer(example: Dict[str, Any], dsname: str, max_length: int, tok):
    instr = str(example.get("instruction","")).strip()
    if not instr: return None
    instr = re.sub(r"\n{3,}", "\n\n", instr)

    fmt = _parse_answer_format(instr)

    if fmt and fmt[0] == "numeric":
        ans = example.get("answer", example.get("output", ""))
        if ans is None: return None
        answer_text = " " + str(ans).strip()
        prompt = _append_answer_stub(instr)

        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        answer_ids = tok.encode(answer_text, add_special_tokens=False)
        keep_ans = min(len(answer_ids), max_length - 1)
        answer_ids = answer_ids[-keep_ans:]
        room = max(0, max_length - keep_ans - 1)
        prompt_ids = prompt_ids[-room:]
        input_ids = prompt_ids + answer_ids + [tok.eos_token_id]
        labels    = [-100]*len(prompt_ids) + answer_ids + [-100]
        input_ids = input_ids[:max_length]; labels = labels[:max_length]
        pad_len = max_length - len(input_ids)
        if pad_len:
            input_ids += [tok.pad_token_id]*pad_len
            labels    += [-100]*pad_len
        attention_mask = [1 if t != tok.pad_token_id else 0 for t in input_ids]
        return input_ids, labels, attention_mask

    if fmt and fmt[0] == "bool":
        tf = _extract_bool_answer(example)
        if tf is None: return None
        labels_for_fmt = [" true", " false"]
        instr = _ensure_prompt_matches_labels(instr, labels_for_fmt)
        answer_text = " true" if tf else " false"
        prompt = _append_answer_stub(instr)
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        answer_ids = tok.encode(answer_text, add_special_tokens=False)
        keep_ans = min(len(answer_ids), max_length - 1)
        answer_ids = answer_ids[-keep_ans:]
        room = max(0, max_length - keep_ans - 1)
        prompt_ids = prompt_ids[-room:]
        input_ids = prompt_ids + answer_ids + [tok.eos_token_id]
        labels    = [-100]*len(prompt_ids) + answer_ids + [-100]
        input_ids = input_ids[:max_length]; labels = labels[:max_length]
        pad_len = max_length - len(input_ids)
        if pad_len:
            input_ids += [tok.pad_token_id]*pad_len
            labels    += [-100]*pad_len
        attention_mask = [1 if t != tok.pad_token_id else 0 for t in input_ids]
        return input_ids, labels, attention_mask

    enum_labels = _extract_enumerated_labels(instr) if not (fmt and fmt[0] in ("bool", "numeric")) else None
    if enum_labels:
        idx = _extract_enum_answer(example, enum_labels)
        if idx is None:
            ans = example.get("answer", example.get("output", ""))
            if ans is None: return None
            prompt = _append_answer_stub(instr)
            answer_text = " " + str(ans).strip()
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            answer_ids = tok.encode(answer_text, add_special_tokens=False)
            keep_ans = min(len(answer_ids), max_length - 1)
            answer_ids = answer_ids[-keep_ans:]
            room = max(0, max_length - keep_ans - 1)
            prompt_ids = prompt_ids[-room:]
            input_ids = prompt_ids + answer_ids + [tok.eos_token_id]
            labels    = [-100]*len(prompt_ids) + answer_ids + [-100]
            input_ids = input_ids[:max_length]; labels = labels[:max_length]
            pad = max_length - len(input_ids)
            if pad:
                input_ids += [tok.pad_token_id]*pad
                labels    += [-100]*pad
            attention_mask = [1 if t != tok.pad_token_id else 0 for t in input_ids]
            return input_ids, labels, attention_mask

        labels_set = _ensure_discriminative_from_list(tok, enum_labels)
        instr = _ensure_prompt_matches_labels(instr, labels_set)
        answer_text = labels_set[idx]
        prompt = _append_answer_stub(instr)
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        answer_ids = tok.encode(answer_text, add_special_tokens=False)
        keep_ans = min(len(answer_ids), max_length - 1)
        answer_ids = answer_ids[-keep_ans:]
        room = max(0, max_length - keep_ans - 1)
        prompt_ids = prompt_ids[-room:]
        input_ids = prompt_ids + answer_ids + [tok.eos_token_id]
        labels    = [-100]*len(prompt_ids) + answer_ids + [-100]
        input_ids = input_ids[:max_length]; labels = labels[:max_length]
        pad = max_length - len(input_ids)
        if pad:
            input_ids += [tok.pad_token_id]*pad
            labels    += [-100]*pad
        attention_mask = [1 if t != tok.pad_token_id else 0 for t in input_ids]
        return input_ids, labels, attention_mask

    if fmt and fmt[0] not in ("bool", "numeric"):
        prefix, K = fmt
        pair = _extract_indexed_answer(example, prefixes=[prefix]) or \
               _extract_indexed_answer(example, prefixes=["answer","solution","ending","option"])
        if pair is None:
            ans = example.get("answer", example.get("output", ""))
            if ans is None: return None
            instr_open = re.sub(r"(?im)^.*\banswer\s*format\s*:\s*.*$", "", instr).strip()
            prompt = _append_answer_stub(instr_open)
            answer_text = " " + str(ans).strip()
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            answer_ids = tok.encode(answer_text, add_special_tokens=False)
            keep_ans = min(len(answer_ids), max_length - 1)
            answer_ids = answer_ids[-keep_ans:]
            room = max(0, max_length - keep_ans - 1)
            prompt_ids = prompt_ids[-room:]
            input_ids = prompt_ids + answer_ids + [tok.eos_token_id]
            labels    = [-100]*len(prompt_ids) + answer_ids + [-100]
            input_ids = input_ids[:max_length]; labels = labels[:max_length]
            pad_len = max_length - len(input_ids)
            if pad_len:
                input_ids += [tok.pad_token_id]*pad_len
                labels    += [-100]*pad_len
            attention_mask = [1 if t != tok.pad_token_id else 0 for t in input_ids]
            return input_ids, labels, attention_mask

        _, idx = pair
        if K is not None and idx > K:
            return None
        K_eff = K if isinstance(K, int) and K >= 2 else 5
        labels_set = _pick_discriminative_labels(tok, K_eff, preferred_prefix=prefix)
        instr = _ensure_prompt_matches_labels(instr, labels_set)
        answer_text = labels_set[idx - 1]
        prompt = _append_answer_stub(instr)
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        answer_ids = tok.encode(answer_text, add_special_tokens=False)
        keep_ans = min(len(answer_ids), max_length - 1)
        answer_ids = answer_ids[-keep_ans:]
        room = max(0, max_length - keep_ans - 1)
        prompt_ids = prompt_ids[-room:]
        input_ids = prompt_ids + answer_ids + [tok.eos_token_id]
        labels    = [-100]*len(prompt_ids) + answer_ids + [-100]
        input_ids = input_ids[:max_length]; labels = labels[:max_length]
        pad_len = max_length - len(input_ids)
        if pad_len:
            input_ids += [tok.pad_token_id]*pad_len
            labels    += [-100]*pad_len
        attention_mask = [1 if t != tok.pad_token_id else 0 for t in input_ids]
        return input_ids, labels, attention_mask

    default_prefix, default_K = _task_defaults(dsname)

    if default_prefix == "bool":
        tf = _extract_bool_answer(example)
        if tf is None: return None
        labels_for_fmt = [" true", " false"]
        instr = _ensure_prompt_matches_labels(instr, labels_for_fmt)
        answer_text = " true" if tf else " false"
        prompt = _append_answer_stub(instr)
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        answer_ids = tok.encode(answer_text, add_special_tokens=False)
        keep_ans = min(len(answer_ids), max_length - 1)
        answer_ids = answer_ids[-keep_ans:]
        room = max(0, max_length - keep_ans - 1)
        prompt_ids = prompt_ids[-room:]
        input_ids = prompt_ids + answer_ids + [tok.eos_token_id]
        labels    = [-100]*len(prompt_ids) + answer_ids + [-100]
        input_ids = input_ids[:max_length]; labels = labels[:max_length]
        pad_len = max_length - len(input_ids)
        if pad_len:
            input_ids += [tok.pad_token_id]*pad_len
            labels    += [-100]*pad_len
        attention_mask = [1 if t != tok.pad_token_id else 0 for t in input_ids]
        return input_ids, labels, attention_mask

    pair = _extract_indexed_answer(example, prefixes=["answer","solution","ending","option"])
    if pair is not None:
        _, idx = pair
        K_eff = default_K if (isinstance(default_K, int) and default_K >= 2) else max(2, idx)
        labels_set = _pick_discriminative_labels(tok, K_eff, preferred_prefix=None)
        instr = _ensure_prompt_matches_labels(instr, labels_set)
        answer_text = labels_set[idx - 1]
        prompt = _append_answer_stub(instr)
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        answer_ids = tok.encode(answer_text, add_special_tokens=False)
        keep_ans = min(len(answer_ids), max_length - 1)
        answer_ids = answer_ids[-keep_ans:]
        room = max(0, max_length - keep_ans - 1)
        prompt_ids = prompt_ids[-room:]
        input_ids = prompt_ids + answer_ids + [tok.eos_token_id]
        labels    = [-100]*len(prompt_ids) + answer_ids + [-100]
        input_ids = input_ids[:max_length]; labels = labels[:max_length]
        pad_len = max_length - len(input_ids)
        if pad_len:
            input_ids += [tok.pad_token_id]*pad_len
            labels    += [-100]*pad_len
        attention_mask = [1 if t != tok.pad_token_id else 0 for t in input_ids]
        return input_ids, labels, attention_mask

    ans = example.get("answer", example.get("output", ""))
    if ans is None: return None
    answer_text = " " + str(ans).strip()
    instr_open = re.sub(r"(?im)^.*\banswer\s*format\s*:\s*.*$", "", instr).strip()
    prompt = _append_answer_stub(instr_open)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    answer_ids = tok.encode(answer_text, add_special_tokens=False)
    keep_ans = min(len(answer_ids), max_length - 1)
    answer_ids = answer_ids[-keep_ans:]
    room = max(0, max_length - keep_ans - 1)
    prompt_ids = prompt_ids[-room:]
    input_ids = prompt_ids + answer_ids + [tok.eos_token_id]
    labels    = [-100]*len(prompt_ids) + answer_ids + [-100]
    input_ids = input_ids[:max_length]; labels = labels[:max_length]
    pad_len = max_length - len(input_ids)
    if pad_len:
        input_ids += [tok.pad_token_id]*pad_len
        labels    += [-100]*pad_len
    attention_mask = [1 if t != tok.pad_token_id else 0 for t in input_ids]
    return input_ids, labels, attention_mask

def build_hf_dataset(records: List[Dict[str, Any]], dsname: str, max_length: int, tok):
    rows, skipped = [], 0
    for ex in records:
        try:
            pair = _build_prompt_and_answer(ex, dsname, max_length, tok)
            if pair is None:
                skipped += 1
                continue
            input_ids, labels, attention_mask = pair
            rows.append({"input_ids": torch.tensor(input_ids, dtype=torch.long),
                         "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                         "labels": torch.tensor(labels, dtype=torch.long)})
        except Exception:
            skipped += 1
            continue
    if skipped:
        print(f"[DATA] Skipped {skipped} malformed rows for task='{dsname}'. Kept {len(rows)} rows.")
    if not rows:
        raise RuntimeError(f"No usable rows after formatting for task '{dsname}'.")
    return HFDataset.from_list(rows)

def _check_lengths(ds, name):
    for fld in ("input_ids", "labels", "attention_mask"):
        lens = {len(r[fld]) for r in ds}
        assert lens == {args.max_length}, f"{name}.{fld} has lengths {lens}"

# ───────────────────────── Tokenizer ───────────────────────── #
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ─────────────────────── Load & build datasets ─────────────────────── #
if is_numglue:
    tr_recs, te_recs = _load_numglue_train_test(args.dataset_root, task_name)
    train_dataset = build_hf_dataset(tr_recs, task_name, args.max_length, tok)
    val_dataset   = build_hf_dataset(te_recs, task_name, args.max_length, tok)
elif is_commonsense:
    tr_recs, te_recs = _load_commonsense_train_test(args.dataset_root, task_name)
    train_dataset = build_hf_dataset(tr_recs, task_name, args.max_length, tok)
    val_dataset   = build_hf_dataset(te_recs, task_name, args.max_length, tok)
else:
    all_records = _load_task_records_from_dir(args.dataset_root, task_name)
    tr_recs, va_recs = _split_train_val(all_records, val_ratio=0.1, seed=_SPLIT_SEED)
    train_dataset = build_hf_dataset(tr_recs, task_name, args.max_length, tok)
    val_dataset   = build_hf_dataset(va_recs, task_name, args.max_length, tok)

_check_lengths(train_dataset, "train_dataset")
_check_lengths(val_dataset,   "val_dataset")
train_val = DatasetDict({"train": train_dataset, "validation": val_dataset})

# ─────────────────────── Step 3.1: compute forced scheme ───────────────────── #
from local_model_utilities import make_forced_labels_scheme

if is_math:
    combined_for_scheme = tr_recs + va_recs
else:
    combined_for_scheme = tr_recs + te_recs

prefix_used, K_used = _prefix_and_K_from_records_or_defaults(combined_for_scheme, task_name)

# Prefer enumerated labels if present in any record's instruction
enum_forced = None
for _rec in combined_for_scheme:
    _instr = _rec.get("instruction", "")
    _enum = _extract_enumerated_labels(_instr)
    if _enum:
        enum_forced = _ensure_discriminative_from_list(tok, _enum)
        break

if enum_forced:
    forced_scheme = make_forced_labels_scheme(tok, enum_forced)
else:
    if prefix_used == "bool":
        forced_scheme = make_forced_labels_scheme(tok, [" true", " false"])
    elif prefix_used == "open":
        forced_scheme = None
    else:
        if not isinstance(K_used, int) or K_used < 2:
            K_used = COMMONSENSE_REGISTRY.get(task_name, {}).get("classes") or 5
        forced_labels = _pick_discriminative_labels(tok, K_used, preferred_prefix=prefix_used)
        forced_scheme = make_forced_labels_scheme(tok, forced_labels)

# ───────────────────────── Model helpers ───────────────────────── #
def _count_proj_layers(model, proj_names=("q_proj", "k_proj", "v_proj", "o_proj")):
    if hasattr(model, "layers"):
        cand = model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        cand = model.model.layers
    else:
        raise RuntimeError("Could not locate transformer layers.")
    n = 0
    for lyr in cand:
        if hasattr(lyr, "self_attn"):
            for name in proj_names:
                if hasattr(lyr.self_attn, name) and getattr(lyr.self_attn, name) is not None:
                    n += 1
    return n

def make_sparse_from_linear(
    lin: nn.Linear,
    *,
    k: int,
    full_grad_every: int = 5,
    mask_mode: str = "topk",
    track_grad_norm_prop: bool = False,
    use_dynamic_k: bool = False,
    k_initial: Optional[int] = None,
    k_final: Optional[int] = None,
    num_decay_steps: int = 10_000,
) -> "SparseUpdateLinear":
    if not use_dynamic_k:
        k_initial = k if k_initial is None else k_initial
        k_final   = k if k_final is None else k_final
    new_layer = SparseUpdateLinear(
        old_linear=lin,
        k=k,
        use_dynamic_k=use_dynamic_k,
        k_initial=k_initial,
        k_final=k_final,
        num_decay_steps=num_decay_steps,
        mask_mode=mask_mode,
        track_grad_norm_prop=track_grad_norm_prop,
        full_grad_every=full_grad_every,
    ).to(lin.weight.device).to(lin.weight.dtype)
    return new_layer

def _fmt_params(n):
    return f"{n/1e9:.2f}B" if n >= 1e9 else (f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.0f}K")

def _print_trainable_summary(model, tag: str):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (100.0 * trainable / max(1, total))
    print(f"[SANITY][{tag}] trainable={_fmt_params(trainable)} / total={_fmt_params(total)} ({pct:.2f}%)")

# ─────────────────────────── Logging helpers ─────────────────────────── #
def _print_modified_layers(modified: list[str], tag: str = ""):
    if not modified:
        print(f"[FT]{f'[{tag}]' if tag else ''} No layers were wrapped/replaced.")
        return
    cap = int(os.getenv("GASDU_LOG_LAYER_MAX", "24"))
    print(f"[FT]{f'[{tag}]' if tag else ''} Wrapped/replaced {len(modified)} layers:")
    for line in modified[:cap]:
        print("   •", line)
    if len(modified) > cap:
        print(f"   • ... (+{len(modified)-cap} more). Set GASDU_LOG_LAYER_MAX to view all.")

# ───────────────── Checkpointing policy (centralized) ───────────────── #
def apply_checkpointing_policy(base):
    try:
        base.config.use_cache = False
    except Exception:
        pass
    try:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        base.gradient_checkpointing_enable()
    except Exception:
        pass
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()

def modify_model_for_gasdu(
    model, *,
    k_val: Optional[int] = None,
    update_percent: float = 0.1,
    use_dynamic_k: bool = False,
    mask_mode: str = "topk",
    track_grad_norm_prop: bool = False,
    full_grad_every: int = 5
):
    try:
        apply_checkpointing_policy(model)
    except Exception:
        pass

    modified_layers: list[str] = []

    for p in model.parameters(): 
        p.requires_grad = False

    if hasattr(model, "layers"):
        candidate_layers = model.layers
        root_prefix = "layers"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        candidate_layers = model.model.layers
        root_prefix = "model.layers"
    else:
        raise RuntimeError("Could not find transformer layers.")

    proj_names = ("q_proj", "k_proj", "v_proj", "o_proj")
    num_proj_layers = _count_proj_layers(model, proj_names=proj_names)

    if k_val is None:
        total_params = sum(p.numel() for p in model.parameters())
        params_to_update = int(total_params * (update_percent / 100.0))
        k_val = max(1, params_to_update // max(1, num_proj_layers))

    def gasdu_layer(old_linear: nn.Linear) -> nn.Module:
        return make_sparse_from_linear(
            old_linear,
            k               = k_val,
            full_grad_every = full_grad_every,
            mask_mode       = mask_mode,
            track_grad_norm_prop = track_grad_norm_prop,
            use_dynamic_k   = use_dynamic_k,
            k_initial       = k_val,
            k_final         = k_val,
        )

    for li, lyr in enumerate(candidate_layers):
        if not hasattr(lyr, "self_attn"):
            continue
        for proj_name in proj_names:
            if not hasattr(lyr.self_attn, proj_name):
                continue
            lin = getattr(lyr.self_attn, proj_name)
            if lin is None:
                continue
            setattr(lyr.self_attn, proj_name, gasdu_layer(lin))
            modified_layers.append(
                f"{root_prefix}[{li}].self_attn.{proj_name} -> SparseUpdateLinear(k={k_val}, mode={mask_mode})"
            )

    model.train()
    _print_trainable_summary(model, "GASDU")

    setattr(model, "_ft_modified_layers", modified_layers)
    _print_modified_layers(modified_layers, "GASDU")

    return model

# ───────────────────────── Lightning wrapper ───────────────────────── #
from local_model_utilities import CustomLightningModule

class GenericDataset(Dataset):
    def __init__(self, ds, split):
        self.data = ds[split]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return {k: torch.tensor(v) for k, v in self.data[idx].items()}

class ForceTrainMode(Callback):
    def on_fit_start(self, trainer, pl_module):
        if hasattr(pl_module, "model"): pl_module.model.train()
    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(pl_module, "model"): pl_module.model.train()

class TrackBestValAcc(Callback):
    """Track best validation accuracy in-memory (no checkpointing). For NumGLUE, this is Exact Match (EM)."""
    def __init__(self, monitor: str = "val_acc", mode: str = "max"):
        self.monitor = monitor
        self.mode = mode
        self.best: Optional[float] = None
    def on_validation_epoch_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get(self.monitor, None)
        if val is None:
            return
        try:
            v = float(val.detach().cpu().item()) if hasattr(val, "detach") else float(val)
        except Exception:
            v = float(val)
        if self.best is None:
            self.best = v
        else:
            if (self.mode == "max" and v > self.best) or (self.mode == "min" and v < self.best):
                self.best = v

# ───── Training-only throughput meter (samples/sec over train steps) ───── #
class ThroughputMeter(Callback):
    def __init__(self, device=None):
        self.device = device
        self.total_samples = 0
        self.total_time = 0.0
        self._t0 = None
    def on_fit_start(self, trainer, pl_module):
        self.total_samples = 0
        self.total_time = 0.0
        self._t0 = None
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.device is not None and torch.cuda.is_available():
            try: torch.cuda.synchronize(self.device)
            except Exception: pass
        self._t0 = time.perf_counter()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._t0 is None:
            return
        if self.device is not None and torch.cuda.is_available():
            try: torch.cuda.synchronize(self.device)
            except Exception: pass
        dt = time.perf_counter() - self._t0
        try:
            bs = int(batch["input_ids"].size(0))
        except Exception:
            bs = getattr(getattr(trainer, "train_dataloader", None), "batch_size", 0) or 0
        self.total_samples += bs
        self.total_time += dt
        self._t0 = None
    @property
    def samples_per_sec(self) -> float:
        return float(self.total_samples) / max(self.total_time, 1e-12)

# ───── Per-step median grad-norm proportion logger (GASDU layers only) ──── #
class GradRatioLogger(Callback):
    """
    After backward (before optimizer step) compute the **median** grad-norm proportion
    across all SparseUpdateLinear modules and log to file.
    """
    def __init__(self, path: Optional[str], enabled: bool):
        self.path = path
        self.enabled = bool(enabled)

    def on_fit_start(self, trainer, pl_module):
        if not (self.enabled and self.path): return
        if getattr(trainer, "is_global_zero", True):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("step\tmedian_grad_ratio\n")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not (self.enabled and self.path): return
        if not getattr(trainer, "is_global_zero", True): return
        model = getattr(pl_module, "model", pl_module)
        ratios: List[float] = []
        for m in model.modules():
            if isinstance(m, SparseUpdateLinear):
                r = getattr(m, "latest_ratio", None)
                if r is not None:
                    try:
                        ratios.append(float(r))
                    except Exception:
                        pass
        if ratios:
            med = float(np.median(ratios))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{trainer.global_step}\t{med:.8f}\n")

def _seed_worker(worker_id):
    worker_seed = (args.seed + worker_id) % (2**32 - 1)
    np.random.seed(worker_seed); random.seed(worker_seed)

def _make_loader(ds, batch_size, shuffle):
    nw = min(4, max(2, (os.cpu_count() or 8) // 4))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0), worker_init_fn=_seed_worker,
        drop_last=False,
    )

# ───────────────────────── Stage-1 (grid) ───────────────────────── #
STAGE1_PATH = os.path.join(out_dir, f"stage1_results_{args.full_grad_every}_{args.update_percent}.txt")

def stage1_grid_search():
    best_combo, best_val = None, -1.0
    monitor_key = "val_acc"  # For NumGLUE, this key reports EM internally.
    GRID_LRS = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    GRID_BATCH = [4, 8]
    with open(STAGE1_PATH, "w", encoding="utf-8") as f:
        f.write("# Stage-1 grid search results\n")
        f.write(f"# model={model_name}  task={task_name}  method=GASDU  mask_mode={args.mask_mode}  max_epoch=1\n")
        f.write("lr\tbatch_size\tval_acc_pct(EM)\tavg_epoch_min\tpeak_gpu_gb\ttrain_samples_per_sec\n")

        for lr in GRID_LRS:
            for bs in GRID_BATCH:
                print(f"[Stage-1] lr={lr:.2g}  batch={bs}")

                set_random_seed(args.seed)
                base = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map={"": SELECTED_GPU} if SELECTED_GPU is not None else None,
                    torch_dtype=torch.bfloat16
                )

                model_ft = modify_model_for_gasdu(
                    base,
                    update_percent=args.update_percent,
                    use_dynamic_k=args.use_dynamic_k,
                    mask_mode=args.mask_mode,
                    track_grad_norm_prop=False,
                    full_grad_every=args.full_grad_every,
                )

                lit = CustomLightningModule(
                    math_numeric_eval=(is_math or is_numglue),
                    max_gen_new_tokens=32,
                    model=model_ft, tokenizer=tok, learning_rate=lr, task_name=task_name, forced_scheme=forced_scheme,
                    # SFT not used in GASDU
                    sft_enabled=False,
                    sft_use_sm3=False,
                    sft_total_train_steps=None,
                    sft_grad_accum=1,
                    sft_peft_config=None,
                )

                ds_tr = GenericDataset(train_val, "train")
                ds_val = GenericDataset(train_val, "validation")
                tr_loader  = _make_loader(ds_tr,  bs, shuffle=True)
                val_loader = _make_loader(ds_val, bs, shuffle=False)

                logger_dir = os.path.join(out_dir, "stage1_csv_logs")
                os.makedirs(logger_dir, exist_ok=True)
                logger = CSVLogger(save_dir=logger_dir, name="", version=None)
                best_cb = TrackBestValAcc(monitor=monitor_key, mode="max")
                tp_meter = ThroughputMeter(device=device)

                trainer = L.Trainer(
                    max_epochs=1, min_epochs=1, check_val_every_n_epoch=1, num_sanity_val_steps=0,
                    accelerator="gpu" if SELECTED_GPU is not None else "cpu",
                    devices=[SELECTED_GPU] if SELECTED_GPU is not None else 1,
                    precision="bf16-mixed", gradient_clip_val=1.0,
                    callbacks=[ForceTrainMode(), best_cb, tp_meter], logger=logger, log_every_n_steps=200,
                    strategy=SingleDeviceStrategy(device=device) if SELECTED_GPU is not None else None,
                    enable_checkpointing=False,
                )

                _reset_gpu_peak()
                start = time.time()

                trainer.fit(lit, tr_loader, val_loader)

                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                except Exception:
                    pass
                elapsed_min = (time.time() - start) / 60.0
                avg_epoch_min = elapsed_min / 1.0
                peak_gb = _peak_gpu_gb()

                val_best = best_cb.best if best_cb.best is not None else 0.0
                val_score = float(val_best) * 100.0  # % EM on NumGLUE
                train_sps = tp_meter.samples_per_sec

                f.write(f"{lr:.8g}\t{bs}\t{val_score:.4f}\t{avg_epoch_min:.3f}\t{peak_gb:.3f}\t{train_sps:.3f}\n"); f.flush()

                if val_score > best_val:
                    best_val = val_score
                    best_combo = (lr, bs)

                del trainer, logger, best_cb, tp_meter, lit, model_ft, base
                torch.cuda.empty_cache()

        if best_combo is not None:
            f.write(f"BEST\t{best_combo[0]:.8g}\t{best_combo[1]}\t{best_val:.4f}\n")
    return best_combo, best_val

perform_stage1 = not args.stage2_only
best_lr, best_bs = args.learning_rate, args.batch_size
best_val_stage1 = 0.0
if perform_stage1:
    best_hparams, best_val_stage1 = stage1_grid_search()
    if best_hparams:
        best_lr, best_bs = best_hparams
print("─"*46)
print(f"Stage-1 best ⇒ LR={best_lr}, BATCH={best_bs}, val(EM)={best_val_stage1:.2f}")
print("─"*46)
if args.stage1_only:
    sys.exit(0)

# ───────────────────────── Stage-2 training ───────────────────────── #
STAGE2_PATH = os.path.join(out_dir, f"stage2_results_{args.full_grad_every}_{args.update_percent}.txt")

if args.stage2_seeds.strip():
    try:
        seeds = [int(s) for s in re.split(r"[,\s]+", args.stage2_seeds.strip()) if s]
    except Exception:
        print("[WARN] --stage2_seeds parse failed; falling back to defaults [42, 43, 233]")
        seeds = [42, 43, 233]
else:
    seeds = [42, 43, 233]

monitor_key = "val_acc"  # For NumGLUE, treat as EM
with open(STAGE2_PATH, "w", encoding="utf-8") as f2:
    f2.write(f"Using LR={best_lr}, BATCH={best_bs}\nseed\tval_em_pct\tavg_epoch_min\tpeak_gpu_gb\ttrain_samples_per_sec\n")
    best_val2, best_seed = -1.0, None
    for sd in seeds:
        print(f"[Stage-2] seed={sd}")
        set_random_seed(sd)

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": SELECTED_GPU} if SELECTED_GPU is not None else None,
            torch_dtype=torch.bfloat16
        )

        model_ft = modify_model_for_gasdu(
            base,
            update_percent=args.update_percent,
            use_dynamic_k=args.use_dynamic_k,
            mask_mode=args.mask_mode,
            track_grad_norm_prop=args.track_grad_norm_prop,
            full_grad_every=args.full_grad_every,
        ).to(device)

        lit = CustomLightningModule(
            math_numeric_eval=(is_math or is_numglue),
            max_gen_new_tokens=32,
            model=model_ft, tokenizer=tok, learning_rate=best_lr, task_name=task_name, forced_scheme=forced_scheme,
            sft_enabled=False,
            sft_use_sm3=False,
            sft_total_train_steps=None,
            sft_grad_accum=1,
            sft_peft_config=None,
        )

        ds_tr = GenericDataset(train_val, "train")
        ds_val = GenericDataset(train_val, "validation")
        tr_loader  = _make_loader(ds_tr,  best_bs, shuffle=True)
        val_loader = _make_loader(ds_val, best_bs, shuffle=False)

        logger_dir = os.path.join(out_dir, f"stage2_seed{sd}_csv_logs")
        os.makedirs(logger_dir, exist_ok=True)
        logger = CSVLogger(save_dir=logger_dir, name="", version=None)
        best_cb = TrackBestValAcc(monitor=monitor_key, mode="max")
        tp_meter = ThroughputMeter(device=device)

        grad_ratio_path = None
        if args.track_grad_norm_prop:
            grad_ratio_path = os.path.join(
                out_dir,
                f"grad_ratio_{task_name}_{out_basename}_{method_tag}_{args.mask_mode}_seed{sd}_{args.full_grad_every}_{args.update_percent}.txt"
            )
        ratio_cb = GradRatioLogger(grad_ratio_path, enabled=bool(args.track_grad_norm_prop))

        trainer = L.Trainer(
            max_epochs=args.max_epoch, min_epochs=args.max_epoch, check_val_every_n_epoch=1, num_sanity_val_steps=0,
            accelerator="gpu" if SELECTED_GPU is not None else "cpu",
            devices=[SELECTED_GPU] if SELECTED_GPU is not None else 1,
            precision="bf16-mixed", gradient_clip_val=1.0,
            callbacks=[ForceTrainMode(), best_cb, tp_meter] + ([ratio_cb] if args.track_grad_norm_prop else []),
            logger=logger, log_every_n_steps=200,
            strategy=SingleDeviceStrategy(device=device) if SELECTED_GPU is not None else None,
            enable_checkpointing=False,
        )

        _reset_gpu_peak()
        start = time.time()

        trainer.fit(lit, tr_loader, val_loader)

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
        except Exception:
            pass
        elapsed = (time.time() - start) / 60.0
        avg_epoch_min = elapsed / max(1, args.max_epoch)
        peak_gb = _peak_gpu_gb()

        best_val_seed = best_cb.best if best_cb.best is not None else 0.0
        val_score = float(best_val_seed) * 100.0  # % EM on NumGLUE
        print(f"seed={sd} ⇒ EM={val_score:.2f}%   (time: {elapsed:.1f} min)")

        train_sps = tp_meter.samples_per_sec

        f2.write(f"{sd}\t{val_score:.4f}\t{avg_epoch_min:.3f}\t{peak_gb:.3f}\t{train_sps:.3f}\n"); f2.flush()

        if val_score > best_val2:
            best_val2, best_seed = val_score, sd

        del trainer, logger, best_cb, tp_meter, ratio_cb, lit, model_ft, base
        torch.cuda.empty_cache()

print("═"*44)
print(f"BEST seed = {best_seed} • EM = {best_val2:.2f}%")
print("All Stage-2 checkpoints were skipped (no weights saved).")
print("═"*44)
