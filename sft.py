#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blur-OCR Evaluation Script
===========================
논문(ICLR 2026) Section 3-4 기준 평가 코드.

주요 수정 사항 (논문 대조):
  1. Gap 분모: pred-count 기반 → GT-length 기반 (|y|_in,L / |y|_out,L)
     - equal/replace/insert 토큰만 GT length에 기여
     - delete(pred extra) 토큰은 ErrIn/ErrOut 증가, GT length 미포함
  2. char-level insert implicit coverage: 논문 정의(src_pos 양쪽 mask 확인)
  3. word-level insert implicit coverage: 앞/뒤 aligned pred word 순서대로 탐색
  4. n/a 처리: |y|_in=0 or |y|_out=0 → e_in/e_out=None → Gap=None

데이터 경로:
  - val jsonl:  /nas/datahub/Blur-OCR/Blur-OCR/SFT/val/cold_start_val_{char|word}_level.jsonl
  - val images: /nas/datahub/Blur-OCR/Blur-OCR/val/images/
  - raw GT:     /nas/datahub/Blur-OCR/Blur-OCR/val/labels.txt

출력:
  - predictions/sample_results.jsonl
  - metrics/eval_summary.json
"""

import os
import re
import json
import time
import argparse
import datetime
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Any

import sys
sys.path.append("/nas/home/ongv1109/prj_ocr_uncertainty/TeachingLLMs/only_sft/utils")

from rapidfuzz.distance import Levenshtein as Lev
from ocr_gt_align import mark_gt_errors

# ============================================================================
# 경로 설정
# ============================================================================

BASE_MODEL_PATH = (
    "/nas/home/ongv1109/prj_ocr_uncertainty/TeachingLLMs/.cache/huggingface/"
    "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
SFT_CHAR_PATH = ""   # 실행 시 --sft_char_model로 지정
SFT_WORD_PATH = ""   # 실행 시 --sft_word_model로 지정

DATA_DIR   = "/nas/datahub/Blur-OCR/Blur-OCR/val"
OUTPUT_DIR = "/nas/datahub/Blur-OCR/results/eval_only_sft"

# BASE prediction 고정 저장 경로 (한 번만 inference 후 재사용)
BASE_PRED_CACHE = "/nas/datahub/Blur-OCR/results/base_predictions/BASE_predictions.json"

# ============================================================================
# 프롬프트
# ============================================================================

PLAIN_SYSTEM_PROMPT = (
    "You are an intelligent OCR system. "
    "Perform OCR to transcribe all text from the image."
)
GENERAL_SYSTEM_PROMPT = (
    "You are an intelligent OCR system.\n"
    "Perform OCR to transcribe all text from the image.\n"
    "When you encounter characters or words you are not confident "
    "about, wrap them with <C>...</C>.\n"
    "You may use <C>...</C> multiple times. Do not nest or overlap <C> regions.\n"
    "Use the smallest span that covers the uncertainty. Preserve "
    "natural reading order and basic line breaks. "
    "Keep the original language, punctuation, and casing. "
    "Output only the final text (no explanations, no code blocks, no quotes)."
)
OURS_SYSTEM_PROMPT = (
    "You are an intelligent OCR system.\n"
    "Perform OCR to transcribe all text from the image."
)

UNC_START   = "<C>"
UNC_END     = "</C>"
UNC_PATTERN = re.compile(re.escape(UNC_START) + r"(.*?)" + re.escape(UNC_END), re.DOTALL)
VALID_EXT   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ============================================================================
# vLLM inference worker
# ============================================================================

def run_inference_worker(args_dict: dict):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args_dict["gpus"]

    from vllm import LLM, SamplingParams

    model_key     = args_dict["model_key"]
    model_path    = args_dict["model_path"]
    system_prompt = args_dict["system_prompt"]
    greedy        = args_dict["greedy"]
    image_paths   = args_dict["image_paths"]
    output_path   = args_dict["output_path"]
    batch_size    = args_dict.get("batch_size", 8)

    print(f"[{model_key}] Loading {model_path} on GPU {args_dict['gpus']}")
    media_root = str(Path(image_paths[0]).parent.parent) if image_paths else "/"

    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=media_root,
    )
    sampling_params = SamplingParams(
        temperature=0.0 if greedy else 0.2,
        top_p=1.0,
        max_tokens=2048,
    )

    results = {}
    total = len(image_paths)

    for i in range(0, total, batch_size):
        batch = image_paths[i:i + batch_size]
        messages_batch = [[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"file://{p}"}},
                {"type": "text", "text": "Extract all text from this image."},
            ]},
        ] for p in batch]

        try:
            outputs = llm.chat(messages_batch, sampling_params=sampling_params, use_tqdm=False)
            for p, out in zip(batch, outputs):
                results[p] = out.outputs[0].text.strip()
        except Exception as e:
            print(f"[{model_key}] Batch {i // batch_size} error: {e}")
            for p in batch:
                try:
                    out = llm.chat([[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"file://{p}"}},
                            {"type": "text", "text": "Extract all text from this image."},
                        ]},
                    ]], sampling_params=sampling_params, use_tqdm=False)
                    results[p] = out[0].outputs[0].text.strip()
                except Exception as e2:
                    print(f"[{model_key}] Single retry failed {p}: {e2}")
                    results[p] = ""

        if (i // batch_size) % 10 == 0:
            print(f"[{model_key}] {min(i + batch_size, total)}/{total} done")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[{model_key}] Saved → {output_path}")


def launch_inference_parallel(model_configs: dict, image_paths: List[str],
                               tmp_dir: str, batch_size: int = 8) -> Dict[str, str]:
    """
    model_configs: {model_key: (model_path, system_prompt, greedy, gpus)}
    """
    processes    = []
    output_paths = {}

    for model_key, (model_path, system_prompt, greedy, gpus) in model_configs.items():
        output_path = os.path.join(tmp_dir, f"{model_key}_predictions.json")
        output_paths[model_key] = output_path
        args_dict = dict(
            model_key=model_key, model_path=model_path,
            system_prompt=system_prompt, greedy=greedy,
            gpus=gpus, image_paths=image_paths,
            output_path=output_path, batch_size=batch_size,
        )
        p = mp.Process(target=run_inference_worker, args=(args_dict,))
        p.start()
        processes.append((model_key, p))
        print(f"[LAUNCH] {model_key} on GPU {gpus}")

    for model_key, p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"[ERROR] {model_key} exitcode={p.exitcode}")
        else:
            print(f"[DONE]  {model_key}")

    return output_paths


# ============================================================================
# tagged_gt 생성
# ============================================================================

def build_tagged_gts(base_plain_preds: Dict[str, str],
                     raw_gts: Dict[str, str]) -> Dict[str, str]:
    tagged_gts = {}
    for img_path, raw_gt in raw_gts.items():
        base_pred = base_plain_preds.get(img_path, "")
        tagged_gts[img_path] = mark_gt_errors(
            ocr_output=base_pred,
            ground_truth=raw_gt,
            marking_level="word",
            verbose=False,
        )
    return tagged_gts


# ============================================================================
# 텍스트 정규화 / 마스크 추출
# ============================================================================

def normalize_text(text: str) -> str:
    import uuid
    ps = f"__PH_S_{uuid.uuid4().hex}__"
    pe = f"__PH_E_{uuid.uuid4().hex}__"
    text = text.replace('<C>', ps).replace('</C>', pe)
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.lower()
    text = ' '.join(text.split())
    text = text.replace(ps.lower(), '<C>').replace(pe.lower(), '</C>')
    return text


def extract_clean_and_mask(text: str) -> Tuple[str, np.ndarray, List[Tuple[int, int]]]:
    keep = [True] * len(text)
    i, last_open = 0, -1
    while i < len(text):
        if text[i:i + 3] == UNC_START:
            if last_open != -1:
                for j in range(last_open, last_open + 3):
                    keep[j] = False
            last_open = i
            i += 3
        elif text[i:i + 4] == UNC_END:
            if last_open == -1:
                for j in range(i, i + 4):
                    keep[j] = False
            else:
                last_open = -1
            i += 4
        else:
            i += 1
    if last_open != -1:
        for j in range(last_open, min(last_open + 3, len(keep))):
            keep[j] = False

    cleaned = "".join(text[k] for k, v in enumerate(keep) if v)

    regions, offset = [], 0
    for m in UNC_PATTERN.finditer(cleaned):
        s = m.start() - offset
        e = s + len(m.group(1))
        regions.append((s, e))
        offset += len(UNC_START) + len(UNC_END)

    clean_text = UNC_PATTERN.sub(r"\1", cleaned)
    mask = np.zeros(len(clean_text), dtype=bool)
    for s, e in regions:
        if 0 <= s < e <= len(mask):
            mask[s:e] = True

    return clean_text, mask, regions


def tokenize_words(text: str) -> List[Tuple[str, int, int]]:
    words, i = [], 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        start = i
        while i < len(text) and not text[i].isspace():
            i += 1
        words.append((text[start:i], start, i))
    return words


# ============================================================================
# 핵심 메트릭 계산 ── 논문 Section 3-4
# ============================================================================

def compute_metrics(pred_text: str, raw_gt: str, tagged_gt: str) -> Dict[str, Any]:
    """
    논문 Section 3-4 기준 메트릭 계산.

    Accuracy : clean_pred vs raw_gt (Levenshtein)
    Precision : ErrIn / (ErrIn + CorrectIn)
    Recall    : ErrIn / (ErrIn + ErrOut)
    F1        : 2·P·R / (P+R)
    Gap       : e_in - e_out
                  e_in  = ErrIn  / |y|_in,L   (GT length 기반, 논문 수식)
                  e_out = ErrOut / |y|_out,L  (GT length 기반)
                  |y|_in / |y|_out: equal·replace·insert 기여, delete(extra pred) 미기여
    n/a : |y|_in=0 or |y|_out=0 → e_in/e_out=None → Gap=None
    """
    raw_gt_norm = normalize_text(raw_gt)
    pred_norm   = normalize_text(pred_text)

    clean_pred, pred_mask, pred_regions = extract_clean_and_mask(pred_norm)
    clean_raw_gt = raw_gt_norm

    # ── Accuracy ─────────────────────────────────────────────────────────
    w_gt   = tokenize_words(clean_raw_gt)
    w_pred = tokenize_words(clean_pred)
    word_ed       = len(Lev.editops([w[0] for w in w_pred], [w[0] for w in w_gt]))
    word_accuracy = 1.0 - word_ed / max(1, len(w_gt))
    char_ed       = len(Lev.editops(clean_pred, clean_raw_gt))
    char_accuracy = 1.0 - char_ed / max(1, len(clean_raw_gt))

    # ── UNC stats helper ─────────────────────────────────────────────────
    def unc_stats(in_e, in_c, out_e, out_c, gt_in, gt_out):
        """
        in_e, in_c, out_e, out_c : pred-count 기반 (P/R/F1 용)
        gt_in, gt_out            : GT-length 기반 (Gap 용, 논문 수식)
        """
        total_in  = in_e + in_c
        total_out = out_e + out_c
        precision = in_e / total_in            if total_in       > 0 else 0.0
        recall    = in_e / (in_e + out_e)      if (in_e + out_e) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        e_in  = in_e  / gt_in  if gt_in  > 0 else None
        e_out = out_e / gt_out if gt_out > 0 else None
        gap   = (e_in - e_out) if (e_in is not None and e_out is not None) else None
        return precision, recall, f1, gap

    # ════════════════════════════════════════════════════════════════════
    # Word-level UNC (논문 Section 3 word-level membership rule)
    # ════════════════════════════════════════════════════════════════════
    w_opcodes = Lev.opcodes([w[0] for w in w_pred], [w[0] for w in w_gt])

    # alignment: (gt_idx, pred_idx, is_correct)
    # pred_idx = -1 → insert (GT에만 있는 누락 word)
    w_alignment = []
    pred_w_used = [False] * len(w_pred)

    for op in w_opcodes:
        if op.tag == "equal":
            for k in range(op.src_start, op.src_end):
                gi = op.dest_start + (k - op.src_start)
                w_alignment.append((gi, k, True))
                pred_w_used[k] = True
        elif op.tag == "replace":
            for k in range(op.src_start, op.src_end):
                gi = op.dest_start + (k - op.src_start)
                w_alignment.append((gi, k, False))
                pred_w_used[k] = True
        elif op.tag == "insert":
            for gi in range(op.dest_start, op.dest_end):
                w_alignment.append((gi, -1, False))
        # "delete" (pred에만 있는 extra word): alignment에 추가 안 함 → GT length 미기여

    def w_pred_in_unc(pred_idx: int) -> bool:
        if pred_idx < 0 or pred_idx >= len(w_pred):
            return False
        ws, we = w_pred[pred_idx][1], w_pred[pred_idx][2]
        return (we <= len(pred_mask)) and bool(np.any(pred_mask[ws:we]))

    def w_insert_in_unc(align_pos: int) -> bool:
        """
        insert(pred 누락) word의 implicit coverage 판단.
        논문 Section 3: 앞/뒤 aligned pred word 중 하나라도 UNC 안이면 inside.
        """
        if len(pred_mask) == 0:
            return False
        check_forward = False
        for i in range(align_pos + 1, len(w_alignment)):
            _, next_pi, _ = w_alignment[i]
            if next_pi != -1:
                check_forward = w_pred_in_unc(next_pi)
                break
        check_backward = False
        for i in range(align_pos - 1, -1, -1):
            _, prev_pi, _ = w_alignment[i]
            if prev_pi != -1:
                check_backward = w_pred_in_unc(prev_pi)
                break
        return check_forward or check_backward

    w_in_e = w_in_c = w_out_e = w_out_c = 0
    w_gt_in = w_gt_out = 0  # |y|_in,word / |y|_out,word (GT length 기반)

    for ai, (gi, pi, is_correct) in enumerate(w_alignment):
        unc = w_insert_in_unc(ai) if pi == -1 else w_pred_in_unc(pi)
        if unc:
            w_gt_in += 1
            if is_correct: w_in_c += 1
            else:          w_in_e += 1
        else:
            w_gt_out += 1
            if is_correct: w_out_c += 1
            else:          w_out_e += 1

    # extra pred words (delete): GT length 기여 없음, ErrIn/Out만 증가
    for pi in range(len(w_pred)):
        if not pred_w_used[pi]:
            unc = w_pred_in_unc(pi)
            if unc: w_in_e  += 1
            else:   w_out_e += 1

    w_prec, w_rec, w_f1, w_gap = unc_stats(
        w_in_e, w_in_c, w_out_e, w_out_c, w_gt_in, w_gt_out
    )

    # ════════════════════════════════════════════════════════════════════
    # Char-level UNC (논문 Section 3 char-level membership rule)
    # ════════════════════════════════════════════════════════════════════
    c_opcodes = Lev.opcodes(clean_pred, clean_raw_gt)

    c_in_e = c_in_c = c_out_e = c_out_c = 0
    c_gt_in = c_gt_out = 0  # |y|_in,char / |y|_out,char (GT length 기반)

    for op in c_opcodes:
        if op.tag == "equal":
            for k in range(op.src_start, op.src_end):
                unc = (k < len(pred_mask)) and bool(pred_mask[k])
                if unc:
                    c_in_c  += 1; c_gt_in  += 1
                else:
                    c_out_c += 1; c_gt_out += 1

        elif op.tag == "replace":
            for k in range(op.src_start, op.src_end):
                unc = (k < len(pred_mask)) and bool(pred_mask[k])
                if unc:
                    c_in_e  += 1; c_gt_in  += 1
                else:
                    c_out_e += 1; c_gt_out += 1

        elif op.tag == "delete":
            # pred에만 있는 extra char → GT length 기여 없음
            for k in range(op.src_start, op.src_end):
                unc = (k < len(pred_mask)) and bool(pred_mask[k])
                if unc: c_in_e  += 1
                else:   c_out_e += 1

        elif op.tag == "insert":
            # GT에만 있는 누락 char → GT length 기여 +1
            # implicit coverage: src_pos 양쪽 mask 확인 (논문 Section 3)
            src_pos = op.src_start
            for _ in range(op.dest_start, op.dest_end):
                if len(pred_mask) == 0:
                    unc = False
                elif src_pos == 0:
                    unc = bool(pred_mask[0])
                elif src_pos >= len(pred_mask):
                    unc = bool(pred_mask[-1])
                else:
                    unc = bool(pred_mask[src_pos]) or bool(pred_mask[src_pos - 1])
                if unc:
                    c_in_e  += 1; c_gt_in  += 1
                else:
                    c_out_e += 1; c_gt_out += 1

    c_prec, c_rec, c_f1, c_gap = unc_stats(
        c_in_e, c_in_c, c_out_e, c_out_c, c_gt_in, c_gt_out
    )

    return {
        "word_accuracy":   round(word_accuracy, 6),
        "char_accuracy":   round(char_accuracy, 6),
        "word_precision":  round(w_prec, 6),
        "word_recall":     round(w_rec,  6),
        "word_f1":         round(w_f1,   6),
        "word_gap":        round(w_gap,  6) if w_gap is not None else None,
        "char_precision":  round(c_prec, 6),
        "char_recall":     round(c_rec,  6),
        "char_f1":         round(c_f1,   6),
        "char_gap":        round(c_gap,  6) if c_gap is not None else None,
        "has_unc":         len(pred_regions) > 0,
        # 디버그용
        "w_in_e": w_in_e, "w_in_c": w_in_c,
        "w_out_e": w_out_e, "w_out_c": w_out_c,
        "w_gt_in": w_gt_in, "w_gt_out": w_gt_out,
        "c_in_e": c_in_e, "c_in_c": c_in_c,
        "c_out_e": c_out_e, "c_out_c": c_out_c,
        "c_gt_in": c_gt_in, "c_gt_out": c_gt_out,
    }


# ============================================================================
# 집계 ── macro average (논문 기준)
# ============================================================================

def aggregate(all_metrics: List[Dict]) -> Dict:
    """macro 평균 (논문 기준). Gap은 n/a(None)인 샘플 제외하여 평균."""
    n = len(all_metrics)
    if n == 0:
        return {}

    def mean(key):
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    return {
        "n":               n,
        "word_accuracy":   mean("word_accuracy"),
        "char_accuracy":   mean("char_accuracy"),
        "word_precision":  mean("word_precision"),
        "word_recall":     mean("word_recall"),
        "word_f1":         mean("word_f1"),
        "word_gap":        mean("word_gap"),
        "char_precision":  mean("char_precision"),
        "char_recall":     mean("char_recall"),
        "char_f1":         mean("char_f1"),
        "char_gap":        mean("char_gap"),
        "has_unc_ratio":   round(sum(1 for m in all_metrics if m["has_unc"]) / n, 6),
        "n_gap_na_word":   sum(1 for m in all_metrics if m.get("word_gap") is None),
        "n_gap_na_char":   sum(1 for m in all_metrics if m.get("char_gap") is None),
    }


# ============================================================================
# 결과 출력
# ============================================================================

PAPER_REF = {
    "word_precision": 0.839, "word_recall": 0.620, "word_f1": 0.685,
    "char_precision": 0.626, "char_recall": 0.599, "char_f1": 0.572,
    "word_accuracy":  0.839, "char_accuracy": 0.881,
    "word_gap":       0.739, "char_gap":      0.694,
}


def print_summary(all_results: Dict):
    W, C = 22, 8

    def fmt(v):
        return f"{v:.3f}" if v is not None else "  n/a "

    def row(name, r, keys):
        return f"{name:<{W}}" + "".join(f"  {fmt(r.get(k)):>{C}}" for k in keys)

    sep = "─" * (W + (C + 2) * 10 + 4)

    keys_word  = ["word_accuracy", "word_precision", "word_recall", "word_f1", "word_gap"]
    short_word = ["w_acc", "w_prec", "w_rec", "w_f1", "w_gap"]
    keys_char  = ["char_accuracy", "char_precision", "char_recall", "char_f1", "char_gap"]
    short_char = ["c_acc", "c_prec", "c_rec", "c_f1", "c_gap"]
    all_keys   = keys_word + keys_char
    all_short  = short_word + short_char
    hdr = f"{'Model':<{W}}" + "".join(f"  {s:>{C}}" for s in all_short)

    n_total = list(all_results.values())[0].get("n", 0)
    print(f"\n[Macro — 전체 {n_total} 샘플, 논문 기준]")
    print(f"{sep}\n{hdr}\n{sep}")
    print(row("[논문] Qwen2.5-VL-7B", PAPER_REF, all_keys))
    print(sep)
    for name, r in all_results.items():
        if r:
            print(row(name, r, all_keys))
    print(sep)

    print(f"\n  {'Model':<{W}} {'has_unc':>10} {'gap_na(w)':>10} {'gap_na(c)':>10}")
    for name, r in all_results.items():
        if r:
            print(f"  {name:<{W}} {r.get('has_unc_ratio', 0):>10.3f}"
                  f" {r.get('n_gap_na_word', 0):>10}"
                  f" {r.get('n_gap_na_char', 0):>10}")
    print(sep)


# ============================================================================
# 데이터 로드 ── labels.txt 기반
# ============================================================================

def load_labels(labels_path: Path) -> Dict[str, str]:
    """labels.txt → {stem: ground_truth}"""
    gt_map = {}
    current_stem  = None
    current_lines = []

    def flush():
        if current_stem is not None:
            gt_map[current_stem] = "\n".join(current_lines).strip()

    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                flush()
                filename, first_text = line.split("\t", 1)
                current_stem  = Path(filename).stem
                current_lines = [first_text]
            else:
                if current_stem is not None:
                    current_lines.append(line)
    flush()
    return gt_map


def load_dataset(data_dir: str) -> Tuple[List[str], Dict[str, str]]:
    """
    val 이미지 경로 리스트, raw_gt dict 반환.
    data_dir 구조:
        {data_dir}/images/  ← 이미지
        {data_dir}/labels.txt ← GT
    """
    image_dir   = os.path.join(data_dir, "images")
    labels_path = Path(data_dir) / "labels.txt"

    gt_map = load_labels(labels_path)

    img_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    ])

    image_paths = []
    raw_gts     = {}

    for fname in img_files:
        stem     = os.path.splitext(fname)[0]
        gt       = gt_map.get(stem)
        if gt is None:
            continue
        img_path = os.path.join(image_dir, fname)
        image_paths.append(img_path)
        raw_gts[img_path] = gt

    print(f"[DATA] {len(image_paths)} samples loaded")
    return image_paths, raw_gts


# ============================================================================
# Main
# ============================================================================

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Blur-OCR evaluation (only SFT)")
    parser.add_argument("--data_dir",       default=DATA_DIR)
    parser.add_argument("--output_dir",     default=OUTPUT_DIR)
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--max_samples",    type=int, default=None)
    parser.add_argument("--skip_inference", action="store_true")
    # 모델 경로
    parser.add_argument("--base_model",     default=BASE_MODEL_PATH)
    parser.add_argument("--sft_char_model", default="", help="char-level cold start SFT 모델 경로")
    parser.add_argument("--sft_word_model", default="", help="word-level cold start SFT 모델 경로")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "metrics"),     exist_ok=True)
    tmp_dir = os.path.join(args.output_dir, "_tmp_preds")
    os.makedirs(tmp_dir, exist_ok=True)

    image_paths, raw_gts = load_dataset(args.data_dir)
    if args.max_samples:
        image_paths = image_paths[:args.max_samples]
        raw_gts     = {k: v for k, v in raw_gts.items() if k in image_paths}

    t0 = time.time()

    # ── BASE 모델 설정 ────────────────────────────────────────────────────
    # BASE: 비교군 (general prompt, C태그 지시)
    base_model_configs = {
        "BASE": (args.base_model, GENERAL_SYSTEM_PROMPT, False, "0,1"),
    }

    # BASE prediction 캐시 확인 및 inference
    base_preds = {}
    base_cache_map = {
        "BASE": BASE_PRED_CACHE,
    }

    base_to_run = {}
    for mk, cache_path in base_cache_map.items():
        if os.path.exists(cache_path):
            print(f"[BASE] {mk} 캐시 존재 → 재사용: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                base_preds[mk] = json.load(f)
        else:
            print(f"[BASE] {mk} 캐시 없음 → inference 필요")
            base_to_run[mk] = base_model_configs[mk]

    if base_to_run:
        os.makedirs(os.path.dirname(BASE_PRED_CACHE), exist_ok=True)
        base_tmp_paths = launch_inference_parallel(
            base_to_run, image_paths, tmp_dir, args.batch_size
        )
        for mk, tmp_path in base_tmp_paths.items():
            cache_path = base_cache_map[mk]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            import shutil
            shutil.copy(tmp_path, cache_path)
            with open(cache_path, "r", encoding="utf-8") as f:
                base_preds[mk] = json.load(f)
            print(f"[BASE] {mk} 캐시 저장 → {cache_path}")

    # ── SFT 모델 설정 ─────────────────────────────────────────────────────
    sft_model_configs = {}
    gpu_offset = 0
    if args.sft_char_model:
        sft_model_configs["SFT_char"] = (args.sft_char_model, OURS_SYSTEM_PROMPT, True,
                                          f"{gpu_offset},{gpu_offset+1}")
        gpu_offset += 2
    if args.sft_word_model:
        sft_model_configs["SFT_word"] = (args.sft_word_model, OURS_SYSTEM_PROMPT, True,
                                          f"{gpu_offset},{gpu_offset+1}")

    sft_pred_paths = {mk: os.path.join(tmp_dir, f"{mk}_predictions.json")
                      for mk in sft_model_configs}

    if args.skip_inference:
        missing = [mk for mk, p in sft_pred_paths.items() if not os.path.exists(p)]
        if missing:
            print(f"[WARN] 파일 없음 → inference 실행: {missing}")
            args.skip_inference = False

    if not args.skip_inference and sft_model_configs:
        sft_pred_paths = launch_inference_parallel(
            sft_model_configs, image_paths, tmp_dir, args.batch_size
        )

    # 모든 prediction 합치기
    all_preds = dict(base_preds)
    for mk, path in sft_pred_paths.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                all_preds[mk] = json.load(f)

    # tagged_gt 생성 없음 → aggregate에서 word_accuracy < 1 샘플을 hard로 자동 처리

    # 메트릭 계산 (BASE + SFT 모델)
    eval_model_keys = ["BASE"] + list(sft_model_configs.keys())
    all_sample_metrics: Dict[str, List[Dict]] = {k: [] for k in eval_model_keys}

    print("\n[METRICS] Computing per-sample metrics...")
    for img_path in image_paths:
        raw_gt = raw_gts[img_path]
        for model_key in eval_model_keys:
            pred = all_preds.get(model_key, {}).get(img_path, "")
            m    = compute_metrics(pred, raw_gt, "")
            m["image_path"] = img_path
            all_sample_metrics[model_key].append(m)

    # sample_results.jsonl 저장
    print("\n[SAVE] Writing sample_results.jsonl...")
    jsonl_path = os.path.join(args.output_dir, "predictions", "sample_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, img_path in enumerate(image_paths):
            sample_id = os.path.splitext(os.path.basename(img_path))[0]
            record = {
                "sample_id":   sample_id,
                "image_path":  img_path,
                "raw_gt":      raw_gts[img_path],
                "predictions": {mk: all_preds.get(mk, {}).get(img_path, "")
                                for mk in eval_model_keys},
                "metrics":     {mk: {k: v for k, v in all_sample_metrics[mk][i].items()
                                     if k != "image_path"}
                                for mk in eval_model_keys},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[SAVE] {jsonl_path}")

    # eval_summary.json 저장
    aggregated = {mk: aggregate(all_sample_metrics[mk])
                  for mk in eval_model_keys}

    summary = {
        "meta": {
            "n_samples":       len(image_paths),
            "tagged_gt_level": "word",
            "eval_date":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "models":          eval_model_keys,
            "gap_denominator": "GT-length based (|y|_in,L / |y|_out,L), per paper Section 4",
        },
        "results":         aggregated,
        "paper_reference": {"Qwen2.5-VL-7B (UNC)": PAPER_REF},
    }

    summary_path = os.path.join(args.output_dir, "metrics", "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {summary_path}")

    print_summary(aggregated)
    print(f"\n총 소요 시간: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
