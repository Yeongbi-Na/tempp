# Blur-OCR BASE 모델 평가 파이프라인 (`eval_base.py`)

BASE 모델(Qwen2.5-VL-7B)을 두 가지 시스템 프롬프트로 평가.
inference → 메트릭 계산 → 결과 저장까지 전 과정을 포함.

---

## 목차

- [1. 경로 및 상수 설정](#1-경로-및-상수-설정)
- [2. 프롬프트 및 모델 설정](#2-프롬프트-및-모델-설정)
- [3. vLLM Inference Worker](#3-vllm-inference-worker)
- [4. 병렬 Inference 실행](#4-병렬-inference-실행)
- [5. 텍스트 정규화 및 마스크 추출](#5-텍스트-정규화-및-마스크-추출)
- [6. 메트릭 계산](#6-메트릭-계산)
- [7. 집계](#7-집계)
- [8. 결과 출력](#8-결과-출력)
- [9. 데이터 로드](#9-데이터-로드)
- [10. Main 함수](#10-main-함수)
- [전체 파이프라인 흐름](#전체-파이프라인-흐름)

---

## 1. 경로 및 상수 설정

```python
BASE_MODEL_PATH = (
    "/nas/home/.../models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898..."
)
DATA_DIR   = "/nas/datahub/Blur-OCR/Blur-OCR/val"
OUTPUT_DIR = "/nas/datahub/Blur-OCR/results/eval_base"

UNC_START   = "<C>"
UNC_END     = "</C>"
UNC_PATTERN = re.compile(re.escape(UNC_START) + r"(.*?)" + re.escape(UNC_END), re.DOTALL)
VALID_EXT   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

PAPER_REF = {
    "word_precision": 0.839, "word_recall": 0.620, "word_f1": 0.685,
    "char_precision": 0.626, "char_recall": 0.599, "char_f1": 0.572,
    "word_accuracy":  0.839, "char_accuracy": 0.881,
    "word_gap":       0.739, "char_gap":      0.694,
}
```

모델 경로, 데이터 경로, 출력 경로 고정.
`PAPER_REF`는 논문에서 보고한 수치로, 최종 출력 시 비교 기준으로 사용.
`UNC_PATTERN`은 `<C>...</C>` 구간을 추출하는 정규식 (re.DOTALL로 줄바꿈 포함).

---

## 2. 프롬프트 및 모델 설정

```python
OURS_SYSTEM_PROMPT = (
    "You are an intelligent OCR system.\n"
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

MODEL_CONFIGS = {
    "BASE_v1": (BASE_MODEL_PATH, OURS_SYSTEM_PROMPT,    False, "4,5"),
    "BASE_v2": (BASE_MODEL_PATH, GENERAL_SYSTEM_PROMPT, False, "6,7"),
}
```

| 모델 키 | 프롬프트 | GPU | 특징 |
|---------|----------|-----|------|
| `BASE_v1` | `OURS_SYSTEM_PROMPT` | 4,5 | 태그 지시 없음 (기본 OCR만) |
| `BASE_v2` | `GENERAL_SYSTEM_PROMPT` | 6,7 | 논문 Appendix D 프롬프트, `<C>` 태그 명시적 지시 |

`MODEL_CONFIGS` 튜플 구조: `(모델 경로, 시스템 프롬프트, greedy 여부, GPU IDs)`
두 모델이 서로 다른 GPU 쌍을 점유하므로 병렬 실행 가능.

---

## 3. vLLM Inference Worker

```python
def run_inference_worker(args_dict: dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = args_dict["gpus"]
    from vllm import LLM, SamplingParams

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
            # 배치 실패 시 개별 재시도
            for p in batch:
                try:
                    out = llm.chat([single_message], ...)
                    results[p] = out[0].outputs[0].text.strip()
                except Exception as e2:
                    results[p] = ""

    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
```

별도 프로세스에서 실행되는 inference 함수. `mp.Process`로 분리된 이유는 vLLM이 GPU를 프로세스 단위로 점유하기 때문.

**주요 vLLM 설정**:

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `tensor_parallel_size` | `2` | GPU 2장으로 모델 분산 |
| `dtype` | `bfloat16` | |
| `max_model_len` | `8192` | 최대 시퀀스 길이 |
| `gpu_memory_utilization` | `0.85` | GPU 메모리 85% 사용 |
| `temperature` | `0.0` (greedy) / `0.2` | greedy=False이므로 0.2 |

**에러 처리**: 배치 단위 실패 시 개별 이미지로 재시도, 그것도 실패하면 빈 문자열(`""`) 저장.
결과는 `{image_path: prediction_text}` 형태의 JSON으로 저장.

---

## 4. 병렬 Inference 실행

```python
def launch_inference_parallel(model_configs, image_paths, tmp_dir, batch_size=8):
    processes, output_paths = [], {}
    for model_key, (model_path, system_prompt, greedy, gpus) in model_configs.items():
        output_path = os.path.join(tmp_dir, f"{model_key}_predictions.json")
        p = mp.Process(target=run_inference_worker, args=(args_dict,))
        p.start()
        processes.append((model_key, p))

    for model_key, p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"[ERROR] {model_key} exitcode={p.exitcode}")
    return output_paths
```

`MODEL_CONFIGS`에 정의된 모든 모델을 동시에 `mp.Process`로 실행.
`BASE_v1`(GPU 4,5)과 `BASE_v2`(GPU 6,7)가 병렬로 inference 수행.
`p.join()`으로 모든 프로세스가 끝날 때까지 대기 후 종료 코드 확인.

---

## 5. 텍스트 정규화 및 마스크 추출

### 5-1. `normalize_text`

```python
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
```

`<C>` 태그를 UUID 플레이스홀더로 임시 치환 → 소문자화/공백 정규화 → 태그 복원.
태그가 소문자 변환이나 공백 처리 중 깨지는 것을 방지하기 위한 방어 처리.

---

### 5-2. `extract_clean_and_mask`

```python
def extract_clean_and_mask(text: str) -> Tuple[str, np.ndarray, List[Tuple[int, int]]]:
    # 태그 제거 후 clean 텍스트 추출
    # <C> 구간에 해당하는 위치를 bool mask 배열로 반환
    ...
    clean_text = UNC_PATTERN.sub(r"\1", cleaned)
    mask = np.zeros(len(clean_text), dtype=bool)
    for s, e in regions:
        mask[s:e] = True
    return clean_text, mask, regions
```

**반환값**:

| 반환값 | 타입 | 설명 |
|--------|------|------|
| `clean_text` | `str` | 태그 제거된 순수 텍스트 |
| `mask` | `np.ndarray[bool]` | `clean_text`와 같은 길이, `<C>` 구간 위치가 `True` |
| `regions` | `List[(int, int)]` | `<C>` 구간의 (start, end) 인덱스 목록 |

비정상 태그(열리지 않은 `</C>`, 닫히지 않은 `<C>`) 처리 포함.

---

### 5-3. `tokenize_words`

```python
def tokenize_words(text: str) -> List[Tuple[str, int, int]]:
    # 공백 기준 단어 분리 → (단어, start_idx, end_idx) 반환
```

공백 기준으로 단어를 분리하면서 각 단어의 문자 인덱스(start, end)를 함께 반환.
이후 word-level 메트릭 계산 시 mask와 위치를 매핑하기 위해 사용.

---

## 6. 메트릭 계산 (`compute_metrics`)

```python
def compute_metrics(pred_text: str, raw_gt: str) -> Dict[str, Any]:
    # 1) 텍스트 정규화
    raw_gt_norm = normalize_text(raw_gt)
    pred_norm   = normalize_text(pred_text)

    # 2) clean 텍스트 + UNC 마스크 추출
    clean_pred, pred_mask, pred_regions = extract_clean_and_mask(pred_norm)
    clean_raw_gt = raw_gt_norm  # GT는 태그 없이 그대로 사용

    # 3) OCR 정확도
    word_accuracy = 1.0 - word_ed / max(1, len(w_gt))
    char_accuracy = 1.0 - char_ed / max(1, len(clean_raw_gt))

    # 4) UNC 태깅 품질 (word / char 각각)
    w_prec, w_rec, w_f1, w_gap = unc_stats(...)
    c_prec, c_rec, c_f1, c_gap = unc_stats(...)
```

**내부 `unc_stats` 함수**:

```
precision = in_e / (in_e + in_c)          → UNC 태그 안에 실제 오류가 얼마나 있는가
recall    = in_e / (in_e + out_e)          → 전체 오류 중 UNC 안에 잡힌 비율
f1        = 2 * precision * recall / ...
gap       = (오류율_in_UNC) - (오류율_out_UNC)   → UNC 안/밖 오류율 차이
```

| 변수 | 의미 |
|------|------|
| `in_e` | UNC 태그 안 + 오답 (오류를 올바르게 태깅) |
| `in_c` | UNC 태그 안 + 정답 (정답을 잘못 태깅) |
| `out_e` | UNC 태그 밖 + 오답 (오류를 못 잡음) |
| `out_c` | UNC 태그 밖 + 정답 (정상 케이스) |

**word-level**: Levenshtein opcodes로 pred/GT 단어 정렬 후, 각 정렬 위치가 `pred_mask`에 걸리는지 확인.
**char-level**: 문자 단위 opcodes로 동일하게 처리. `insert` op는 인접 위치의 mask 값으로 귀속.

---

## 7. 집계 (`aggregate`)

```python
def aggregate(all_metrics: List[Dict]) -> Dict:
    def mean(key):
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        return round(sum(vals) / len(vals), 6) if vals else 0.0
    return {
        "n":             n,
        "word_f1":       mean("word_f1"),
        "char_f1":       mean("char_f1"),
        "has_unc_ratio": round(sum(1 for m in all_metrics if m["has_unc"]) / n, 6),
        "n_gap_na_word": sum(1 for m in all_metrics if m.get("word_gap") is None),
        ...
    }
```

샘플별 메트릭을 macro 평균으로 집계.
`gap`이 `None`인 샘플(UNC 태그 없거나 GT가 0인 경우) 제외 후 평균 계산.
`has_unc_ratio`: 전체 샘플 중 `<C>` 태그를 최소 1개 이상 출력한 비율.

---

## 8. 결과 출력 (`print_summary`)

```python
def print_summary(all_results: Dict):
    # 논문 수치와 모델별 결과를 표 형태로 출력
    print(row("[논문] Qwen2.5-VL-7B", PAPER_REF, all_keys))
    for name, r in all_results.items():
        print(row(name, r, all_keys))
```

출력 컬럼: `w_acc | w_prec | w_rec | w_f1 | w_gap | c_acc | c_prec | c_rec | c_f1 | c_gap`
첫 행은 항상 논문 기준값 → 그 아래에 각 모델 결과 출력.
추가로 `has_unc_ratio`, `gap_na` 건수를 별도 테이블로 출력.

---

## 9. 데이터 로드

### 9-1. `load_labels`

```python
def load_labels(labels_path: Path) -> Dict[str, str]:
    # labels.txt 파싱
    # 형식: "filename.jpg\t첫째줄\n둘째줄\n..."
    # stem → gt_text 딕셔너리 반환
```

`labels.txt`는 탭(tab)으로 파일명과 텍스트가 구분되며, 이후 줄은 줄바꿈 텍스트로 처리.
파일명의 확장자를 제거한 stem을 key로 사용.

---

### 9-2. `load_dataset`

```python
def load_dataset(data_dir: str) -> Tuple[List[str], Dict[str, str]]:
    gt_map = load_labels(labels_path)
    img_files = sorted([f for f in os.listdir(image_dir)
                        if os.path.splitext(f)[1].lower() in VALID_EXT])
    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        gt   = gt_map.get(stem)
        if gt is None: continue   # GT 없는 이미지 자동 제외
        image_paths.append(img_path)
        raw_gts[img_path] = gt
    return image_paths, raw_gts
```

`images/` 디렉토리의 유효 확장자 파일만 수집.
`gt_map`에 매칭되는 GT가 없는 이미지는 자동 제외.
반환값: `image_paths` (리스트), `raw_gts` (`{image_path: gt_text}` 딕셔너리)

---

## 10. Main 함수

```python
def main():
    mp.set_start_method("spawn", force=True)

    # 1) 데이터 로드
    image_paths, raw_gts = load_dataset(args.data_dir)
    if args.max_samples:
        image_paths = image_paths[:args.max_samples]

    # 2) inference (--skip_inference 시 기존 JSON 재사용)
    configs_to_run = {}
    for mk in MODEL_CONFIGS:
        pred_path = os.path.join(tmp_dir, f"{mk}_predictions.json")
        if args.skip_inference and os.path.exists(pred_path):
            print(f"[SKIP] {mk} → 재사용: {pred_path}")
        else:
            configs_to_run[mk] = MODEL_CONFIGS[mk]

    if configs_to_run:
        launch_inference_parallel(configs_to_run, image_paths, tmp_dir, args.batch_size)

    # 3) predictions JSON 로드
    all_preds = {mk: json.load(open(path)) for mk, path in all_pred_paths.items()}

    # 4) 샘플별 메트릭 계산
    for img_path in image_paths:
        for mk in MODEL_CONFIGS:
            pred = all_preds[mk].get(img_path, "")
            m    = compute_metrics(pred, raw_gts[img_path])
            all_sample_metrics[mk].append(m)

    # 5) 모델별 결과 저장
    for mk in MODEL_CONFIGS:
        # predictions/{mk}_sample_results.jsonl
        # metrics/{mk}_eval_summary.json

    # 6) 전체 비교 출력
    aggregated = {mk: aggregate(all_sample_metrics[mk]) for mk in MODEL_CONFIGS}
    print_summary(aggregated)
```

**실행 순서 요약**:

| 단계 | 내용 |
|------|------|
| ① | `mp.set_start_method("spawn")` — vLLM 호환을 위해 필수 |
| ② | 데이터 로드 (`load_dataset`) |
| ③ | inference 실행 또는 기존 결과 재사용 (`--skip_inference`) |
| ④ | 샘플별 `compute_metrics` 순회 |
| ⑤ | `{mk}_sample_results.jsonl` + `{mk}_eval_summary.json` 저장 |
| ⑥ | `print_summary`로 논문 대비 전체 비교 출력 |

**`--skip_inference` 플래그**: `_tmp_preds/` 에 이미 prediction JSON이 있으면 inference 생략, 메트릭 계산만 재실행. 디버깅/재평가 시 유용.

**출력 디렉토리 구조**:
```
output_dir/
├── _tmp_preds/
│   ├── BASE_v1_predictions.json
│   └── BASE_v2_predictions.json
├── predictions/
│   ├── BASE_v1_sample_results.jsonl
│   └── BASE_v2_sample_results.jsonl
└── metrics/
    ├── BASE_v1_eval_summary.json
    └── BASE_v2_eval_summary.json
```

---

## 전체 파이프라인 흐름

```
load_dataset()
    ├─ load_labels()  →  stem → gt_text 딕셔너리
    └─ 이미지 경로 수집 (GT 없는 샘플 자동 제외)
          ↓
launch_inference_parallel()
    ├─ BASE_v1 (GPU 4,5) → mp.Process → run_inference_worker()
    └─ BASE_v2 (GPU 6,7) → mp.Process → run_inference_worker()
          ↓ (병렬 실행 후 join)
_tmp_preds/{mk}_predictions.json 로드
          ↓
compute_metrics(pred, raw_gt)  ← 샘플 순회
    ├─ normalize_text()
    ├─ extract_clean_and_mask()  →  clean_text + bool mask
    ├─ word-level: Levenshtein opcodes → unc_stats()
    └─ char-level: Levenshtein opcodes → unc_stats()
          ↓
aggregate()  →  macro 평균
          ↓
저장: sample_results.jsonl + eval_summary.json
          ↓
print_summary()  →  논문 대비 전체 비교 출력
```

---

## 인자 전체 목록

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | `DATA_DIR` | val 데이터 경로 |
| `--output_dir` | `OUTPUT_DIR` | 결과 저장 경로 |
| `--batch_size` | `8` | inference 배치 크기 |
| `--max_samples` | `None` | 평가 샘플 수 제한 (디버깅용) |
| `--skip_inference` | `False` | 기존 prediction JSON 재사용 |
| `--base_model` | `BASE_MODEL_PATH` | 모델 경로 override |
