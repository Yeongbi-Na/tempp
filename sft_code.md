# Blur-OCR SFT 코드 파이프라인 (`sft_ver4.py`)

---

## 1. 프롬프트 정의

```python
SYSTEM_PROMPT = """You are an intelligent OCR system.
Perform OCR to transcribe all text from the image."""

UNC_START = "<C>"
UNC_END   = "</C>"
```

모델에게 주입하는 시스템 프롬프트 및 불확실성 태그 상수 정의.
`<C>...</C>` 태그는 OCR 결과 중 불확실한 구간을 표시하는 용도.

---

## 2. UNC 태그 내부 토큰 마스킹 (`mask_unc_tag_tokens`)

```python
def mask_unc_tag_tokens(input_ids: torch.Tensor, labels: torch.Tensor,
                        tokenizer, assistant_start: int) -> torch.Tensor:
    labels = labels.clone()

    resp_ids = input_ids[assistant_start:].tolist()
    if not resp_ids:
        return labels

    resp_text = tokenizer.decode(resp_ids, skip_special_tokens=False)

    unc_char_spans = []
    for m in re.finditer(r"<C>(.*?)</C>", resp_text, flags=re.DOTALL):
        inner_start = m.start() + len(UNC_START)
        inner_end   = m.end()   - len(UNC_END)
        if inner_start < inner_end:
            unc_char_spans.append((inner_start, inner_end))

    if not unc_char_spans:
        return labels

    encoded = tokenizer(
        resp_text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offset_mapping = encoded["offset_mapping"]

    for tok_offset, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == char_end:
            continue
        abs_idx = assistant_start + tok_offset
        if abs_idx >= len(labels):
            break
        for (span_start, span_end) in unc_char_spans:
            if char_start >= span_start and char_end <= span_end:
                if labels[abs_idx] != -100:
                    labels[abs_idx] = -100
                break

    return labels
```

**역할**: `<C>...</C>` 사이의 오류 텍스트 토큰을 loss 계산에서 제외 (`labels = -100`)

**핵심 로직**:

| 단계 | 내용 |
|------|------|
| ① | assistant 응답 구간 디코딩 |
| ② | 정규식으로 `<C>...</C>` 내부 char span 추출 |
| ③ | `tokenizer(return_offsets_mapping=True)`로 토큰↔문자 위치 매핑 |
| ④ | UNC 구간에 속하는 토큰의 label을 -100으로 마스킹 |

> **포인트**: `<C>`, `</C>` 태그 자체는 loss에 포함 → 태그 생성법 학습 목적

---

## 3. 데이터셋 클래스 (`BlurOCRDataset`)

### 3-1. `__init__` — 데이터 로드

```python
class BlurOCRDataset(Dataset):
    def __init__(self, jsonl_path: str, processor: AutoProcessor,
                 mask_unc_content: bool = True):
        self.data             = []
        self.processor        = processor
        self.mask_unc_content = mask_unc_content

        if not jsonl_path:
            return

        print(f"📂 Loading data from {jsonl_path}...")
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if os.path.exists(item['image_path']):
                        self.data.append(item)
        except Exception as e:
            print(f"❌ Error loading data: {e}")

        print(f"✅ Loaded {len(self.data)} samples.")
```

`.jsonl` 파일을 한 줄씩 읽어 `image_path` 존재 여부를 확인 후 `self.data`에 적재.
존재하지 않는 이미지 경로는 자동으로 필터링.

---

### 3-2. `_find_assistant_start` — assistant 응답 시작점 탐색

```python
def _find_assistant_start(self, input_ids: torch.Tensor) -> int:
    assistant_marker = "<|im_start|>assistant"
    marker_ids = self.processor.tokenizer.encode(
        assistant_marker, add_special_tokens=False
    )
    input_list = input_ids.tolist()
    marker_len = len(marker_ids)

    for i in range(len(input_list) - marker_len + 1):
        if input_list[i:i + marker_len] == marker_ids:
            return i + marker_len + 1  # +1 for \n token

    print(f"[WARNING] assistant marker not found in input_ids! Masking all labels.")
    return len(input_list)
```

`input_ids` 내에서 `<|im_start|>assistant` 마커를 탐색하여 assistant 응답 시작 인덱스 반환.
`+1`은 마커 직후 `\n` 토큰을 건너뛰기 위함.
마커를 찾지 못하면 전체를 마스킹 (안전 처리).

---

### 3-3. `__getitem__` — 샘플 전처리

```python
def __getitem__(self, idx):
    while True:
        try:
            item        = self.data[idx]
            image_path  = item['image_path']
            target_text = item['ocr_with_tags']

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
                },
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image_path}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_text}]
                }
            ]

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
                max_length=4096,
                truncation=True,
            )

            input_ids = inputs["input_ids"][0]
            labels    = input_ids.clone()

            # Step 1: prompt 구간 마스킹
            assistant_start      = self._find_assistant_start(input_ids)
            labels[:assistant_start] = -100

            # Step 2: UNC 태그 내부 토큰 마스킹
            if self.mask_unc_content and UNC_START in target_text:
                labels = mask_unc_tag_tokens(
                    input_ids, labels,
                    self.processor.tokenizer,
                    assistant_start
                )

            valid_label_count = (labels != -100).sum().item()
            if valid_label_count == 0:
                print(f"[WARNING] idx={idx} has 0 valid label tokens! Skipping.")
                raise ValueError("No valid labels")

            return {
                "input_ids":      input_ids,
                "attention_mask": inputs["attention_mask"][0],
                "labels":         labels,
                "pixel_values":   inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"]
            }

        except Exception as e:
            import random
            current_path = self.data[idx].get('image_path', 'Unknown')
            print(f"⚠️ Error at idx={idx} ({current_path}): {e}")
            idx = random.randint(0, len(self.data) - 1)
            continue
```

**처리 순서**:

| 단계 | 내용 |
|------|------|
| ① | `apply_chat_template` → system/user/assistant 대화 포맷 텍스트 생성 |
| ② | `processor` → `input_ids`, `pixel_values`, `image_grid_thw` 추출 |
| ③ | `labels = input_ids.clone()` → 기본 labels 생성 |
| ④ | **Step 1**: 프롬프트 구간 `-100` 마스킹 |
| ⑤ | **Step 2**: UNC 태그 내부 `-100` 마스킹 |
| ⑥ | `valid_label_count == 0` 체크 → 유효 토큰 없으면 skip |

> **에러 처리**: `while True` 루프로 이미지 로드 실패 또는 유효 토큰 없는 경우 자동으로 다른 샘플로 대체

---

## 4. 데이터 콜레이터 (`QwenDataCollator`)

```python
class QwenDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_ids      = [f["input_ids"]       for f in features]
        attention_mask = [f["attention_mask"]  for f in features]
        labels         = [f["labels"]          for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        pixel_values   = torch.cat([f["pixel_values"]   for f in features], dim=0)
        image_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0)

        return {
            "input_ids":      input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels":         labels_padded,
            "pixel_values":   pixel_values,
            "image_grid_thw": image_grid_thw
        }
```

배치 내 가변 길이 시퀀스를 패딩하여 텐서로 정렬.

| 필드 | 패딩값 | 비고 |
|------|--------|------|
| `input_ids` | `pad_token_id` | |
| `attention_mask` | `0` | |
| `labels` | `-100` | loss 자동 무시 |
| `pixel_values` | `torch.cat` | 패딩 불필요 |
| `image_grid_thw` | `torch.cat` | 패딩 불필요 |

---

## 5. 학습 함수 (`train`)

### 5-1. 모델 & 프로세서 로드

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None
)

processor = AutoProcessor.from_pretrained(
    args.model_name,
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28,
    trust_remote_code=True
)
```

- `bfloat16` + `flash_attention_2` 조합으로 메모리 효율 극대화
- `device_map=None` → DeepSpeed가 직접 디바이스 배치 담당
- `min/max_pixels`로 이미지 해상도 범위 고정 (256~1024 패치)

---

### 5-2. Train / Val 내부 분할

```python
full_dataset = BlurOCRDataset(
    args.train_file, processor,
    mask_unc_content=args.mask_unc_content
)

n_total = len(full_dataset)
n_train = args.n_train

if n_train >= n_total:
    raise ValueError(
        f"n_train({n_train}) >= total samples({n_total}). "
        "Cannot construct val set."
    )

train_dataset = Subset(full_dataset, list(range(n_train)))
val_dataset   = Subset(full_dataset, list(range(n_train, n_total)))
print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
```

단일 `.jsonl` 파일을 `--n_train` 기준으로 앞→train, 뒤→val로 분할.
기본값: `n_train=107520`

---

### 5-3. DeepSpeed ZeRO-2 설정

```python
ds_config = {
    "fp16": {"enabled": False},
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": False,          # [핵심 1] 통신 오버랩 시 bf16 텐서 오염 방지
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": False,  # [핵심 2] 메모리 단편화로 인한 그래디언트 파괴 방지
    },
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "train_batch_size": "auto",
    "gradient_clipping": 1.0,
    "steps_per_print": 50,
    "wall_clock_breakdown": False,
}
```

| 옵션 | 기본값 | 변경 이유 |
|------|--------|-----------|
| `overlap_comm` | `True` | 통신 오버랩 중 bf16 텐서 allreduce 오염 → `False` |
| `contiguous_gradients` | `True` | 메모리 단편화 → 그래디언트 파괴 → `False` |

---

### 5-4. TrainingArguments

```python
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,

    eval_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=1,

    learning_rate=2.5e-6,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,

    logging_steps=10,
    logging_first_step=True,

    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=False,

    bf16=True,
    gradient_checkpointing=True,
    # [핵심 3] DeepSpeed + PyTorch 역전파 충돌로 인한 NaN 발생 완벽 차단
    gradient_checkpointing_kwargs={"use_reentrant": False},

    dataloader_num_workers=8,
    report_to="wandb",
    run_name=args.run_name,
    deepspeed=ds_config,
    remove_unused_columns=False,
)
```

| 파라미터 | 값 | 비고 |
|----------|----|------|
| `learning_rate` | `2.5e-6` | sft_ver3 기준 |
| `warmup_ratio` | `0.1` | |
| `lr_scheduler_type` | `cosine` | |
| `eval_steps` | `100` | |
| `save_steps` | `200` | |
| `use_reentrant` | `False` | **[핵심 3]** DeepSpeed + PyTorch 역전파 충돌 NaN 완벽 차단 |

> **NaN 해결 3가지 핵심 요소**:
> 1. `overlap_comm=False` → bf16 텐서 오염 방지
> 2. `contiguous_gradients=False` → 그래디언트 메모리 파괴 방지
> 3. `use_reentrant=False` → DeepSpeed + PyTorch 역전파 충돌 차단

---

### 5-5. Trainer 실행 및 저장

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=QwenDataCollator(processor)
)

print("🔥 Starting Training...")
resume_checkpoint = args.resume_from_checkpoint
if resume_checkpoint == "True":
    resume_checkpoint = True

trainer.train(resume_from_checkpoint=resume_checkpoint)

print("💾 Saving Model...")
trainer.save_model(args.output_dir)
processor.save_pretrained(args.output_dir)
```

HuggingFace 순정 `Trainer` 사용. 커스텀 로직 없이 표준 학습 루프 활용.
`resume_from_checkpoint`에 문자열 `"True"` 전달 시 Python `True`로 변환 (최신 체크포인트 자동 탐색).

---

## 6. 인자 파싱 (`argparse`)

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",  type=str, required=True)
    parser.add_argument("--output_dir",  type=str, default="./checkpoints/blur_ocr_sft")
    parser.add_argument("--model_name",  type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--n_train",     type=int, default=107520,
                        help="train으로 사용할 앞쪽 샘플 수 (나머지는 val)")
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--grad_accum",  type=int, default=16)
    parser.add_argument("--epochs",      type=int, default=1)
    parser.add_argument("--run_name",    type=str, default="sft-blur-ocr")
    parser.add_argument("--use_deepspeed",       action="store_true")
    parser.add_argument("--local_rank",           type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--mask_unc_content",     action="store_true", default=True)
    parser.add_argument("--no_mask_unc_content",  dest="mask_unc_content",
                        action="store_false")

    args = parser.parse_args()
    train(args)
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--train_file` | 필수 | JSONL 데이터 경로 |
| `--output_dir` | `./checkpoints/blur_ocr_sft` | 체크포인트 저장 경로 |
| `--model_name` | `Qwen/Qwen2.5-VL-7B-Instruct` | 베이스 모델 |
| `--n_train` | `107520` | train 샘플 수 (나머지는 val) |
| `--batch_size` | `1` | GPU당 배치 크기 |
| `--grad_accum` | `16` | gradient accumulation steps |
| `--epochs` | `1` | 학습 에폭 수 |
| `--run_name` | `sft-blur-ocr` | WandB run 이름 |
| `--use_deepspeed` | `False` | DeepSpeed 활성화 여부 |
| `--mask_unc_content` | `True` | UNC 내부 마스킹 여부 |
| `--resume_from_checkpoint` | `None` | 체크포인트 재시작 경로 |

---

## 전체 파이프라인 흐름

```
JSONL 데이터 로드 (BlurOCRDataset)
       ↓
train/val 분할 (Subset)
       ↓
모델 & 프로세서 로드 (Qwen2.5-VL-7B + flash_attention_2)
       ↓
__getitem__: 이미지 + 텍스트 → input_ids, labels 생성
       ├─ 프롬프트 구간 마스킹 (-100)
       └─ UNC 태그 내부 마스킹 (-100)
       ↓
QwenDataCollator: 배치 패딩 & 정렬
       ↓
Trainer.train() (DeepSpeed ZeRO-2 + bf16)
       ↓
모델 & 프로세서 저장
```
