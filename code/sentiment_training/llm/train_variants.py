#!/usr/bin/env python3
"""Train the P0 LoRA sentiment model and select a validation checkpoint."""

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import sys
import time
import traceback
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path


LABEL_TEXT = {-1: "负面", 0: "中性", 1: "正面"}
TEXT_LABEL = {value: key for key, value in LABEL_TEXT.items()}
LABELS = (-1, 0, 1)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CHECKPOINT_RETENTION = 3


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SOURCE_RUNNER_SHA256 = sha256_file(Path(__file__))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path, rows):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def resolve_path(base, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def read_rows(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty JSON list")
    ids = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or not all(k in row for k in ("sample_id", "bank", "title", "label")):
            raise ValueError(f"{path}: row {position} lacks required fields")
        if type(row["label"]) is not int or row["label"] not in LABEL_TEXT:
            raise ValueError(f"{path}: row {position} has invalid label")
        if not isinstance(row["bank"], str) or not isinstance(row["title"], str):
            raise ValueError(f"{path}: row {position} bank/title must be strings")
        sample_id = str(row["sample_id"])
        if sample_id in ids:
            raise ValueError(f"{path}: duplicate sample_id {sample_id}")
        ids.add(sample_id)
    return rows


def load_job(args):
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('tasks'), list):
        raise ValueError('manifest needs a tasks list')
    matches = [task for task in manifest['tasks'] if isinstance(task, dict) and task.get('id') == args.task_id]
    if len(matches) != 1:
        raise ValueError(f'task id {args.task_id!r} must occur exactly once')
    task = matches[0]
    allowed = 'hfl/llama-3-chinese-8b-instruct-v3'
    if task['model_id'] != allowed:
        raise ValueError('Only v3 is authorized for LoRA')
    source_labels = manifest.get('source_labels_file')
    source_hash = manifest.get('source_labels_sha256')
    if not isinstance(source_labels, str) or not source_labels or (not isinstance(source_hash, str)):
        raise ValueError('manifest needs source_labels_file and source_labels_sha256')
    if sha256_file(resolve_path(manifest_path.parent, source_labels)) != source_hash:
        raise ValueError('source labels SHA-256 mismatch')
    for key in ('model_id', 'model_path', 'kind', 'regime', 'output_dir'):
        if not isinstance(task.get(key), str) or not task[key]:
            raise ValueError(f'task needs nonempty {key}')
    if task['kind'] != 'causal' or task['regime'] != 'lora':
        raise ValueError('this runner requires causal lora tasks')
    expected_epochs = 4
    if 'epochs' in task and task['epochs'] != expected_epochs:
        raise ValueError(f'protocol fixes epochs to {expected_epochs}')
    if manifest.get('seed', 20) != 20:
        raise ValueError('protocol fixes seed to 20')
    expected_hashes = manifest.get('data_sha256')
    if not isinstance(expected_hashes, dict):
        raise ValueError('data_sha256 must be a mapping')
    base = manifest_path.parent
    data_dir = resolve_path(base, manifest.get('data_dir', '.'))
    data = {}
    paths = {}
    hashes = {}
    for name, default in (('train', 'train.json'), ('validation', 'validation.json'), ('test', 'held_out.json')):
        value = manifest.get(f'{name}_file')
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f'{name}_file must be a path')
        path = resolve_path(base, value) if value is not None else resolve_path(data_dir, default)
        data[name] = read_rows(path)
        paths[name] = str(path)
        hashes[name] = sha256_file(path)
        expected_hash = expected_hashes.get('held_out' if name == 'test' else name)
        if not isinstance(expected_hash, str) or hashes[name] != expected_hash:
            raise ValueError(f'{name} data SHA-256 mismatch: {path}')
    if {k: len(v) for k, v in data.items()} != {'train': 7000, 'validation': 2000, 'test': 1000}:
        raise ValueError('Exact seed42 split counts required')
    id_sets = [{str(row['sample_id']) for row in rows} for rows in data.values()]
    if any((id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3))):
        raise ValueError('sample_id overlaps across splits')
    model_path = resolve_path(base, task['model_path'])
    if not model_path.is_dir():
        raise ValueError(f'local model directory missing: {model_path}')
    ready = json.loads(resolve_path(base, task['ready_file']).read_text(encoding='utf-8'))
    if ready.get('status') != 'ready' or ready.get('model_id') != task['model_id'] or ready.get('revision') != task['revision'] or (ready.get('path') != str(model_path)):
        raise ValueError('Cache readiness identity mismatch')
    for filename, expected in ready['files_sha256'].items():
        if sha256_file(model_path / filename) != expected:
            raise ValueError(f'Cache file SHA256 mismatch: {filename}')
    if (model_path / 'adapter_config.json').exists():
        raise ValueError('Base cache contains an adapter; refusing ambiguous model source')
    output_dir = resolve_path(base, task['output_dir'])
    return (manifest_path, task, model_path, output_dir, data, paths, hashes)


TEMPLATE_ID = "P0"
TEMPLATES = {"P0": "判断银行股吧标题的情绪。只输出负面、中性或正面。"}

def prompt_text(tokenizer, row):
    template = TEMPLATE_ID
    if template not in TEMPLATES:
        raise ValueError(f"unsupported template_id: {TEMPLATE_ID}")
    system = TEMPLATES[template]
    user = f"银行：{row['bank']}\n标题：{row['title']}"
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"指令：{system}\n用户：{user}\n回答："


def encode_causal(tokenizer, rows, max_length=1024):
    result = []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    for row in rows:
        prompt = tokenizer.encode(prompt_text(tokenizer, row), add_special_tokens=False)
        completion = tokenizer.encode(LABEL_TEXT[row["label"]], add_special_tokens=False) + eos
        if not completion or len(completion) >= max_length:
            raise ValueError("invalid label completion length")
        prompt = prompt[-(max_length - len(completion)):]
        if not prompt:
            raise ValueError("empty prompt tokens")
        result.append({"input_ids": prompt + completion, "labels": [-100] * len(prompt) + completion,
                       "prompt_ids": prompt, "target_tokens": len(completion),
                       "completion_eos_position": len(prompt + completion) - 1 if eos else None})
    return result


def weighted_completion_loss(logits, labels, completion_eos_positions, torch, eos_weight=1.0, label_smoothing=0.0):
    """Shift causal targets; weight only explicitly tracked terminal completion EOS."""
    if not 0 < eos_weight <= 1 or not 0 <= label_smoothing < 1:
        raise ValueError("invalid EOS weight or smoothing")
    suffix = logits.shape[1] - 1
    if suffix < 1:
        raise ValueError("at least two logits positions required")
    targets = labels[:, -suffix:]
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=-100, reduction="none", label_smoothing=label_smoothing).reshape_as(targets)
    weights = (targets != -100).to(losses.dtype)
    # Column j represents label position labels_length - suffix + j.
    columns = completion_eos_positions - (labels.shape[1] - suffix)
    rows = torch.arange(labels.shape[0], device=labels.device)
    valid = (columns >= 0) & (columns < suffix)
    if valid.any():
        if (targets[rows[valid], columns[valid]] == -100).any():
            raise ValueError("completion EOS must be an active target")
        weights[rows[valid], columns[valid]] = eos_weight
    denominator = weights.sum()
    if denominator.item() == 0:
        raise ValueError("no active completion targets")
    return (losses * weights).sum() / denominator


def collate_causal(items, pad_id, torch, device):
    length = max(len(item["input_ids"]) for item in items)
    ids = []
    masks = []
    labels = []
    eos_positions = []
    for item in items:
        pad = length - len(item["input_ids"])
        ids.append([pad_id] * pad + item["input_ids"])
        masks.append([0] * pad + [1] * len(item["input_ids"]))
        labels.append([-100] * pad + item["labels"])
        eos_positions.append(pad + item["completion_eos_position"] if item.get("completion_eos_position") is not None else -1)
    mask = torch.tensor(masks, dtype=torch.long, device=device)
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device=device),
            "attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
            "completion_eos_positions": torch.tensor(eos_positions, dtype=torch.long, device=device)}


def classification_metrics(references, predictions):
    if len(references) != len(predictions) or not references:
        raise ValueError("predictions must match nonempty references")
    per_class = {}
    for label in LABELS:
        tp = sum(ref == label and pred == label for ref, pred in zip(references, predictions))
        fp = sum(ref != label and pred == label for ref, pred in zip(references, predictions))
        fn = sum(ref == label and pred != label for ref, pred in zip(references, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
    total = len(references)
    return {
        "count": total,
        "accuracy": sum(ref == pred for ref, pred in zip(references, predictions)) / total,
        "macro_f1": sum(per_class[str(label)]["f1"] for label in LABELS) / 3,
        "weighted_precision": sum(per_class[str(label)]["precision"] * per_class[str(label)]["support"] for label in LABELS) / total,
        "weighted_recall": sum(per_class[str(label)]["recall"] * per_class[str(label)]["support"] for label in LABELS) / total,
        "weighted_f1": sum(per_class[str(label)]["f1"] * per_class[str(label)]["support"] for label in LABELS) / total,
        "invalid_count": sum(pred not in LABELS for pred in predictions),
        "labels": list(LABELS), "per_class": per_class,
    }


def row_identity(row):
    payload = f"{row['bank']}\n{row['title']}".encode("utf-8")
    return {"sample_id": row["sample_id"], "bank_title_sha256": hashlib.sha256(payload).hexdigest(),
            "reference": row["label"]}


INFER_BATCH_SIZE = 16
MAX_NEW_TOKENS = 8

def infer_causal(model, tokenizer, rows, torch, device, batch_size=None):
    batch_size = batch_size or INFER_BATCH_SIZE
    predictions = []
    model.eval()
    old_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), batch_size):
                part = rows[offset:offset + batch_size]
                prompts = [tokenizer.encode(prompt_text(tokenizer, row), add_special_tokens=False)[-1024:] for row in part]
                length = max(map(len, prompts))
                masks = torch.tensor([[0] * (length - len(ids)) + [1] * len(ids) for ids in prompts],
                                     dtype=torch.long, device=device)
                batch = {"input_ids": torch.tensor([[tokenizer.pad_token_id] * (length - len(ids)) + ids
                                                    for ids in prompts], dtype=torch.long, device=device),
                         "attention_mask": masks}
                generated = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                                           pad_token_id=tokenizer.pad_token_id,
                                           eos_token_id=tokenizer.eos_token_id, use_cache=True)
                for ids in generated[:, length:]:
                    token_ids = ids.tolist()
                    if tokenizer.eos_token_id in token_ids:
                        token_ids = token_ids[:token_ids.index(tokenizer.eos_token_id)]
                    raw = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    predictions.append((TEXT_LABEL.get(raw, -99), raw))
    finally:
        model.config.use_cache = old_cache
    return predictions


def evaluate(model, tokenizer, rows, kind, torch, device, prediction_path=None):
    if kind != "causal":
        raise ValueError("causal validation required")
    outputs = infer_causal(model, tokenizer, rows, torch, device)
    references = [row["label"] for row in rows]
    predictions = [pred for pred, _ in outputs]
    metrics = classification_metrics(references, predictions)
    if prediction_path is not None:
        records = [{**row_identity(row), "pred_label": pred, "raw_output": raw}
                   for row, (pred, raw) in zip(rows, outputs)]
        atomic_jsonl(prediction_path, records)
    return metrics


def build_config(kind, regime):
    if kind != "causal" or regime != "lora":
        raise ValueError("causal LoRA training required")
    return {"epochs": 4, "batch_size": 2, "gradient_accumulation": 8, "learning_rate": 1e-4,
            "scheduler": "cosine", "warmup_ratio": 0.03, "weight_decay": 0.0,
            "max_length": 1024, "eval_steps": 100, "save_steps": 200,
            "checkpoint_retention": CHECKPOINT_RETENTION, "gradient_clip_norm": 1.0, "precision": "bf16",
            "lora": {"r": 64, "alpha": 128, "dropout": 0.05, "targets": LORA_TARGETS}}


def available_checkpoints(root):
    """Return committed checkpoints ordered by their saved training step."""
    found = []
    for path in root.glob("checkpoint-*"):
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit() and (path / "state.json").exists():
            state = json.loads((path / "state.json").read_text(encoding="utf-8"))
            status = state.get("status", "complete")
            if status == "writing":
                continue
            if status != "complete" or type(state.get("step")) is not int or state["step"] < 1:
                raise RuntimeError(f"invalid checkpoint state: {path}")
            slot = state.get("checkpoint_slot")
            if slot is not None and (type(slot) is not int or not 1 <= slot <= CHECKPOINT_RETENTION
                                     or path.name != f"checkpoint-{slot:06d}"):
                raise RuntimeError(f"checkpoint slot mismatch: {path}")
            found.append((state["step"], path, state))
    if len({step for step, _, _ in found}) != len(found):
        raise RuntimeError("checkpoint training steps must be unique")
    return sorted(found, key=lambda entry: entry[0])


def checkpoint_for_step(root, step, identity):
    """Resolve a saved training step and verify its checkpoint identity."""
    if type(step) is not int or step < 1:
        raise ValueError("checkpoint step must be a positive integer")
    matches = [(path, state) for saved_step, path, state in available_checkpoints(root) if saved_step == step]
    if len(matches) != 1:
        raise RuntimeError(f"expected one committed checkpoint for step {step}, found {len(matches)}")
    path, state = matches[0]
    if state["identity"] != identity:
        raise ValueError("selected checkpoint identity mismatch")
    return path


def load_checkpoint_weights(model, model_dir, regime):
    """Restore adapter tensors without replacing optimizer-bound parameters."""
    if regime != "lora":
        raise ValueError("LoRA checkpoint required")
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    weights = load_file(model_dir / "adapter_model.safetensors", device="cpu")
    result = set_peft_model_state_dict(model, weights)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected adapter weights: {result.unexpected_keys}")
    del weights


def random_state(torch):
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_random_state(value, torch):
    random.setstate(value["python"])
    torch.set_rng_state(value["torch_cpu"])
    torch.cuda.set_rng_state_all(value["torch_cuda"])


def save_checkpoint(root, model, tokenizer, optimizer, scheduler, torch, state, identity):
    """Reuse three slots while protecting the committed best and latest points."""
    step = state["step"]
    if type(step) is not int or step < 1:
        raise ValueError("checkpoint step must be a positive integer")
    slots = [root / f"checkpoint-{slot:06d}" for slot in range(1, CHECKPOINT_RETENTION + 1)]
    checkpoints = available_checkpoints(root)
    if any(path not in slots for _, path, _ in checkpoints):
        raise RuntimeError("training requires the three checkpoint slot paths")
    for slot, path in enumerate(slots, start=1):
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"checkpoint slot is not a directory: {path}")
        marker = path / "state.json"
        if marker.exists():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved.get("identity") != identity or saved.get("checkpoint_slot") != slot:
                raise RuntimeError(f"checkpoint slot identity mismatch: {path}")
        elif path.exists() and any(path.iterdir()):
            raise RuntimeError(f"checkpoint slot lacks a state marker: {path}")
    by_step = {saved_step: path for saved_step, path, _ in checkpoints}
    protected = set()
    previous_best = None
    if checkpoints:
        latest_step, latest_path, latest_state = checkpoints[-1]
        if step <= latest_step:
            raise ValueError("checkpoint steps must increase")
        previous_best = latest_state["best_step"]
        if previous_best not in by_step:
            raise RuntimeError("the selected best checkpoint is unavailable")
        protected = {latest_path, by_step[previous_best]}
    if state["best_step"] not in (previous_best, step) or state["best_step"] is None:
        raise ValueError("best checkpoint must be the committed best or the current step")
    occupied = set(by_step.values())
    candidates = [path for path in slots if path not in occupied]
    if not candidates:
        candidates = [path for _, path, _ in checkpoints if path not in protected]
    if not candidates:
        raise RuntimeError("no reusable checkpoint slot")
    final = candidates[0]
    slot = slots.index(final) + 1
    atomic_json(final / "state.json", {"status": "writing", "checkpoint_slot": slot,
                                       "step": step, "identity": identity})
    model.save_pretrained(final / "model", safe_serialization=True)
    tokenizer.save_pretrained(final / "tokenizer")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "random": random_state(torch)}, final / "training_state.pt")
    atomic_json(final / "state.json", {**state, "identity": identity,
                                      "checkpoint_slot": slot, "status": "complete"})
    return final


def model_revision(model_path):
    config = model_path / "config.json"
    if not config.exists():
        return None
    data = json.loads(config.read_text(encoding="utf-8"))
    return data.get("_commit_hash") or data.get("revision")


def run(args):
    global INFER_BATCH_SIZE, MAX_NEW_TOKENS, TEMPLATE_ID
    TEMPLATE_ID = args.template_id
    if (args.beta2, args.init, args.eos_weight, args.label_smoothing, args.template_id) != (0.98, 'default', 1.0, 0.0, 'P0'):
        raise ValueError('fixed training parameters are beta2=.98, init=default, eos_weight=1, label_smoothing=0, template_id=P0')
    INFER_BATCH_SIZE = args.infer_batch_size
    MAX_NEW_TOKENS = args.max_new_tokens
    manifest_path, task, model_path, output_dir, data, data_paths, data_hashes = load_job(args)
    if task['kind'] != 'causal' or task['regime'] != 'lora':
        raise ValueError('train_variants supports causal LoRA only')
    variant = {
        'beta2': args.beta2,
        'init': args.init,
        'eos_weight': args.eos_weight,
        'label_smoothing': args.label_smoothing,
        'template_id': args.template_id,
    }
    if task.get('train_args') != variant:
        raise ValueError('CLI variant must exactly match manifest train_args')
    if args.infer_batch_size != 16 or args.max_new_tokens != 8:
        raise ValueError('validation uses batch16 and max_new_tokens8')
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'training_done.json').exists():
        raise RuntimeError('done.json already exists; refusing to overwrite a completed task')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f'--gpu {args.gpu} unavailable; visible count={torch.cuda.device_count()}')
    torch.cuda.set_device(args.gpu)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('selected CUDA runtime lacks bf16 support')
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    random.seed(20)
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = build_config(task['kind'], task['regime'])
    config['variant'] = variant
    config['adam_betas'] = [0.9, args.beta2]
    config['adam_epsilon'] = 1e-08
    manifest_hash = sha256_file(manifest_path)
    identity = {
        'manifest_sha256': manifest_hash,
        'data_sha256': data_hashes,
        'variant': variant,
        'model_revision': task['revision'],
        'source_runner_sha256': SOURCE_RUNNER_SHA256,
        'task_id': task['id'],
        'model_path': str(model_path),
        'model_id': task['model_id'],
        'kind': task['kind'],
        'regime': task['regime'],
        'inference_batch_size': INFER_BATCH_SIZE,
        'max_new_tokens': MAX_NEW_TOKENS,
        'script_sha256': sha256_file(Path(__file__)),
    }
    header = {
        'identity': identity,
        'training_regime': task['regime'],
        'model_revision': model_revision(model_path),
        'model_path': str(model_path),
        'data_paths': data_paths,
        'data_counts': {k: len(v) for k, v in data.items()},
        'seed': 20,
        'config': config,
        'gpu': args.gpu,
        'gpu_name': torch.cuda.get_device_name(args.gpu),
        'torch_version': torch.__version__,
        'transformers_version': transformers.__version__,
        'started_at_unix': time.time(),
        'inference_batch_size': INFER_BATCH_SIZE,
        'max_new_tokens': MAX_NEW_TOKENS,
        'script_sha256': sha256_file(Path(__file__)),
        'source_labels_sha256': json.loads(manifest_path.read_text(encoding='utf-8'))['source_labels_sha256'],
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('tokenizer lacks pad and eos tokens')
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    forward_params = inspect.signature(model.forward).parameters
    keep_arg = next((key for key in ('logits_to_keep', 'num_logits_to_keep') if key in forward_params), None)
    header['logits_suffix_argument'] = keep_arg
    import peft
    from peft import LoraConfig, get_peft_model
    header['peft_version'] = peft.__version__
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=LORA_TARGETS, init_lora_weights=True))
    adapter_stats = {}
    for name, parameter in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            values = parameter.detach().float()
            adapter_stats[name] = {
                'shape': list(values.shape),
                'mean': values.mean().item(),
                'std': values.std(unbiased=False).item(),
                'nonzero': torch.count_nonzero(values).item(),
            }
    header['initial_adapter_statistics'] = adapter_stats
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    encoded_train = encode_causal(tokenizer, data['train'])
    prompt_digest = hashlib.sha256()
    truncated_count = 0
    for row, encoded in zip(data['train'], encoded_train):
        text = prompt_text(tokenizer, row)
        prompt_digest.update(json.dumps([row['sample_id'], text], ensure_ascii=False).encode('utf-8') + b'\n')
        truncated_count += len(tokenizer.encode(text, add_special_tokens=False)) > len(encoded['prompt_ids'])
    header['train_prompt_audit'] = {
        'rendered_prompt_sha256': prompt_digest.hexdigest(),
        'truncated_count': truncated_count,
        'count': len(encoded_train),
        'truncated_fraction': truncated_count / len(encoded_train),
    }
    model.to(device)
    header['generation_config'] = model.generation_config.to_dict()
    header['tokenizer_eos_token_id'] = tokenizer.eos_token_id
    header['tokenizer_pad_token_id'] = tokenizer.pad_token_id
    header['trainable_parameters'] = sum((p.numel() for p in model.parameters() if p.requires_grad))
    atomic_json(output_dir / 'header.json', header)
    batch_size = config['batch_size']
    accum = config['gradient_accumulation']
    batches_per_epoch = math.ceil(len(encoded_train) / batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / accum)
    total_steps = config['epochs'] * steps_per_epoch
    warmup_steps = max(1, math.ceil(total_steps * config['warmup_ratio']))
    max_steps = total_steps
    header['total_optimizer_steps'] = total_steps
    header['warmup_steps'] = warmup_steps
    atomic_json(output_dir / 'header.json', header)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config['learning_rate'], weight_decay=config['weight_decay'], betas=(0.9, args.beta2), eps=1e-08)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        fraction = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, fraction)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    completed_step = 0
    start_epoch = 0
    start_batch = 0
    best_step = None
    best_accuracy = -1.0
    history = []
    checkpoints = available_checkpoints(output_dir)
    if checkpoints:
        completed_step, checkpoint, state = checkpoints[-1]
        if state['identity'] != identity:
            raise RuntimeError('checkpoint manifest/data/task hash mismatch')
        if completed_step > max_steps:
            raise RuntimeError('checkpoint exceeds requested steps')
        checkpoint_for_step(output_dir, state['best_step'], identity)
        load_checkpoint_weights(model, checkpoint / 'model', task['regime'])
        saved = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        restore_random_state(saved['random'], torch)
        start_epoch, start_batch = (state['next_epoch'], state['next_batch'])
        best_step, best_accuracy = (state['best_step'], state['best_validation_accuracy'])
        history = state['validation_history']
    validation_rows = data['validation']
    for epoch in range(start_epoch, config['epochs']):
        order = list(range(len(encoded_train)))
        random.Random(20 + epoch).shuffle(order)
        batch_start = start_batch if epoch == start_epoch else 0
        while batch_start < batches_per_epoch:
            if completed_step >= max_steps:
                break
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            group_end = min(batch_start + accum, batches_per_epoch)
            group_size = group_end - batch_start
            for batch_idx in range(batch_start, group_end):
                positions = order[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                items = [encoded_train[index] for index in positions]
                batch = collate_causal(items, tokenizer.pad_token_id, torch, device)
                targets = batch.pop('labels')
                completion_eos_positions = batch.pop('completion_eos_positions')
                keep = max((item['target_tokens'] for item in items)) + 1
                extra = {keep_arg: keep} if keep_arg else {}
                logits = model(**batch, **extra).logits
                loss = weighted_completion_loss(logits, targets, completion_eos_positions, torch, args.eos_weight, args.label_smoothing)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'nonfinite loss at step {completed_step + 1}')
                micro_losses.append(float(loss.detach()))
                (loss / group_size).backward()
                del loss, batch
                del logits, completion_eos_positions, targets
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config['gradient_clip_norm'])
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError(f'nonfinite gradient at step {completed_step + 1}')
            grad_norm_value = float(grad_norm)
            del grad_norm
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_step += 1
            batch_start = group_end
            epoch_done = batch_start == batches_per_epoch
            formal_final = completed_step == total_steps
            evaluate_now = completed_step % 100 == 0 or formal_final
            validation = None
            if evaluate_now:
                validation = evaluate(model, tokenizer, validation_rows, task['kind'], torch, device)
                history.append({'step': completed_step, 'epoch': epoch + 1, 'metrics': validation})
            save_now = completed_step % 200 == 0 or formal_final
            if save_now:
                if validation is None:
                    validation = evaluate(model, tokenizer, validation_rows, task['kind'], torch, device)
                    history.append({'step': completed_step, 'epoch': epoch + 1, 'metrics': validation})
                if validation['accuracy'] > best_accuracy:
                    best_accuracy, best_step = (validation['accuracy'], completed_step)
                next_epoch = epoch + 1 if epoch_done else epoch
                next_batch = 0 if epoch_done else batch_start
                state = {
                    'step': completed_step,
                    'next_epoch': next_epoch,
                    'next_batch': next_batch,
                    'best_step': best_step,
                    'best_validation_accuracy': best_accuracy,
                    'validation_history': history,
                }
                save_checkpoint(output_dir, model, tokenizer, optimizer, scheduler, torch, state, identity)
            progress = {
                'identity': identity,
                'step': completed_step,
                'total_steps': total_steps,
                'epoch': epoch + 1,
                'batch': batch_start,
                'loss': sum(micro_losses) / len(micro_losses),
                'gradient_norm': grad_norm_value,
                'learning_rate': scheduler.get_last_lr()[0],
                'validation': validation,
                'best_step': best_step,
                'updated_at_unix': time.time(),
            }
            atomic_json(output_dir / 'progress.json', progress)
            if completed_step == 1 or completed_step % 10 == 0 or evaluate_now or save_now:
                print(json.dumps({'task_id': task['id'], 'step': completed_step, 'total_steps': total_steps, 'loss': progress['loss'], 'validation_accuracy': validation['accuracy'] if validation else None}), flush=True)
        if completed_step >= max_steps:
            break
    if best_step is None:
        raise RuntimeError('no checkpoint selected')
    atomic_json(output_dir / 'training_done.json', {'status': 'completed', 'identity': identity, 'manifest_sha256': manifest_hash, 'data_sha256': data_hashes, 'best_checkpoint_step': best_step, 'selection_metric': 'validation_accuracy_at_saved_checkpoint', 'validation_history': history, 'finished_at_unix': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--infer-batch-size', type=int, default=16, choices=(16,))
    parser.add_argument('--max-new-tokens', type=int, default=8, choices=(8,))
    parser.add_argument('--beta2', type=float, default=0.98, choices=(0.98,))
    parser.add_argument('--init', default='default', choices=('default',))
    parser.add_argument('--eos-weight', type=float, default=1.0, choices=(1.0,))
    parser.add_argument('--label-smoothing', type=float, default=0.0, choices=(0.0,))
    parser.add_argument('--template-id', default='P0', choices=('P0',))
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        try:
            manifest_path, task, _, output_dir, _, _, _ = load_job(args)
            if (output_dir / 'training_done.json').exists():
                pass
            else:
                atomic_json(output_dir / 'error.json', {'status': 'error', 'task_id': args.task_id, 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(), 'manifest': str(manifest_path), 'time_unix': time.time()})
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
        print(f'Training task failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
