#!/usr/bin/env python3
"""Run untuned Llama sentiment inference with greedy decoding."""

import argparse
import hashlib
import json
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


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    source_labels = manifest.get('source_labels_file')
    source_hash = manifest.get('source_labels_sha256')
    if not isinstance(source_labels, str) or not source_labels or (not isinstance(source_hash, str)):
        raise ValueError('manifest needs source_labels_file and source_labels_sha256')
    if sha256_file(resolve_path(manifest_path.parent, source_labels)) != source_hash:
        raise ValueError('source labels SHA-256 mismatch')
    for key in ('model_id', 'model_path', 'kind', 'regime', 'output_dir'):
        if not isinstance(task.get(key), str) or not task[key]:
            raise ValueError(f'task needs nonempty {key}')
    if task['kind'] != 'causal' or task['regime'] != 'untuned':
        raise ValueError('this runner requires causal untuned tasks')
    expected_epochs = 0
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


def prompt_text(tokenizer, row):
    system = "判断银行股吧标题的情绪。只输出负面、中性或正面。"
    user = f"银行：{row['bank']}\n标题：{row['title']}"
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
    return f"指令：{system}\n用户：{user}\n回答："


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
        raise ValueError("causal inference required")
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
    if kind != "causal" or regime != "untuned":
        raise ValueError("causal untuned inference required")
    return {"epochs": 0, "max_length": 1024, "precision": "bf16",
            "inference_batch_size": 16, "max_new_tokens": 8}


def model_revision(model_path):
    config = model_path / "config.json"
    if not config.exists():
        return None
    data = json.loads(config.read_text(encoding="utf-8"))
    return data.get("_commit_hash") or data.get("revision")


def run(args):
    global INFER_BATCH_SIZE, MAX_NEW_TOKENS
    if args.infer_batch_size != 16 or args.max_new_tokens != 8:
        raise ValueError('inference uses batch16 and max_new_tokens8')
    INFER_BATCH_SIZE = args.infer_batch_size
    MAX_NEW_TOKENS = args.max_new_tokens
    manifest_path, task, model_path, output_dir, data, data_paths, data_hashes = load_job(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'done.json').exists():
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
    manifest_hash = sha256_file(manifest_path)
    identity = {
        'manifest_sha256': manifest_hash,
        'data_sha256': data_hashes,
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
    model.to(device)
    header['trainable_parameters'] = sum((p.numel() for p in model.parameters() if p.requires_grad))
    atomic_json(output_dir / 'header.json', header)
    model.requires_grad_(False)
    metrics = {split: evaluate(model, tokenizer, rows, task['kind'], torch, device, output_dir / f'predictions_{split}.jsonl') for split, rows in data.items() if split in ('train', 'test')}
    atomic_json(output_dir / 'metrics.json', {'header': header, 'splits': metrics, 'selection': 'untuned'})
    atomic_json(output_dir / 'done.json', {'status': 'completed', 'identity': identity, 'manifest_sha256': manifest_hash, 'data_sha256': data_hashes, 'finished_at_unix': time.time()})
    return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--infer-batch-size', type=int, default=16, choices=(16,))
    parser.add_argument('--max-new-tokens', type=int, default=8, choices=(8,))
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        try:
            manifest_path, task, _, output_dir, _, _, _ = load_job(args)
            if (output_dir / 'done.json').exists():
                pass
            else:
                atomic_json(output_dir / 'error.json', {'status': 'error', 'task_id': args.task_id, 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(), 'manifest': str(manifest_path), 'time_unix': time.time()})
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
        print(f'Inference task failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
