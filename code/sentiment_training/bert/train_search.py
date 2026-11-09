#!/usr/bin/env python3
"""Train one offline BERT classifier or evaluate its selected checkpoint.

Task paths resolve relative to the task JSON. Training selects the highest
validation accuracy, retaining the earliest epoch on exact ties.
Use --evaluate-only to evaluate a saved checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def weighted_metrics(refs, preds):
    if not refs or len(refs) != len(preds):
        raise ValueError('empty or mismatched predictions')
    n = len(refs)
    per_class = {}
    for label in (-1, 0, 1):
        tp = sum(a == label and b == label for a, b in zip(refs, preds))
        fp = sum(a != label and b == label for a, b in zip(refs, preds))
        fn = sum(a == label and b != label for a, b in zip(refs, preds))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_class[str(label)] = dict(support=tp + fn, precision=p, recall=r,
                                    f1=2 * p * r / (p + r) if p + r else 0.0)
    return dict(count=n, accuracy=sum(a == b for a, b in zip(refs, preds)) / n,
                macro_f1=sum(x['f1'] for x in per_class.values()) / 3,
                **{'weighted_' + k: sum(x['support'] * x[k] for x in per_class.values()) / n
                   for k in ('precision', 'recall', 'f1')}, per_class=per_class)


def input_text(row, mode='bank_title'):
    if mode == 'title_only':
        return row['title']
    if mode != 'bank_title':
        raise ValueError('invalid input_mode')
    return f"银行：{row['bank']}\n标题：{row['title']}"


def read_rows(path):
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('split must be a nonempty JSON array')
    ids = set()
    for r in rows:
        if not all(k in r for k in ('sample_id', 'bank', 'title', 'label')):
            raise ValueError('missing row field')
        if not isinstance(r['bank'], str) or not isinstance(r['title'], str):
            raise ValueError('bank/title must be text')
        if type(r['label']) is not int or r['label'] not in LABEL_TO_ID:
            raise ValueError('invalid label')
        if r['sample_id'] in ids:
            raise ValueError('duplicate sample_id')
        ids.add(r['sample_id'])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', required=True, type=Path)
    ap.add_argument('--evaluate-only', action='store_true')
    args = ap.parse_args()
    task_path = args.task.resolve(strict=True)
    task = json.loads(task_path.read_text(encoding='utf-8'))
    base = task_path.parent
    def path(key):
        p = Path(task[key])
        return (p if p.is_absolute() else base / p).resolve()
    required = ('output_dir', 'seed', 'max_length', 'precision', 'model_files_sha256', 'data_files_sha256')
    required += ('selected_model', 'eval_file') if args.evaluate_only else (
        'model_path', 'model_id', 'train_file', 'validation_file', 'lr', 'epochs',
        'train_batch', 'grad_accum', 'unfreeze_last_n')
    for key in required:
        if key not in task:
            raise ValueError(f'missing task key: {key}')
    if task['max_length'] not in (128, 256, 512) or task['precision'] not in ('bf16', 'fp32'):
        raise ValueError('max_length must be 128/256/512 and precision bf16/fp32')
    if task.get('padding', 'dynamic') not in ('dynamic', 'max_length'):
        raise ValueError('padding must be dynamic/max_length')
    selection_metric = task.get('selection_metric', 'accuracy')
    if selection_metric not in ('accuracy', 'loss'):
        raise ValueError('selection_metric must be accuracy/loss')
    if type(task['seed']) is not int:
        raise ValueError('seed must be integer')
    if task.get('input_mode', 'bank_title') not in ('bank_title', 'title_only'):
        raise ValueError('invalid input_mode')
    out = path('output_dir')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('output directory is nonempty; use a new output_dir')
    model_path = path('selected_model' if args.evaluate_only else 'model_path')
    if not model_path.is_dir():
        raise ValueError('local model directory missing')
    model_hashes = {p.relative_to(model_path).as_posix(): sha(p) for p in sorted(model_path.rglob('*')) if p.is_file()}
    if not model_hashes:
        raise ValueError('model directory empty')
    expected = task['model_files_sha256']
    if not isinstance(expected, dict) or not expected or model_hashes != expected:
        raise ValueError('model cache file set or SHA256 mismatch')
    split_keys = {'evaluation': 'eval_file'} if args.evaluate_only else {'train': 'train_file', 'validation': 'validation_file'}
    splits = {name: read_rows(path(key)) for name, key in split_keys.items()}
    if not args.evaluate_only:
        if {r['sample_id'] for r in splits['train']} & {r['sample_id'] for r in splits['validation']}:
            raise ValueError('train/validation overlap')
        if any(type(task[k]) is not int or task[k] < 1 for k in ('train_batch', 'grad_accum')) or task['lr'] <= 0 or task['epochs'] <= 0:
            raise ValueError('invalid training hyperparameters')
    hashes = {name: sha(path(key)) for name, key in split_keys.items()}
    expected_data = task['data_files_sha256']
    if not isinstance(expected_data, dict) or hashes != expected_data:
        raise ValueError('data split names or SHA256 mismatch')
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('expose exactly one CUDA GPU')
    if task['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support bf16')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(task['seed'])
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True,
        num_labels=3, torch_dtype=torch.float32, attn_implementation='sdpa',
        id2label={0: '负面', 1: '中性', 2: '正面'}, label2id={'负面': 0, '中性': 1, '正面': 2})
    layer_indices = []
    if not args.evaluate_only:
        if not hasattr(model, 'bert'):
            raise TypeError('requires BERT model.bert')
        n = task['unfreeze_last_n']
        layers = model.bert.encoder.layer
        if type(n) is not int or n < -1:
            raise ValueError('invalid unfreeze_last_n')
        requested_n = n
        if n != -1:
            n = min(n, len(layers))
        if n != -1:
            for p in model.bert.parameters():
                p.requires_grad = False
            layer_indices = list(range(len(layers) - n, len(layers)))
            for i in layer_indices:
                for p in layers[i].parameters():
                    p.requires_grad = True
            if model.bert.pooler is not None:
                for p in model.bert.pooler.parameters():
                    p.requires_grad = True
        else:
            layer_indices = list(range(len(layers)))
    class TextRows(Dataset):
        def __init__(self, rows):
            self.items = []
            for r in rows:
                item = tokenizer(input_text(r, task.get('input_mode', 'bank_title')), truncation=True, max_length=task['max_length'])
                item['labels'] = LABEL_TO_ID[r['label']]
                self.items.append(item)
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]
    datasets = {k: TextRows(v) for k, v in splits.items()}
    def compute_metrics(result):
        logits, refs = result
        m = weighted_metrics([ID_TO_LABEL[int(x)] for x in refs],
                             [ID_TO_LABEL[int(x)] for x in np.asarray(logits).argmax(-1)])
        return {k: v for k, v in m.items() if k not in ('count', 'per_class')}
    out.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(output_dir=str(out / 'checkpoints'),
        per_device_train_batch_size=task.get('train_batch', 1),
        per_device_eval_batch_size=task.get('eval_batch', 64),
        gradient_accumulation_steps=task.get('grad_accum', 1),
        num_train_epochs=task.get('epochs', 1), learning_rate=task.get('lr', 2e-5),
        weight_decay=task.get('weight_decay', 0.01), warmup_ratio=task.get('warmup_ratio', 0.1),
        lr_scheduler_type=task.get('scheduler', 'linear'), optim='adamw_torch',
        eval_strategy='no' if args.evaluate_only else 'epoch',
        save_strategy='no' if args.evaluate_only else 'epoch', save_total_limit=None,
        save_only_model=True, load_best_model_at_end=not args.evaluate_only,
        metric_for_best_model=selection_metric, greater_is_better=selection_metric != 'loss',
        bf16=task['precision'] == 'bf16', fp16=False, tf32=False,
        seed=task['seed'], data_seed=task['seed'], report_to='none', save_safetensors=True)
    class ExperimentTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                super().create_optimizer()
                if task.get('optimizer', 'adamw_torch') == 'sgd':
                    groups = [dict(params=g['params'], weight_decay=g.get('weight_decay', 0.0)) for g in self.optimizer.param_groups]
                    self.optimizer = torch.optim.SGD(groups, lr=task['lr'], momentum=task.get('momentum', 0.9))
                elif task.get('optimizer', 'adamw_torch') != 'adamw_torch':
                    raise ValueError('unsupported optimizer')
            return self.optimizer
    trainer = ExperimentTrainer(model=model, args=training_args, train_dataset=datasets.get('train'),
        eval_dataset=datasets.get('validation'), data_collator=DataCollatorWithPadding(tokenizer,
            padding='max_length' if task.get('padding', 'dynamic') == 'max_length' else True,
            max_length=task['max_length'] if task.get('padding', 'dynamic') == 'max_length' else None),
        compute_metrics=compute_metrics)
    observed_optimizer = None
    if not args.evaluate_only:
        optimizer = trainer.create_optimizer()
        expected_class = torch.optim.SGD if task.get('optimizer', 'adamw_torch') == 'sgd' else torch.optim.AdamW
        if not isinstance(optimizer, expected_class):
            raise TypeError(f'optimizer class mismatch: {type(optimizer)} expected {expected_class}')
        observed_optimizer = dict(observed_optimizer_class=f'{type(optimizer).__module__}.{type(optimizer).__name__}',
                                  defaults_lr=optimizer.defaults.get('lr'), defaults_momentum=optimizer.defaults.get('momentum'),
                                  parameter_group_lr=[g['lr'] for g in optimizer.param_groups])
    header = dict(runner_sha256=sha(Path(__file__)), requested_unfreeze_last_n=None if args.evaluate_only else requested_n, actual_unfreeze_layer_count=None if args.evaluate_only else len(layer_indices), observed_optimizer=observed_optimizer, task=task, task_sha256=sha(task_path), input_mode=task.get('input_mode', 'bank_title'), actual_optimizer=task.get('optimizer', 'adamw_torch'), optimizer_momentum=task.get('momentum') if task.get('optimizer') == 'sgd' else None,
        model_id=task.get('model_id'), model_path=str(model_path), model_files_sha256=model_hashes,
        data_sha256=hashes, row_counts={k: len(v) for k, v in splits.items()},
        selected_layer_indices=layer_indices, label_map=LABEL_TO_ID,
        trainable_parameter_names=[k for k, p in model.named_parameters() if p.requires_grad],
        trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameter_count=sum(p.numel() for p in model.parameters()),
        selection_rule=('highest validation accuracy' if selection_metric == 'accuracy' else 'lowest validation loss') + '; earliest epoch on exact ties',
        evaluate_only=args.evaluate_only, training_args=training_args.to_dict(),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        gpu_name=torch.cuda.get_device_name(0), started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / 'header.json', header)
    train_metrics = None
    if not args.evaluate_only:
        train_metrics = trainer.train().metrics
        if trainer.state.best_model_checkpoint is None:
            raise RuntimeError('no validation-selected checkpoint')
        trainer.save_model(str(out / 'selected_model'))
        tokenizer.save_pretrained(str(out / 'selected_model'))
    name = 'evaluation' if args.evaluate_only else 'validation'
    pred = trainer.predict(datasets[name], metric_key_prefix=name)
    predicted = [ID_TO_LABEL[int(x)] for x in np.asarray(pred.predictions).argmax(-1)]
    metrics = weighted_metrics([r['label'] for r in splits[name]], predicted)
    metrics['loss'] = pred.metrics.get(name + '_loss')
    records = [dict(sample_id=r['sample_id'], reference=r['label'], prediction=p,
                    bank_title_sha256=hashlib.sha256(f"{r['bank']}\n{r['title']}".encode()).hexdigest(),
                    input_text_sha256=hashlib.sha256(input_text(r, task.get('input_mode', 'bank_title')).encode()).hexdigest())
               for r, p in zip(splits[name], predicted)]
    prediction_path = out / f'predictions_{name}.jsonl'
    prediction_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    write_json(out / 'metrics.json', dict(header=header, splits={name: metrics},
        best_checkpoint=trainer.state.best_model_checkpoint,
        best_validation_accuracy=trainer.state.best_metric if selection_metric == 'accuracy' else None,
        best_validation_loss=trainer.state.best_metric if selection_metric == 'loss' else None,
        train_metrics=train_metrics, training_log_history=trainer.state.log_history,
        observed_scheduler=None if args.evaluate_only or trainer.lr_scheduler is None else dict(
            scheduler_class=f'{type(trainer.lr_scheduler).__module__}.{type(trainer.lr_scheduler).__name__}',
            configured_schedule=task.get('scheduler', 'linear'), state=trainer.lr_scheduler.state_dict()),
        total_runtime_seconds=time.monotonic() - started))
    write_json(out / 'done.json', dict(status='completed', task_sha256=header['task_sha256'],
        data_sha256=hashes, metrics_sha256=sha(out / 'metrics.json'),
        predictions_sha256=sha(prediction_path), selected_model=str(model_path if args.evaluate_only else out / 'selected_model'),
        ended_at_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
