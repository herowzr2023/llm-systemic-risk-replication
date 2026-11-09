#!/usr/bin/env python3
"""Prepare the fixed sentiment splits and model tasks from local model files."""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1]
DATA = PACKAGE / 'data-raw/sentiment_classification'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def hashes(path):
    result = {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
    if not result:
        raise ValueError(f'empty model cache: {path}')
    return result

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def normalized(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).casefold()

def resolve_model(model, cache_root, mappings):
    model_id, revision = model['model_id'], model['revision']
    if model_id in mappings:
        path = Path(mappings[model_id]).expanduser().resolve(strict=True)
    else:
        if cache_root is None:
            raise ValueError(f'provide --cache-root or mapping for {model_id}')
        base = cache_root / ('models--' + model_id.replace('/', '--'))
        path = base / 'snapshots' / revision
        if not path.is_dir():
            ref = base / 'refs' / revision
            if ref.is_file():
                revision = ref.read_text().strip()
                path = base / 'snapshots' / revision
        path = path.resolve(strict=True)
    if not path.is_dir() or (path / 'adapter_config.json').exists():
        raise ValueError(f'not a clean base model directory: {path}')
    return path, revision, hashes(path)

def load_data(labels_path, split_ids_path):
    labels_path = Path(labels_path).resolve(strict=True)
    if labels_path == PACKAGE or PACKAGE in labels_path.parents:
        raise ValueError('sentiment training labels must be outside the replication package')
    with labels_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['sample_id', 'bank', 'title', 'final_label']:
            raise ValueError('labels columns must be sample_id,bank,title,final_label')
        rows = list(reader)
    by_id = {}
    for row in rows:
        sid = row['sample_id']
        if not sid or sid in by_id or row['final_label'] not in ('-1', '0', '1'):
            raise ValueError('duplicate/empty ID or invalid label')
        by_id[sid] = dict(sample_id=sid, bank=row['bank'], title=row['title'], label=int(row['final_label']))
    split = read(split_ids_path)
    if set(split) != {'train', 'validation', 'test'}:
        raise ValueError('split IDs need exactly train/validation/test')
    counts = {'train': 7000, 'validation': 2000, 'test': 1000}
    data, seen, groups = {}, set(), {}
    for name, count in counts.items():
        ids = split[name]
        if not isinstance(ids, list) or len(ids) != count:
            raise ValueError(f'{name} requires {count} ordered IDs')
        data[name] = []
        for raw in ids:
            sid = str(raw)
            if sid in seen or sid not in by_id:
                raise ValueError(f'overlapping or unknown ID: {sid}')
            seen.add(sid)
            row = by_id[sid]
            group = (normalized(row['bank']), normalized(row['title']))
            if group in groups and groups[group] != name:
                raise ValueError('normalized bank/title group crosses splits')
            groups[group] = name
            data[name].append(row)
    if seen != set(by_id):
        raise ValueError('labels and split IDs must have identical coverage')
    return data


def load_configs(args):
    configs = {}
    if args.families in ('llm', 'both'):
        configs['llm'] = read(args.llm_config)
    if args.families in ('bert', 'both'):
        configs['bert'] = read(args.bert_config)
    available = [t['id'] for t in configs.get('llm', {}).get('tasks', [])]
    available += ['bert_' + t['model_key'] for t in configs.get('bert', {}).get('models', [])]
    selected = getattr(args, 'task_ids', None)
    if selected:
        if len(selected) != len(set(selected)) or set(selected) - set(available):
            raise ValueError('task IDs must be unique and belong to the configured models')
        if 'llm' in configs:
            configs['llm']['tasks'] = [t for t in configs['llm']['tasks'] if t['id'] in selected]
        if 'bert' in configs:
            configs['bert']['models'] = [t for t in configs['bert']['models'] if 'bert_' + t['model_key'] in selected]
    return configs


def prepare(args):
    run = args.run_dir.expanduser().resolve()
    if run == PACKAGE or PACKAGE in run.parents:
        raise ValueError('run directory must be outside the replication package')
    if run.exists():
        raise FileExistsError('run directory must be new; use a fresh --run-dir')
    labels_path = args.labels.resolve(strict=True)
    data = load_data(labels_path, args.split_ids)
    counts = {'train': 7000, 'validation': 2000, 'test': 1000}
    mappings = read(args.model_paths) if args.model_paths else {}
    cache = args.cache_root.expanduser().resolve() if args.cache_root else None
    configs = load_configs(args)
    if 'llm' in configs and configs['llm']['seed'] != 20:
        raise ValueError('LLM seed must be 20')
    ids = [t['id'] for t in configs.get('llm', {}).get('tasks', [])]
    ids += ['bert_' + t['model_key'] for t in configs.get('bert', {}).get('models', [])]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', x) or x in ('.', '..') for x in ids):
        raise ValueError('task IDs must be unique safe path components')
    resolved = {}
    for family, config in configs.items():
        for model in config['tasks' if family == 'llm' else 'models']:
            key = (model['model_id'], model['revision'])
            if key not in resolved:
                resolved[key] = resolve_model(model, cache, mappings)
    # All validation/cache checks happen before creating the requested output.
    run.mkdir(parents=True)
    files = {name: run / 'data' / ('held_out.json' if name == 'test' else name + '.json') for name in counts}
    for name, path in files.items():
        write(path, data[name])
    data_hashes = {name: sha(path) for name, path in files.items()}
    evidence = {}
    for index, (key, (path, revision, model_hashes)) in enumerate(resolved.items()):
        ready_path = run / 'cache_evidence' / f'model_{index:02d}.json'
        write(ready_path, dict(status='ready', model_id=key[0], revision=revision, path=str(path), files_sha256=model_hashes))
        evidence[key] = ready_path
    registry = []
    if 'llm' in configs:
        config = configs['llm']
        if config['seed'] != 20:
            raise ValueError('LLM seed must be 20')
        tasks = []
        for item in config['tasks']:
            task = dict(item)
            key = (task['model_id'], task['revision'])
            path, revision, _ = resolved[key]
            task.update(model_path=str(path), revision=revision, ready_file=str(evidence[key]), output_dir=str(run / 'outputs' / task['id']))
            tasks.append(task)
            registry.append(dict(id=task['id'], display_name=task['display_name'], model_id=task['model_id'], family='llm', regime=task['regime'], output_dir=task['output_dir'], train_args=task.get('train_args', {})))
        write(run / 'llm_manifest.json', dict(seed=20, tasks=tasks, source_labels_file=str(labels_path), source_labels_sha256=sha(labels_path), data_dir=str(run / 'data'), train_file=str(files['train']), validation_file=str(files['validation']), test_file=str(files['test']), data_sha256={'train': data_hashes['train'], 'validation': data_hashes['validation'], 'held_out': data_hashes['test']}))
    if 'bert' in configs:
        config = configs['bert']
        for item in config['models']:
            key = (item['model_id'], item['revision'])
            path, revision, model_hashes = resolved[key]
            task_id = 'bert_' + item['model_key']
            task = {**config['training'], **item, 'revision': revision, 'model_path': str(path), 'output_dir': str(run / 'outputs' / task_id / 'training'), 'train_file': str(files['train']), 'validation_file': str(files['validation']), 'model_files_sha256': model_hashes, 'data_files_sha256': {k: data_hashes[k] for k in ('train', 'validation')}}
            task_file = run / 'tasks' / (task_id + '.json')
            write(task_file, task)
            registry.append(dict(id=task_id, display_name=item['display_name'], model_id=item['model_id'], family='bert', task_file=str(task_file), output_dir=str(run / 'outputs' / task_id)))
    if len({t['id'] for t in registry}) != len(registry):
        raise ValueError('task IDs must be globally unique')
    write(run / 'run.json', dict(tasks=registry, data_files={k: str(v) for k, v in files.items()}, data_sha256=data_hashes, source_labels_sha256=sha(labels_path), split_ids_sha256=sha(args.split_ids)))
    return run

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--cache-root', type=Path)
    p.add_argument('--model-paths', type=Path, help='JSON mapping model_id to local directory')
    p.add_argument('--families', choices=('llm', 'bert', 'both'), default='both')
    p.add_argument('--task-ids', nargs='+')
    p.add_argument('--labels', required=True, type=Path,
                   help='External CSV with sample_id,bank,title,final_label; sample IDs must match the public fixed split')
    p.add_argument('--split-ids', type=Path, default=DATA / 'split_ids.json')
    p.add_argument('--llm-config', type=Path, default=ROOT / 'configs/llm.json')
    p.add_argument('--bert-config', type=Path, default=ROOT / 'bert/config.json')
    print(prepare(p.parse_args()))

if __name__ == '__main__':
    main()
