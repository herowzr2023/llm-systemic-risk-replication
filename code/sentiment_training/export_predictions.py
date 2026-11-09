"""Export paired reference and predicted sentiment labels for each classifier."""
import csv
import hashlib
import json
import os
from pathlib import Path

from prediction_parser import parse
from prepare import PACKAGE, read, sha

FIELDS = ('model', 'split', 'sample_id', 'true_label', 'predicted_label')


def prediction_path(task, split):
    output = Path(task['output_dir'])
    if task['family'] == 'bert':
        return output / ('evaluation_' + split) / 'predictions_evaluation.jsonl'
    if task['regime'] == 'lora':
        return output / 'evaluation_methods_P0' / f'predictions_{split}_sampling_seed_1002.jsonl'
    return output / f'predictions_{split}.jsonl'


def model_rows(task, split, truth):
    path = prediction_path(task, split)
    if task['family'] == 'bert':
        done_path = path.parent / 'done.json'
        if not done_path.is_file():
            raise FileNotFoundError(f'{task["id"]}/{split}: missing BERT completion record: {done_path}')
        done = read(done_path)
        if done.get('status') != 'completed':
            raise ValueError(f'{task["id"]}/{split}: BERT evaluation is not completed: {done_path}')
        if done.get('predictions_sha256') != sha(path):
            raise ValueError(f'{task["id"]}/{split}: BERT predictions SHA256 mismatch: {path}')
    with path.open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if [r['sample_id'] for r in records] != [r['sample_id'] for r in truth]:
        raise ValueError(f'{task["id"]}/{split}: prediction IDs or order differ from the fixed split')
    exported = []
    for row, record in zip(truth, records, strict=True):
        fingerprint = hashlib.sha256(f'{row["bank"]}\n{row["title"]}'.encode('utf-8')).hexdigest()
        if record['reference'] != row['label'] or record['bank_title_sha256'] != fingerprint:
            raise ValueError(f'{task["id"]}/{split}: reference label or input text differs')
        if task['family'] == 'bert':
            prediction = record['prediction']
        else:
            prediction = parse(record['raw_output'])[0]
        if type(prediction) is not int or prediction not in (-1, 0, 1):
            raise ValueError(f'{task["id"]}/{split}: invalid predicted label')
        exported.append(dict(model=task['display_name'], split=split, sample_id=row['sample_id'],
                             true_label=row['label'], predicted_label=prediction))
    return exported


def export_predictions(run_dir, task_ids=None):
    run = Path(run_dir).resolve(strict=True)
    if run == PACKAGE or PACKAGE in run.parents:
        raise ValueError('run directory must be outside the replication package')
    registry = read(run / 'run.json')
    tasks = registry['tasks']
    if task_ids is not None:
        if len(task_ids) != len(set(task_ids)) or set(task_ids) - {t['id'] for t in tasks}:
            raise ValueError('export task IDs must uniquely identify prepared models')
        tasks = [t for t in tasks if t['id'] in task_ids]
    if not tasks:
        raise ValueError('no models selected for export')
    data = {}
    for split, count in (('train', 7000), ('test', 1000)):
        path = Path(registry['data_files'][split])
        if sha(path) != registry['data_sha256'][split]:
            raise ValueError(f'{split}: data SHA256 differs from the prepared split')
        data[split] = read(path)
        if len(data[split]) != count or len({r['sample_id'] for r in data[split]}) != count:
            raise ValueError(f'{split}: incorrect sample count or duplicate IDs')
    if {r['sample_id'] for r in data['train']} & {r['sample_id'] for r in data['test']}:
        raise ValueError('training and test samples overlap')
    rows = [row for task in tasks for split in ('train', 'test')
            for row in model_rows(task, split, data[split])]
    target = run / 'sentiment_classification_predictions.csv'
    temporary = target.with_name(target.name + f'.tmp-{os.getpid()}')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    return dict(output=str(target), models=len(tasks), rows=len(rows),
                train_rows=7000 * len(tasks), test_rows=1000 * len(tasks))
