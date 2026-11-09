#!/usr/bin/env python3
"""Train and evaluate the 12 sentiment classifiers and export their predictions."""
import argparse
from collections import Counter
import json
import subprocess
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent / 'sentiment_training'
sys.path.insert(0, str(TRAINING))
import prepare
from export_predictions import export_predictions


def task_commands(task, args, run):
    untuned = task['family'] == 'llm' and task['regime'] == 'untuned'
    if args.mode == 'train' and untuned:
        return []
    stages = ('eval',) if untuned else ('train', 'eval') if args.mode == 'all' else (args.mode,)
    commands = []
    for stage in stages:
        command = [sys.executable, '-B', str(TRAINING / 'run_task.py'), '--run-dir', str(run),
                   '--task-id', task['id'], '--gpu', str(args.gpu), '--mode', stage]
        if args.cpu_offload and task['family'] == 'llm' and task['regime'] == 'lora' and stage == 'eval':
            command += ['--cpu-offload']
        commands.append(command)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=prepare.PACKAGE.parent / 'sentiment_training_output')
    parser.add_argument('--labels', required=True, type=Path,
                        help='External CSV with sample_id,bank,title,final_label; sample IDs must match the public fixed split')
    parser.add_argument('--cache-root', type=Path, help='Hugging Face hub cache directory')
    parser.add_argument('--model-paths', type=Path, help='JSON mapping model IDs to local directories')
    parser.add_argument('--task-id', nargs='+', dest='task_ids', help='Restrict execution to these model task IDs')
    parser.add_argument('--gpu', type=int, default=0, help='Physical CUDA GPU index')
    parser.add_argument('--mode', choices=('all', 'train', 'eval', 'export'), default='all')
    parser.add_argument('--cpu-offload', action='store_true', help='Offload LoRA inference layers to CPU')
    parser.add_argument('--dry-run', action='store_true', help='Validate data and display the model plan without loading models')
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    args.split_ids = prepare.DATA / 'split_ids.json'
    args.llm_config = TRAINING / 'configs/llm.json'
    args.bert_config = TRAINING / 'bert/config.json'
    args.families = 'both'
    data = prepare.load_data(args.labels, args.split_ids)
    configs = prepare.load_configs(args)
    run = args.run_dir.expanduser().resolve()
    if run == prepare.PACKAGE or prepare.PACKAGE in run.parents:
        parser.error('--run-dir must be outside the replication package')
    if args.dry_run:
        tasks = [dict(task_id=t['id'], model=t['display_name'], regime=t['regime'],
                      training_seed=20 if t['regime'] == 'lora' else None,
                      inference='sampling_seed_1002' if t['regime'] == 'lora' else 'greedy')
                 for t in configs['llm']['tasks']]
        tasks += [dict(task_id='bert_' + t['model_key'], model=t['display_name'],
                       regime='supervised', training_seed=123, inference='argmax')
                  for t in configs['bert']['models']]
        print(json.dumps(dict(split_seed=42,
                              splits={k: dict(count=len(v), class_counts=dict(Counter(r['label'] for r in v)))
                                      for k, v in data.items()},
                              model_count=len(tasks), model_fits=sum(t['regime'] != 'untuned' for t in tasks),
                              prediction_rows=8000 * len(tasks), tasks=tasks, run_dir=str(run)),
                         ensure_ascii=False, indent=2))
        return
    if not (run / 'run.json').is_file():
        if args.mode == 'export':
            raise FileNotFoundError(run / 'run.json')
        prepare.prepare(args)
    registry = prepare.read(run / 'run.json')
    if (registry['source_labels_sha256'] != prepare.sha(args.labels)
            or registry['split_ids_sha256'] != prepare.sha(args.split_ids)):
        raise ValueError('run data differ from the supplied sentiment labels or fixed split')
    tasks = registry['tasks']
    if args.task_ids:
        if set(args.task_ids) - {t['id'] for t in tasks}:
            raise ValueError('requested task is absent from the prepared run')
        tasks = [t for t in tasks if t['id'] in args.task_ids]
    if args.mode != 'export':
        for task in tasks:
            for command in task_commands(task, args, run):
                stage = command[command.index('--mode') + 1]
                print(f"{task['display_name']}: {stage}", flush=True)
                subprocess.run(command, check=True)
    if args.mode in ('all', 'eval', 'export'):
        result = export_predictions(run, [t['id'] for t in tasks])
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
