#!/usr/bin/env python3
"""Run one sentiment model task on one CUDA GPU."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from prepare import PACKAGE, ROOT, hashes, read, sha, write

def commands(run, task, gpu, mode, offload):
    base = [sys.executable, '-B']
    result = []
    if task['family'] == 'llm':
        common = ['--manifest', str(run / 'llm_manifest.json'), '--task-id', task['id'], '--gpu', '0']
        if task['regime'] == 'lora':
            if mode in ('train', 'all'):
                args = []
                for key, value in task['train_args'].items():
                    args += ['--' + key.replace('_', '-'), str(value)]
                result.append(base + [str(ROOT / 'llm/train_variants.py')] + common + args)
            if mode in ('eval', 'all'):
                result.append(base + [str(ROOT / 'llm/evaluate_methods_v3.py')] + common + (['--cpu-offload'] if offload else []))
        else:
            if offload:
                raise ValueError('untuned run_llm does not support CPU offload')
            if mode == 'train':
                raise ValueError('untuned tasks have no training mode; use eval/all')
            result.append(base + [str(ROOT / 'llm/run_llm.py')] + common)
    else:
        if offload:
            raise ValueError('BERT does not support CPU offload')
        if mode in ('train', 'all'):
            result.append(base + [str(ROOT / 'bert/train_search.py'), '--task', task['task_file']])
        if mode in ('eval', 'all'):
            for split in ('train', 'test'):
                result.append(base + [str(ROOT / 'bert/train_search.py'), '--task', str(run / 'tasks' / (task['id'] + '_eval_' + split + '.json')), '--evaluate-only'])
    return result

def bert_eval_tasks(run, task, registry):
    training = read(task['task_file'])
    done = read(Path(training['output_dir']) / 'done.json')
    if done.get('status') != 'completed' or done.get('task_sha256') != sha(task['task_file']):
        raise ValueError('BERT training completion provenance mismatch')
    selected = Path(done['selected_model']).resolve(strict=True)
    if selected != (Path(training['output_dir']) / 'selected_model').resolve():
        raise ValueError('selected BERT model escaped task output')
    model_hashes = hashes(selected)
    for split in ('train', 'test'):
        path = run / 'tasks' / (task['id'] + '_eval_' + split + '.json')
        evaluation = {**training, 'selected_model': str(selected), 'eval_file': registry['data_files'][split], 'output_dir': str(Path(task['output_dir']) / ('evaluation_' + split)), 'model_files_sha256': model_hashes, 'data_files_sha256': {'evaluation': registry['data_sha256'][split]}}
        if path.exists():
            if read(path) != evaluation:
                raise ValueError('existing evaluation task differs; use a fresh run directory')
        else:
            write(path, evaluation)

def done_paths(run, task, mode):
    output = Path(task['output_dir'])
    if task['family'] == 'bert':
        return ([output / 'training/done.json'] if mode in ('train', 'all') else []) + ([output / ('evaluation_' + split) / 'done.json' for split in ('train', 'test')] if mode in ('eval', 'all') else [])
    if task['regime'] == 'lora':
        return ([output / 'training_done.json'] if mode in ('train', 'all') else []) + ([output / 'evaluation_methods_P0/done.json'] if mode in ('eval', 'all') else [])
    return [output / 'done.json']

def completion_hashes(run, task, mode):
    paths = done_paths(run, task, mode)
    for path in paths:
        done = read(path)
        if done.get('status') != 'completed':
            raise ValueError(f'missing successful completion: {path}')
        if task['family'] == 'bert':
            prediction_name = ('predictions_validation.jsonl' if path.parent == Path(task['output_dir']) / 'training'
                               else 'predictions_evaluation.jsonl')
            for name, artifact in (('metrics_sha256', path.parent / 'metrics.json'),
                                   ('predictions_sha256', path.parent / prediction_name)):
                if done.get(name) != sha(artifact):
                    raise ValueError(f'BERT completion {name} mismatch: {artifact}')
    return {str(path.relative_to(run)): sha(path) for path in paths}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--task-id', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--mode', choices=('train', 'eval', 'all'), default='all')
    p.add_argument('--cpu-offload', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.gpu < 0:
        raise ValueError('GPU must be a nonnegative physical index')
    run = args.run_dir.resolve(strict=True)
    if run == PACKAGE or PACKAGE in run.parents:
        raise ValueError('run directory must be outside the replication package')
    registry = read(run / 'run.json')
    matches = [t for t in registry['tasks'] if t['id'] == args.task_id]
    if len(matches) != 1:
        raise ValueError('task ID must match exactly one prepared task')
    task = matches[0]
    output = Path(task['output_dir']).resolve()
    if run not in output.parents:
        raise ValueError('task output must remain inside run directory')
    plan = commands(run, task, args.gpu, args.mode, args.cpu_offload)
    if args.dry_run:
        print(json.dumps({'CUDA_VISIBLE_DEVICES': str(args.gpu), 'commands': plan}, indent=2))
        return
    # POSIX flock releases automatically on exit; Windows is dry-run only.
    import fcntl
    locks = run / 'locks'
    locks.mkdir(exist_ok=True)
    with (locks / (task['id'] + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('task is already running') from None
        stamp = run / 'completion' / (task['id'] + '_' + args.mode + '.json')
        identity = {'run_sha256': sha(run / 'run.json'), 'task_source_sha256': sha(run / 'llm_manifest.json' if task['family'] == 'llm' else task['task_file']), 'task_id': task['id'], 'mode': args.mode}
        if stamp.exists():
            if read(stamp) != {'status': 'completed', **identity, 'completion_sha256': completion_hashes(run, task, args.mode)}:
                raise ValueError('completion identity mismatch')
            print('completed task skipped')
            return
        if args.mode in ('train', 'all') and output.exists() and any(output.iterdir()):
            raise FileExistsError('task output is nonempty; incomplete recovery requires inspection and a fresh run directory')
        if args.mode == 'eval' and task['family'] == 'llm' and task['regime'] == 'untuned' and output.exists() and any(output.iterdir()):
            raise FileExistsError('untuned output is nonempty; use a fresh run directory')
        eval_outputs = ([output / 'evaluation_methods_P0'] if task['family'] == 'llm' and task['regime'] == 'lora' else
                        [] if task['family'] == 'llm' else
                        [output / ('evaluation_' + split) for split in ('train', 'test')])
        if args.mode in ('eval', 'all') and any(path.exists() and any(path.iterdir()) for path in eval_outputs):
            raise FileExistsError('evaluation output is nonempty; inspect completed/failed state and use a fresh run directory')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        for command in plan:
            if task['family'] == 'bert' and '--evaluate-only' in command:
                bert_eval_tasks(run, task, registry)
            subprocess.run(command, env=env, cwd=run, check=True)
        write(stamp, {'status': 'completed', **identity, 'completion_sha256': completion_hashes(run, task, args.mode)})

if __name__ == '__main__':
    main()
