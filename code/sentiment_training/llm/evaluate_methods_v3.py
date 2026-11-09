#!/usr/bin/env python3
"""Evaluate the selected LoRA checkpoint with sampling seed 1002."""
import argparse
import json
import random
from pathlib import Path
import train_variants as runner

SAMPLING_SEEDS = (1002,)


def sampling_records(model, tokenizer, rows, torch, device, batch_size, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    records = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            part = rows[offset:offset + batch_size]
            prompts = [tokenizer.encode(runner.prompt_text(tokenizer, row), add_special_tokens=False)[-1024:]
                       for row in part]
            length = max(map(len, prompts))
            masks = torch.tensor([[0] * (length - len(p)) + [1] * len(p) for p in prompts], device=device)
            ids = torch.tensor([[tokenizer.pad_token_id] * (length - len(p)) + p for p in prompts], device=device)
            outputs = model.generate(input_ids=ids, attention_mask=masks, max_new_tokens=runner.MAX_NEW_TOKENS,
                do_sample=True, temperature=1.0, top_p=1.0, top_k=0, num_beams=1,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
            for row, output in zip(part, outputs[:, length:]):
                raw_ids = output.tolist()
                tokens = raw_ids[:raw_ids.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in raw_ids else raw_ids
                raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                records.append({**runner.row_identity(row), "pred_label": runner.TEXT_LABEL.get(raw, -99),
                    "raw_output": raw, "generated_token_ids": raw_ids, "seed": seed})
    return records


def run(args):
    import torch
    from transformers import AutoTokenizer
    manifest_path, task, model_path, train_output, data, paths, hashes = runner.load_job(args)
    if task['kind'] != 'causal':
        raise ValueError('causal models only')
    if args.batch_size < 1 or not torch.cuda.is_available():
        raise ValueError('positive batch size and CUDA required')
    if args.batch_size != 4 or args.max_new_tokens != 8:
        raise ValueError('sampling requires batch4 and max_new_tokens8')
    torch.cuda.set_device(args.gpu)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 required')
    device = torch.device(f'cuda:{args.gpu}')
    selection = None
    done = json.loads((train_output / 'training_done.json').read_text(encoding='utf-8'))
    if done['status'] != 'completed' or done['manifest_sha256'] != runner.sha256_file(manifest_path) or done['data_sha256'] != hashes:
        raise ValueError('training completion provenance mismatch')
    variant = done['identity']['variant']
    runner.TEMPLATE_ID = variant['template_id']
    selection = done['best_checkpoint_step']
    checkpoint = runner.checkpoint_for_step(train_output, selection, done['identity']) / 'model'
    output = train_output / ('evaluation_methods_' + runner.TEMPLATE_ID)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'done.json').exists():
        raise RuntimeError('refusing to overwrite completed evaluation')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    from model_loader_v3 import load_model
    model = load_model(model_path, checkpoint if selection is not None else None, torch, args.gpu, args.cpu_offload, args.max_memory_gib)
    model.requires_grad_(False)
    model.config.use_cache = True
    runner.MAX_NEW_TOKENS = args.max_new_tokens
    identity = {
        'manifest_sha256': runner.sha256_file(manifest_path),
        'data_sha256': hashes,
        'task_id': args.task_id,
        'best_checkpoint_step': selection,
        'template_id': runner.TEMPLATE_ID,
        'script_sha256': runner.sha256_file(Path(__file__)),
        'loader_sha256': runner.sha256_file(Path(__file__).with_name('model_loader_v3.py')),
        'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        'training_runner_sha256': runner.sha256_file(Path(runner.__file__)),
        'source_runner_sha256': runner.SOURCE_RUNNER_SHA256,
        'batch_size': args.batch_size,
        'max_new_tokens': args.max_new_tokens,
        'cpu_offload': args.cpu_offload,
        'max_memory_gib': args.max_memory_gib,
        'device_map': {str(k): str(v) for k, v in getattr(model, 'hf_device_map', {}).items()},
        'generation_config': model.generation_config.to_dict(),
        'tokenizer_eos_token_id': tokenizer.eos_token_id,
        'tokenizer_pad_token_id': tokenizer.pad_token_id,
        'sampling_seeds': list(SAMPLING_SEEDS),
        'sampling': {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0},
    }
    runner.atomic_json(output / 'header.json', {'identity': identity, 'data_paths': paths})
    metrics = {}
    for split in ('train', 'test'):
        rows = data[split]
        methods = {}
        for seed in SAMPLING_SEEDS:
            methods[f'sampling_seed_{seed}'] = sampling_records(model, tokenizer, rows, torch, device, args.batch_size, seed)
        metrics[split] = {}
        for method, records in methods.items():
            runner.atomic_jsonl(output / f'predictions_{split}_{method}.jsonl', records)
            metrics[split][method] = runner.classification_metrics([r['label'] for r in rows], [r['pred_label'] for r in records])
    runner.atomic_json(output / 'metrics.json', {'identity': identity, 'splits': metrics})
    runner.atomic_json(output / 'done.json', {'status': 'completed', 'identity': identity})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4, help="Sampling batch size")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--max-memory-gib", type=int, default=14)
    parser.add_argument("--max-new-tokens", type=int, choices=(8,), default=8)
    parser.add_argument("--template-id", choices=("P0",), default="P0", help="Training prompt template")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
