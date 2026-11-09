"""Placement-only loader: CPU base + PEFT, then explicit single-GPU dispatch."""
import tempfile
from pathlib import Path

def load_model(base, checkpoint, torch, gpu, cpu_offload, max_memory_gib):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    if torch.cuda.device_count() != 1 or gpu != 0:
        raise RuntimeError('v3 loader requires exactly one visible logical CUDA device 0')
    import fcntl  # Acquire the file lock only when loading a CUDA model.
    lock = open(Path(tempfile.gettempdir()) / "train-llm-evaluation-loading.lock", "a")
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        model = AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True)
        if checkpoint is not None:
            model = PeftModel.from_pretrained(model,checkpoint,local_files_only=True,is_trainable=False)
        if cpu_offload:
            from accelerate import infer_auto_device_map, dispatch_model
            placement = infer_auto_device_map(model,max_memory={0:f'{max_memory_gib}GiB','cpu':'128GiB'},
                no_split_module_classes=['LlamaDecoderLayer'])
            if 'cpu' not in placement.values() or not any(str(v) in ('0','cuda:0') for v in placement.values()) or 'disk' in placement.values():
                raise RuntimeError('expected GPU0 plus CPU placement without disk')
            model = dispatch_model(model,device_map=placement,main_device=0,offload_buffers=True)
            model.hf_device_map = placement
        else:
            model.to(torch.device('cuda:0'))
            model.hf_device_map = {'':0}
        return model
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
