"""Process-local single-GPU limits; no global GPU or other-process changes."""
import os
import time
import subprocess
import torch

GPU_DUTY = 0.35
GPU_FRACTION = 0.04
MIN_FREE_GIB = 12


def configure_cuda():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA device through CUDA_VISIBLE_DEVICES')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION,0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    check_headroom()
    return torch.device('cuda:0')


def check_headroom():
    free,_ = torch.cuda.mem_get_info(0)
    if free < MIN_FREE_GIB*(1024**3):
        raise RuntimeError('Shared GPU free memory fell below 12 GiB; stopping this run to release its allocations')


def synchronize():
    torch.cuda.synchronize(0)


def cooperate(active_seconds):
    delay = max(0.0,active_seconds*(1/GPU_DUTY-1))
    time.sleep(delay)
    return delay


def hardware_metadata():
    physical = os.environ.get('CUDA_VISIBLE_DEVICES','')
    rows = subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=name,memory.total,driver_version',
                                    '--format=csv,noheader,nounits'],text=True).strip().split(',')
    cpu = 'unknown'
    with open('/proc/cpuinfo') as stream:
        for line in stream:
            if line.startswith('model name'):
                cpu = line.split(':',1)[1].strip()
                break
    return dict(device='cuda:0',gpu_name=rows[0].strip(),gpu_total_memory_mib=float(rows[1]),
                driver_version=rows[2].strip(),cuda_version=torch.version.cuda,cpu_model=cpu,
                cpu_affinity_count=len(os.sched_getaffinity(0)),gpu_duty_target=GPU_DUTY,
                gpu_allocator_fraction=GPU_FRACTION,tf32=False,sharing_policy='single GPU, sequential workers, cooperative pacing')
