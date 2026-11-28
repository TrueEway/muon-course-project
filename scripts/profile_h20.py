"""Short controlled GPU smoke test before the full shared-server protocol."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc
import json
import time
import torch
from experiments.vision_model import ResNet18,vision_optimizers
from experiments.resources import configure_cuda,synchronize,cooperate,hardware_metadata
from experiments.common import ROOT,write_json

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
device=configure_cuda()
print(json.dumps(hardware_metadata()),flush=True)
rows=[]
for method in ['sgd','adamw','muon']:
    torch.manual_seed(17)
    model=ResNet18().to(device=device,memory_format=torch.channels_last)
    opts,_=vision_optimizers(model,method,.02)
    x=torch.randn(128,3,32,32,device=device).contiguous(memory_format=torch.channels_last)
    y=torch.arange(128,device=device)%10
    durations=[]
    for step in range(5):
        synchronize()
        tic=time.perf_counter()
        for opt,_ in opts:
            opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(x),y)
        loss.backward()
        for opt,_ in opts:
            opt.step()
        synchronize()
        elapsed=time.perf_counter()-tic
        cooperate(elapsed)
        if step>=2:
            durations.append(elapsed)
    row=dict(method=method,active_step_ms=1000*sum(durations)/len(durations),
             peak_allocated_mib=torch.cuda.max_memory_allocated(0)/(1024**2),
             peak_reserved_mib=torch.cuda.max_memory_reserved(0)/(1024**2),finite_loss=bool(torch.isfinite(loss)))
    rows.append(row)
    print(json.dumps(row),flush=True)
    del model,opts,x,y,loss,opt
    gc.collect()
    torch.cuda.empty_cache()
write_json(ROOT/'output'/'revision'/'h20_smoke_test.json',dict(hardware=hardware_metadata(),results=rows))
