import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def matmul_operations():
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    c = torch.randn(2048, 2048, device=device)

    x = torch.mm(a, b)
    y = torch.mm(x, c)
    z = torch.mm(a, c)
    return z

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA 
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True  
) as prof:
    matmul_operations()

prof.export_chrome_trace("trace.json")

print("Profiling done. Open 'chrome://tracing' in Chrome, then load 'trace.json'.")