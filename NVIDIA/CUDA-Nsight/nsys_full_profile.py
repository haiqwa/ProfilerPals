"""
NVIDIA Nsight Systems (nsys) Profile 示例 - 完整程序分析
使用 PyTorch 演示 NVTX 标记和 Range Profile

运行方式:
    bash run_full_profile.sh
    或
    nsys profile --stats=true --trace=cuda,nvtx --force-overwrite=true --output=profile_full python nsys_full_profile.py
"""

import torch
import torch.cuda.nvtx as nvtx


def matrix_multiplication():
    """基础矩阵乘法示例"""
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    return c


def conv_operations():
    """卷积操作示例"""
    x = torch.randn(64, 3, 224, 224, device="cuda")
    conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
    out = conv(x)
    torch.cuda.synchronize()
    return out


def mixed_operations():
    """混合操作示例"""
    nvtx.range_push("mixed_operations_init")
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    x = torch.randn(64, 64, 512, 512, device="cuda")
    nvtx.range_pop()

    nvtx.range_push("mixed_operations_compute")
    c = torch.mm(a, b)
    y = torch.nn.functional.relu(x)
    z = torch.nn.functional.max_pool2d(y, 2)
    torch.cuda.synchronize()
    nvtx.range_pop()

    return c, z


def main():
    torch.cuda.init()

    nvtx.range_push("warmup")
    for _ in range(3):
        _ = torch.mm(torch.randn(1024, 1024, device="cuda"),
                     torch.randn(1024, 1024, device="cuda"))
    torch.cuda.synchronize()
    nvtx.range_pop()

    nvtx.range_push("profile_section_1_matrix_mul")
    nvtx.mark("start_matrix_multiplication")
    for i in range(5):
        nvtx.range_push(f"matrix_mul_iter_{i}")
        matrix_multiplication()
        nvtx.range_pop()
    nvtx.mark("end_matrix_multiplication")
    nvtx.range_pop()

    nvtx.range_push("profile_section_2_conv")
    nvtx.mark("start_conv_operations")
    for i in range(3):
        nvtx.range_push(f"conv_iter_{i}")
        conv_operations()
        nvtx.range_pop()
    nvtx.mark("end_conv_operations")
    nvtx.range_pop()

    nvtx.range_push("profile_section_3_mixed")
    mixed_operations()
    nvtx.range_pop()

    print("Profiling complete. Check profile_full.nsys-rep")


if __name__ == "__main__":
    main()