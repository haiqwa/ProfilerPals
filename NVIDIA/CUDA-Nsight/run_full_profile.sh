#!/bin/bash
# 完整程序分析 - 捕获程序开始到结束的所有 CUDA 活动

nsys profile \
    --stats=true \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    --output=profile_full \
    python nsys_full_profile.py

echo ""
echo "Profile complete. Output: profile_full.nsys-rep"
echo "Open with: nsys-ui profile_full.nsys-rep"