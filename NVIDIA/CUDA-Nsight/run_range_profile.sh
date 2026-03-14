#!/bin/bash
# 精确控制分析范围 - 使用 profiler.start()/stop() 控制捕获范围
# 注意: warmup 阶段不会被捕获

nsys profile \
    --stats=true \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --force-overwrite=true \
    --output=profile_range \
    python nsys_range_profile.py

echo ""
echo "Profile complete. Output: profile_range.nsys-rep"
echo "Open with: nsys-ui profile_range.nsys-rep"