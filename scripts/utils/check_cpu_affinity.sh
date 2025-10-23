#!/bin/bash
# 檢查進程的 CPU 使用情況

echo "=========================================="
echo "CPU 核心信息"
echo "=========================================="
echo "物理核心數: $(sysctl -n hw.physicalcpu)"
echo "邏輯核心數: $(sysctl -n hw.logicalcpu)"
echo ""

if [ -z "$1" ]; then
    echo "用法: $0 <PID>"
    echo ""
    echo "範例: $0 3183"
    exit 1
fi

PID=$1

echo "=========================================="
echo "進程 $PID 的 CPU 使用情況"
echo "=========================================="

# 檢查進程是否存在
if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ 進程 $PID 不存在"
    exit 1
fi

# 顯示進程信息
echo ""
echo "主進程:"
ps -p $PID -o pid,%cpu,%mem,command

# 顯示所有相關的子進程
echo ""
echo "子進程:"
PGID=$(ps -p $PID -o pgid=)
ps -g $PGID -o pid,%cpu,%mem,command | grep python | grep -v grep

echo ""
echo "=========================================="
echo "系統 CPU 負載"
echo "=========================================="
top -l 1 | grep "CPU usage"

echo ""
echo "=========================================="
echo "注意事項"
echo "=========================================="
echo "• macOS 不支持像 Linux taskset 那樣的 CPU 親和性設置"
echo "• 系統會自動在所有核心間調度進程"
echo "• Python multiprocessing 會自動利用多核心"
echo "• 當前配置使用 2 個 worker，會占用約 2 個核心"
