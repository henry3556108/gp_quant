#!/bin/bash
# 檢查多樣性分析進度

# 顏色定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 查找正在運行的多樣性分析進程
PIDS=$(ps aux | grep "compute_diversity_metrics.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo -e "${RED}❌ 沒有找到正在運行的多樣性分析進程${NC}"
    echo ""
    echo "提示："
    echo "  1. 檢查進程是否已完成"
    echo "  2. 或使用以下命令啟動分析："
    echo "     conda run -n gp_quant python scripts/analysis/compute_diversity_metrics.py \\"
    echo "         --exp_dir <實驗目錄> --n_workers 2 --no_batch_parallel --cooldown 5"
    exit 1
fi

# 取第一個進程（通常是主進程）
MAIN_PID=$(echo $PIDS | awk '{print $1}')

echo "============================================================"
echo -e "${BLUE}📊 多樣性分析進度報告${NC}"
echo "============================================================"
echo ""

# 獲取進程運行時間
ELAPSED=$(ps -p $MAIN_PID -o etime= | tr -d ' ')
echo -e "${GREEN}🔍 進程信息${NC}"
echo "  PID: $MAIN_PID"
echo "  運行時間: $ELAPSED"

# 轉換運行時間為秒
if [[ $ELAPSED == *:*:* ]]; then
    # 格式: HH:MM:SS
    IFS=':' read -r hours minutes seconds <<< "$ELAPSED"
    # 移除前導零避免八進制問題
    hours=$((10#$hours))
    minutes=$((10#$minutes))
    seconds=$((10#$seconds))
    ELAPSED_SECONDS=$((hours * 3600 + minutes * 60 + seconds))
elif [[ $ELAPSED == *:* ]]; then
    # 格式: MM:SS
    IFS=':' read -r minutes seconds <<< "$ELAPSED"
    # 移除前導零避免八進制問題
    minutes=$((10#$minutes))
    seconds=$((10#$seconds))
    ELAPSED_SECONDS=$((minutes * 60 + seconds))
else
    # 格式: SS
    ELAPSED_SECONDS=$((10#$ELAPSED))
fi

echo "  已運行: ${ELAPSED_SECONDS} 秒 ($((ELAPSED_SECONDS / 60)) 分鐘)"
echo ""

# 配置參數（根據您的設置調整）
COOLDOWN=5
GEN_COMPUTE_TIME=200

# 自動檢測總世代數
EXP_DIR=$(ps -p $MAIN_PID -o command= | sed -n 's/.*--exp_dir \([^ ]*\).*/\1/p')
if [ ! -z "$EXP_DIR" ] && [ -d "${EXP_DIR}/generations" ]; then
    TOTAL_GENS=$(ls ${EXP_DIR}/generations/*.pkl 2>/dev/null | wc -l | tr -d ' ')
    if [ "$TOTAL_GENS" -eq 0 ]; then
        TOTAL_GENS=25  # 預設值
    fi
else
    TOTAL_GENS=25  # 預設值
fi

TIME_PER_GEN=$((GEN_COMPUTE_TIME + COOLDOWN))
TOTAL_ESTIMATED_TIME=$((TIME_PER_GEN * TOTAL_GENS))

# 計算進度
COMPLETED_GENS=$((ELAPSED_SECONDS / TIME_PER_GEN))
PROGRESS_PCT=$((ELAPSED_SECONDS * 100 / TOTAL_ESTIMATED_TIME))

# 基於實際速度計算剩餘時間
if [ $COMPLETED_GENS -gt 0 ]; then
    # 實際每個世代的平均時間
    ACTUAL_TIME_PER_GEN=$((ELAPSED_SECONDS / COMPLETED_GENS))
    REMAINING_GENS=$((TOTAL_GENS - COMPLETED_GENS))
    REMAINING_TIME=$((REMAINING_GENS * ACTUAL_TIME_PER_GEN))
else
    # 如果還沒完成任何世代，使用預估值
    REMAINING_TIME=$((TOTAL_ESTIMATED_TIME - ELAPSED_SECONDS))
fi

REMAINING_MINS=$((REMAINING_TIME / 60))
ACTUAL_TIME_PER_GEN=${ACTUAL_TIME_PER_GEN:-$TIME_PER_GEN}

echo -e "${GREEN}📈 進度統計${NC}"
echo "  預估完成世代: ~${COMPLETED_GENS} / ${TOTAL_GENS}"
echo "  預估進度: ${PROGRESS_PCT}%"

# 繪製進度條
BAR_LENGTH=50
FILLED=$((PROGRESS_PCT * BAR_LENGTH / 100))
EMPTY=$((BAR_LENGTH - FILLED))
BAR=$(printf "█%.0s" $(seq 1 $FILLED))$(printf "░%.0s" $(seq 1 $EMPTY))
echo "  進度條: [${BAR}] ${PROGRESS_PCT}%"
echo ""

# 計算預計完成時間
CURRENT_TIME=$(date +%s)
FINISH_TIME=$((CURRENT_TIME + REMAINING_TIME))
FINISH_TIME_STR=$(date -r $FINISH_TIME "+%H:%M:%S")

echo -e "${GREEN}⏰ 時間預估${NC}"
echo "  當前時間: $(date '+%H:%M:%S')"
if [ $COMPLETED_GENS -gt 0 ]; then
    echo "  實際速度: ~$((ACTUAL_TIME_PER_GEN / 60)) 分 $((ACTUAL_TIME_PER_GEN % 60)) 秒/世代"
    echo "  預估速度: ~$((TIME_PER_GEN / 60)) 分 $((TIME_PER_GEN % 60)) 秒/世代"
fi
echo "  剩餘時間: ~${REMAINING_MINS} 分鐘"
echo "  預計完成: ${FINISH_TIME_STR}"
echo "  總預估時間: ~$((TOTAL_ESTIMATED_TIME / 60)) 分鐘"
echo ""

# 檢查子進程
WORKER_PIDS=$(ps -g $(ps -p $MAIN_PID -o pgid=) -o pid,command | grep "multiprocessing" | grep -v grep | awk '{print $1}')
WORKER_COUNT=$(echo "$WORKER_PIDS" | wc -l | tr -d ' ')

echo -e "${GREEN}💻 系統狀態${NC}"
echo "  活躍 Worker: $WORKER_COUNT"

# 顯示 CPU 使用率
if [ ! -z "$WORKER_PIDS" ]; then
    echo "  Worker CPU 使用率:"
    for pid in $WORKER_PIDS; do
        CPU=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ')
        if [ ! -z "$CPU" ]; then
            echo "    PID $pid: ${CPU}%"
        fi
    done
fi

# 檢查系統總 CPU 使用率
SYSTEM_CPU=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')
echo "  系統 CPU 使用: ${SYSTEM_CPU}%"
echo ""

# 檢查輸出文件
echo -e "${GREEN}📁 輸出狀態${NC}"
EXP_DIR=$(ps -p $MAIN_PID -o command= | sed -n 's/.*--exp_dir \([^ ]*\).*/\1/p')

if [ ! -z "$EXP_DIR" ]; then
    OUTPUT_FILE="${EXP_DIR}/diversity_metrics.json"
    if [ -f "$OUTPUT_FILE" ]; then
        echo -e "  ${GREEN}✅ 輸出文件已生成: $OUTPUT_FILE${NC}"
        FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        echo "     文件大小: $FILE_SIZE"
    else
        echo -e "  ${YELLOW}⏳ 輸出文件尚未生成（完成後才會寫入）${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠️  無法確定實驗目錄${NC}"
fi

echo ""
echo "============================================================"

# 提供建議
if [ $PROGRESS_PCT -lt 30 ]; then
    echo -e "${YELLOW}💡 提示: 分析剛開始，建議您休息一下或處理其他工作${NC}"
elif [ $PROGRESS_PCT -lt 70 ]; then
    echo -e "${BLUE}💡 提示: 分析進行中，預計還需要 ${REMAINING_MINS} 分鐘${NC}"
else
    echo -e "${GREEN}💡 提示: 分析即將完成，請保持電腦運行${NC}"
fi

echo ""
echo "使用方法："
echo "  # 實時監控（每 30 秒更新一次）"
echo "  watch -n 30 bash scripts/utils/check_diversity_progress.sh"
echo ""
echo "  # 或手動執行檢查"
echo "  bash scripts/utils/check_diversity_progress.sh"
echo ""
