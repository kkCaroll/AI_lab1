#!/usr/bin/env bash
# 一次任务内依次测量五个模型在固定通用文本上的困惑度。
#
# 五个模型读同一份语料、同一组窗口，故读数可横向比较。每个模型单独一个进程，
# 使显存随进程退出而释放，避免适配器叠加。M1 为完整微调检查点，直接作为
# --model-path；M2 至 M4 以基座加适配器载入。
#
# 用法
#   OUTPUT_ROOT=/mnt/data/exp1/runs/P4-ppl bash scripts/run_ppl.sh
#   # 只测基座，压小 token 数
#   OUTPUT_ROOT=... METHODS=M0 MAX_TOKENS=8192 bash scripts/run_ppl.sh
#
# 任一项失败不中断其余项，退出码取各项的非零值。

set -uo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-/mnt/data/exp1/models/Qwen3-0.6B}
CORPUS=${CORPUS:-/mnt/data/exp1/data/raw/wikitext2_train_2m.txt}
OUTPUT_ROOT=${OUTPUT_ROOT:?需给出本次运行的输出根目录}
METHODS=${METHODS:-M0,M1,M2,M3,M4}
MAX_TOKENS=${MAX_TOKENS:-2000000}
BATCH_SIZE=${BATCH_SIZE:-8}
DTYPE=${DTYPE:-float16}
DEVICE=${DEVICE:-cuda}

M1_PATH=${M1_PATH:-/mnt/data/exp1/runs/P2-6/M1/train/epoch-3}
M2_ADAPTER=${M2_ADAPTER:-/mnt/data/exp1/runs/P2-6c/M2/train/epoch-3}
M3_ADAPTER=${M3_ADAPTER:-/mnt/data/exp1/runs/P2-6c/M3/train/epoch-3}
M4_ADAPTER=${M4_ADAPTER:-/mnt/data/exp1/runs/P3-1/kl1.6/train/epoch-3}

cd "$(dirname "$0")/.." || exit 1

status=0

run_one() {
    local name=$1
    local model=$2
    local adapter=${3:-}
    local args=(--model-path "$model" --corpus "$CORPUS" --output-dir "$OUTPUT_ROOT/$name"
        --max-tokens "$MAX_TOKENS" --batch-size "$BATCH_SIZE" --dtype "$DTYPE" --device "$DEVICE")
    if [ -n "$adapter" ]; then
        args+=(--adapter "$adapter")
    fi
    echo "===== $name ====="
    "$PYTHON" -m scripts.measure_ppl "${args[@]}" || status=1
}

case ",$METHODS," in
    *,M0,*) run_one M0 "$MODEL_PATH" ;;
esac
case ",$METHODS," in
    *,M1,*) run_one M1 "$M1_PATH" ;;
esac
case ",$METHODS," in
    *,M2,*) run_one M2 "$MODEL_PATH" "$M2_ADAPTER" ;;
esac
case ",$METHODS," in
    *,M3,*) run_one M3 "$MODEL_PATH" "$M3_ADAPTER" ;;
esac
case ",$METHODS," in
    *,M4,*) run_one M4 "$MODEL_PATH" "$M4_ADAPTER" ;;
esac

exit $status
