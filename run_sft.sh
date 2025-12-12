#!/bin/bash

# 基于 DeepSpeed + HuggingFace TRL 的32B模型SFT训练脚本
# 使用 TRL SFTTrainer + DeepSpeed ZeRO 进行分布式训练
# 使用方法: bash run_sft.sh 或 ./run_sft.sh

# 设置脚本在遇到错误时退出
set -e

# 设置内存和资源限制（避免 OOM）
# 增加虚拟内存限制（如果系统支持）
ulimit -v unlimited 2>/dev/null || true
# 增加文件描述符限制
ulimit -n 65536 2>/dev/null || true
# 增加进程数限制
ulimit -u 32768 2>/dev/null || true

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建日志目录
LOGS_DIR=".logs"
mkdir -p "${LOGS_DIR}"

# 生成日志文件名（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/sft_train_${TIMESTAMP}.log"

# ============================================
# 训练参数配置（32B模型专用）
# ============================================

# 模型配置
MODEL_NAME="/data/Qwen3-32B"
OUTPUT_DIR="/data/outputs-sft-32b"

# 数据集配置
DATASET=""  # 留空使用默认数据集
SUBSET_NAME=""  # 留空使用所有默认子集
TEXT_COLUMN="text"

# 训练超参数（32B模型优化配置）
LEARNING_RATE=1e-5  # 32B模型建议使用较小的学习率
BATCH_SIZE=1  # 每个GPU的批次大小（32B模型需要很小的batch size）
GRADIENT_ACCUMULATION_STEPS=1  # 增加梯度累积以补偿小batch size（32B模型需要大梯度累积）
NUM_EPOCHS=3
MAX_SEQ_LENGTH=1024  # 减小序列长度以节省显存（从2048减小到1024）
WARMUP_STEPS=500
SEED=42

# 保存和日志配置
SAVE_STEPS=500  # 32B模型保存检查点较慢，适当增加间隔
LOGGING_STEPS=10
EVAL_STRATEGY="no"
EVAL_STEPS=500
SAVE_TOTAL_LIMIT=2  # 只保留2个检查点以节省磁盘空间

# 性能优化选项（32B模型强烈推荐启用）
USE_BF16="--bf16"  # 使用bfloat16精度（推荐用于A100/H100等GPU）
# USE_FP16="--fp16"  # 如果GPU不支持BF16，使用FP16
GRADIENT_CHECKPOINTING="--gradient_checkpointing"  # 必须启用以节省显存
USE_8BIT_OPTIMIZER=""  # 8-bit优化器选项

# ============================================
# DeepSpeed专用OOM优化选项（不使用LoRA）
# ============================================
# 激进内存优化模式（最激进的显存节省，可节省额外20-30%显存）
DEEPSPEED_AGGRESSIVE_MEMORY=""  # 启用激进内存优化，取消注释以启用
# DEEPSPEED_AGGRESSIVE_MEMORY="--deepspeed_aggressive_memory"

# CPU Offload选项（默认启用，可节省大量显存）
DEEPSPEED_OFFLOAD_OPTIMIZER="--deepspeed_offload_optimizer"  # 优化器CPU Offload（默认启用）
DEEPSPEED_OFFLOAD_PARAM="--deepspeed_offload_param"  # 参数CPU Offload（Stage 3专用，默认启用）

# 自定义DeepSpeed参数（可选，留空使用默认值）
# DEEPSPEED_REDUCE_BUCKET_SIZE="--deepspeed_reduce_bucket_size 5000000"  # 5MB（激进模式）
# DEEPSPEED_STAGE3_MAX_LIVE_PARAMS="--deepspeed_stage3_max_live_params 50000000"  # 5千万（激进模式）

# 其他选项
DATALOADER_NUM_WORKERS=0  # 减少数据加载器工作进程数以节省内存（AMD GPU建议设为0）
DATALOADER_PIN_MEMORY=""  # AMD GPU建议禁用pin_memory以节省内存

# ============================================
# 分布式训练配置
# ============================================
# 对于32B模型，使用 DeepSpeed ZeRO Stage 3 + CPU Offload

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'  # 使用8个GPU
NPROC_PER_NODE=8  # 总GPU数量

# AMD GPU HIP 内存优化环境变量
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128  # 限制内存块大小，减少碎片
export HIP_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  # 设置可见的HIP设备

# HSA运行时环境变量（解决HSA_STATUS_ERROR_OUT_OF_RESOURCES错误）
export HSA_MAX_QUEUE_SIZE=4096  # 限制HSA队列大小，避免资源耗尽
export HSA_QUEUE_PRIORITY=normal  # 设置队列优先级
export HSA_AMD_SDMA_COMPUTE=0  # 禁用SDMA计算队列，避免资源竞争
export HIP_FORCE_DEV_KERNARG=0  # 禁用强制设备kernel参数
# 如果遇到GFX版本问题，可以取消下面这行的注释并设置正确的版本
# export HSA_OVERRIDE_GFX_VERSION=10.3.0

# PyTorch 分布式训练优化
export TORCH_NCCL_BLOCKING_WAIT=0  # 非阻塞等待，避免死锁（关键！）
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 详细的分布式调试信息（可选，生产环境可关闭）

# 使用 DeepSpeed（推荐）
USE_DEEPSPEED="--use_deepspeed"  # 使用DeepSpeed（支持ZeRO优化）

# DeepSpeed配置（如果使用DeepSpeed）
# 可以指定配置文件路径，或使用代码中的默认配置（ZeRO Stage 3 + CPU Offload）
DEEPSPEED_CONFIG=""  # 留空使用默认配置，或指定配置文件路径
# DEEPSPEED_CONFIG="--deepspeed_config /path/to/deepspeed_config.json"

# ============================================
# 构建训练命令
# ============================================

# 使用 torchrun 启动分布式训练
# torchrun 会自动设置分布式训练所需的环境变量（RANK, WORLD_SIZE, LOCAL_RANK等）
# --standalone: 单节点训练模式
# --nproc_per_node: 每个节点的进程数（GPU数量）
# 如果是多节点训练，需要添加: --nnodes, --node_rank, --master_addr, --master_port

CMD="/opt/conda/envs/python3.10.13/bin/torchrun \
    --standalone \
    --nproc_per_node=${NPROC_PER_NODE} \
    train_sft.py \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate ${LEARNING_RATE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --warmup_steps ${WARMUP_STEPS} \
    --seed ${SEED} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --eval_strategy ${EVAL_STRATEGY} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT}"

# 添加分布式训练方案
if [ -n "${USE_DEEPSPEED}" ]; then
    CMD="${CMD} ${USE_DEEPSPEED}"
    if [ -n "${DEEPSPEED_CONFIG}" ]; then
        CMD="${CMD} ${DEEPSPEED_CONFIG}"
    fi
fi

# 添加数据集参数
if [ -n "${DATASET}" ]; then
    CMD="${CMD} --dataset ${DATASET}"
fi

if [ -n "${SUBSET_NAME}" ]; then
    CMD="${CMD} --subset_name"
    for subset in ${SUBSET_NAME}; do
        CMD="${CMD} ${subset}"
    done
fi

if [ -n "${TEXT_COLUMN}" ]; then
    CMD="${CMD} --text_column ${TEXT_COLUMN}"
fi

# 添加精度选项
if [ -n "${USE_BF16}" ]; then
    CMD="${CMD} ${USE_BF16}"
fi

if [ -n "${USE_FP16}" ]; then
    CMD="${CMD} ${USE_FP16}"
fi

# 添加优化选项
if [ -n "${GRADIENT_CHECKPOINTING}" ]; then
    CMD="${CMD} ${GRADIENT_CHECKPOINTING}"
fi

if [ -n "${USE_8BIT_OPTIMIZER}" ]; then
    CMD="${CMD} ${USE_8BIT_OPTIMIZER}"
fi

# 添加DeepSpeed专用优化选项
if [ -n "${DEEPSPEED_AGGRESSIVE_MEMORY}" ]; then
    CMD="${CMD} ${DEEPSPEED_AGGRESSIVE_MEMORY}"
fi

if [ -n "${DEEPSPEED_OFFLOAD_OPTIMIZER}" ]; then
    CMD="${CMD} ${DEEPSPEED_OFFLOAD_OPTIMIZER}"
fi

if [ -n "${DEEPSPEED_OFFLOAD_PARAM}" ]; then
    CMD="${CMD} ${DEEPSPEED_OFFLOAD_PARAM}"
fi

if [ -n "${DEEPSPEED_REDUCE_BUCKET_SIZE}" ]; then
    CMD="${CMD} ${DEEPSPEED_REDUCE_BUCKET_SIZE}"
fi

if [ -n "${DEEPSPEED_STAGE3_MAX_LIVE_PARAMS}" ]; then
    CMD="${CMD} ${DEEPSPEED_STAGE3_MAX_LIVE_PARAMS}"
fi

if [ "${EVAL_STRATEGY}" != "no" ]; then
    CMD="${CMD} --eval_steps ${EVAL_STEPS}"
fi

# ============================================
# 打印配置信息
# ============================================
{
    echo "=========================================="
    echo "基于 DeepSpeed + TRL 的32B模型SFT训练配置"
    echo "=========================================="
    echo "模型路径: ${MODEL_NAME}"
    echo "输出目录: ${OUTPUT_DIR}"
    echo "数据集: ${DATASET:-默认数据集 (AI-ModelScope/COIG-CQIA)}"
    if [ -n "${SUBSET_NAME}" ]; then
        echo "子集名称: ${SUBSET_NAME}"
    else
        echo "子集名称: 所有默认子集"
    fi
    echo ""
    echo "--- 训练超参数 ---"
    echo "学习率: ${LEARNING_RATE}"
    echo "批次大小: ${BATCH_SIZE} (每GPU)"
    echo "梯度累积步数: ${GRADIENT_ACCUMULATION_STEPS}"
    echo "有效批次大小: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE))"
    echo "训练轮数: ${NUM_EPOCHS}"
    echo "最大序列长度: ${MAX_SEQ_LENGTH}"
    echo "预热步数: ${WARMUP_STEPS}"
    echo ""
    echo "--- 分布式训练配置 ---"
    echo "总GPU数量: ${NPROC_PER_NODE}"
    if [ -n "${USE_DEEPSPEED}" ]; then
        echo "训练方案: DeepSpeed + TRL SFTTrainer (推荐)"
        if [ -n "${DEEPSPEED_CONFIG}" ]; then
            echo "DeepSpeed配置: ${DEEPSPEED_CONFIG}"
        else
            echo "DeepSpeed配置: 默认 (ZeRO Stage 3 + CPU Offload)"
        fi
    else
        echo "训练方案: DDP + TRL SFTTrainer"
    fi
    echo ""
    echo "--- 优化选项 ---"
    if [ -n "${USE_BF16}" ]; then
        echo "精度: BF16 (bfloat16)"
    elif [ -n "${USE_FP16}" ]; then
        echo "精度: FP16 (float16)"
    else
        echo "精度: FP32 (float32)"
    fi
    if [ -n "${GRADIENT_CHECKPOINTING}" ]; then
        echo "梯度检查点: 已启用（可节省约40-50%显存）"
    fi
    if [ -n "${USE_8BIT_OPTIMIZER}" ]; then
        echo "8-bit优化器: 已启用（可节省约50-75%优化器显存）"
    fi
    echo ""
    echo "--- DeepSpeed专用OOM优化选项 ---"
    if [ -n "${USE_DEEPSPEED}" ]; then
        if [ -n "${DEEPSPEED_AGGRESSIVE_MEMORY}" ]; then
            echo "⚠️  激进内存优化: 已启用（可额外节省20-30%显存）"
        else
            echo "激进内存优化: 未启用（标准模式）"
        fi
        if [ -n "${DEEPSPEED_OFFLOAD_OPTIMIZER}" ]; then
            echo "✓ 优化器CPU Offload: 已启用"
        fi
        if [ -n "${DEEPSPEED_OFFLOAD_PARAM}" ]; then
            echo "✓ 参数CPU Offload: 已启用（Stage 3专用）"
        fi
        if [ -n "${DEEPSPEED_REDUCE_BUCKET_SIZE}" ]; then
            echo "  自定义reduce_bucket_size: ${DEEPSPEED_REDUCE_BUCKET_SIZE}"
        fi
        if [ -n "${DEEPSPEED_STAGE3_MAX_LIVE_PARAMS}" ]; then
            echo "  自定义stage3_max_live_parameters: ${DEEPSPEED_STAGE3_MAX_LIVE_PARAMS}"
        fi
    else
        echo "DeepSpeed优化选项: 未启用（未使用DeepSpeed）"
    fi
    echo ""
    echo "--- 其他配置 ---"
    echo "保存检查点限制: ${SAVE_TOTAL_LIMIT}"
    echo "随机种子: ${SEED}"
    echo "日志文件: ${LOG_FILE}"
    echo "=========================================="
    echo ""
} | tee -a "${LOG_FILE}"

# ============================================
# 验证配置
# ============================================

# 检查Python是否可用
if ! command -v python &> /dev/null; then
    echo "错误: 未找到python命令"
    exit 1
fi

# 检查训练脚本是否存在
if [ ! -f "train_sft.py" ]; then
    echo "错误: 未找到train_sft.py文件"
    exit 1
fi

# 验证分布式训练配置（已移除FSDP选项，无需验证）

# ============================================
# 执行训练
# ============================================

echo "开始执行训练..."
echo "命令: ${CMD}"
echo "日志文件: ${LOG_FILE}"
echo ""

# 使用 nohup 在后台运行，并将输出追加到日志文件
# 添加错误处理，捕获 exitcode -6 等错误
nohup bash -c "
    set -e
    # 设置错误处理
    trap 'echo \"训练进程异常退出，退出码: \$?\" >> \"${LOG_FILE}\" 2>&1; exit 1' ERR
    # 执行训练命令
    ${CMD}
" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

# 等待一下确保进程启动
sleep 3

# 检查进程是否还在运行
if ps -p ${TRAIN_PID} > /dev/null 2>&1; then
    echo "训练已在后台启动！"
    echo "进程ID: ${TRAIN_PID}"
    echo "日志文件: ${LOG_FILE}"
    echo ""
    echo "查看日志: tail -f ${LOG_FILE}"
    echo "查看进程: ps -p ${TRAIN_PID}"
    echo "停止训练: kill ${TRAIN_PID}"
    echo ""
else
    echo "错误: 训练进程启动失败，请查看日志文件: ${LOG_FILE}"
    echo ""
    echo "常见问题排查:"
    echo "1. 检查 GPU 是否可用: nvidia-smi"
    echo "2. 检查内存是否充足: free -h"
    echo "3. 检查 CUDA 版本: nvcc --version"
    echo "4. 查看日志文件末尾: tail -100 ${LOG_FILE}"
    exit 1
fi
