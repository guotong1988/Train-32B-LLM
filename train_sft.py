#!/usr/bin/env python3
"""
基于 DeepSpeed + HuggingFace TRL 的SFT训练脚本，用于训练32B大模型
使用 TRL SFTTrainer + DeepSpeed ZeRO 进行分布式训练

使用方法：
1. DeepSpeed训练（推荐）:
   torchrun --nproc_per_node=8 train_sft_megatron.py --use_deepspeed ...

2. 标准DDP训练:
   torchrun --nproc_per_node=8 train_sft_megatron.py ...
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from modelscope.msdatasets import MsDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

# 尝试导入DeepSpeed（推荐用于大模型训练）
DEEPSPEED_AVAILABLE = False
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
    print("✓ DeepSpeed 可用")
except ImportError:
    print("提示: DeepSpeed 未安装，将使用标准分布式训练")
    print("  安装命令: pip install deepspeed")


def format_dataset_for_sft(ds, text_column: str = "text", max_length: int = 2048, tokenizer=None):
    """将数据集格式化为SFT训练所需的格式（使用Qwen3 chat template）

    Args:
        ds: 数据集对象
        text_column: 文本列名，如果不存在则尝试从对话格式构建
        max_length: 最大序列长度
        tokenizer: 分词器，用于应用chat template（Qwen3系列）
    """
    # 打印原始数据集信息
    print(f"原始数据集大小: {len(ds)}")
    print(f"原始数据集列名: {ds.column_names}")
    if len(ds) > 0:
        print(f"第一个样本的键: {list(ds[0].keys())}")
        print(f"第一个样本示例（前200字符）: {str(ds[0])[:200]}")

    # 检查是否可以使用chat template
    use_chat_template = tokenizer is not None and hasattr(tokenizer, "apply_chat_template")
    if use_chat_template:
        print("使用Qwen3 chat template格式化数据")
    else:
        print("警告: 未提供tokenizer或tokenizer不支持chat template，将使用简单格式")

    def format_conversation(example):
        """格式化对话数据为文本格式（使用Qwen3 chat template）"""
        # 如果已经有text字段，检查是否已经是chat template格式
        if text_column in example and example[text_column]:
            text_val = example[text_column]
            if text_val and str(text_val).strip():
                # 如果已经包含chat template标记，直接返回
                if "<|im_start|>" in str(text_val) or "<|im_end|>" in str(text_val):
                    return {"text": str(text_val)}
                # 否则尝试用chat template重新格式化（如果有conversations信息）
                # 这里先返回原文本，后续可以改进

        # 尝试从对话格式构建（优先使用chat template）
        if "conversations" in example or "messages" in example:
            conversations = example.get("conversations") or example.get("messages", [])
            if isinstance(conversations, str):
                # 如果是字符串，尝试解析JSON
                import json
                try:
                    conversations = json.loads(conversations)
                except:
                    pass
            
            if isinstance(conversations, list) and len(conversations) > 0:
                # 构建messages列表用于chat template
                messages = []
                for msg in conversations:
                    if isinstance(msg, dict):
                        role = msg.get("role", msg.get("from", ""))
                        content = msg.get("content", msg.get("value", ""))
                        if role and content:
                            # 标准化角色名称
                            if role in ["user", "human"]:
                                messages.append({"role": "user", "content": str(content)})
                            elif role in ["assistant", "gpt", "bot"]:
                                messages.append({"role": "assistant", "content": str(content)})
                            else:
                                messages.append({"role": role, "content": str(content)})
                
                # 使用chat template格式化
                if messages and use_chat_template:
                    try:
                        formatted_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False  # 训练时不需要generation prompt
                        )
                        return {"text": formatted_text}
                    except Exception as e:
                        print(f"警告: 使用chat template失败: {e}，回退到简单格式")
                        # 回退到简单格式
                        text_parts = []
                        for msg in messages:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "user":
                                text_parts.append(f"用户: {content}")
                            elif role == "assistant":
                                text_parts.append(f"助手: {content}")
                            else:
                                text_parts.append(f"{role}: {content}")
                        if text_parts:
                            return {"text": "\n".join(text_parts)}
                elif messages:
                    # 没有chat template，使用简单格式
                    text_parts = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user":
                            text_parts.append(f"用户: {content}")
                        elif role == "assistant":
                            text_parts.append(f"助手: {content}")
                        else:
                            text_parts.append(f"{role}: {content}")
                    if text_parts:
                        return {"text": "\n".join(text_parts)}

        # 尝试从prompt和response构建
        if "prompt" in example and "response" in example:
            prompt = example.get("prompt", "")
            response = example.get("response", "")
            if prompt and response:
                if use_chat_template:
                    try:
                        messages = [
                            {"role": "user", "content": str(prompt)},
                            {"role": "assistant", "content": str(response)}
                        ]
                        formatted_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        return {"text": formatted_text}
                    except Exception as e:
                        print(f"警告: 使用chat template失败: {e}，回退到简单格式")
                return {"text": f"用户: {prompt}\n助手: {response}"}

        # 尝试从input和output构建
        if "input" in example and "output" in example:
            input_text = example.get("input", "")
            output_text = example.get("output", "")
            if input_text and output_text:
                if use_chat_template:
                    try:
                        messages = [
                            {"role": "user", "content": str(input_text)},
                            {"role": "assistant", "content": str(output_text)}
                        ]
                        formatted_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        return {"text": formatted_text}
                    except Exception as e:
                        print(f"警告: 使用chat template失败: {e}，回退到简单格式")
                return {"text": f"用户: {input_text}\n助手: {output_text}"}

        # 尝试从instruction和output构建
        if "instruction" in example and "output" in example:
            instruction = example.get("instruction", "")
            output = example.get("output", "")
            if instruction and output:
                if use_chat_template:
                    try:
                        messages = [
                            {"role": "user", "content": str(instruction)},
                            {"role": "assistant", "content": str(output)}
                        ]
                        formatted_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        return {"text": formatted_text}
                    except Exception as e:
                        print(f"警告: 使用chat template失败: {e}，回退到简单格式")
                return {"text": f"用户: {instruction}\n助手: {output}"}

        # 如果都没有，尝试将所有非空字段拼接
        text_parts = []
        for key, value in example.items():
            if value and isinstance(value, (str, int, float)):
                text_parts.append(f"{key}: {value}")
        if text_parts:
            return {"text": "\n".join(text_parts)}

        # 如果都没有，返回空字符串
        return {"text": ""}

    # 应用格式化函数
    formatted_ds = ds.map(format_conversation, remove_columns=ds.column_names)
    print(f"格式化后数据集大小（过滤前）: {len(formatted_ds)}")

    # 检查格式化后的数据
    if len(formatted_ds) > 0:
        non_empty_count = sum(1 for i in range(min(10, len(formatted_ds))) if formatted_ds[i]["text"].strip())
        print(f"前10个样本中非空文本数量: {non_empty_count}")
        if non_empty_count > 0:
            print(f"第一个非空样本示例: {formatted_ds[0]['text'][:200]}")

    # 过滤空文本
    formatted_ds = formatted_ds.filter(lambda x: len(x["text"].strip()) > 0)
    print(f"过滤后数据集大小: {len(formatted_ds)}")

    return formatted_ds


def load_dataset_for_sft(
        dataset_name_or_path: Optional[str] = None,
        subset_name: Optional[Union[str, List[str]]] = None,
        split: str = "train",
        text_column: str = "text",
        max_length: int = 2048,
        tokenizer=None,
) -> Dataset:
    """加载并格式化数据集用于SFT训练"""
    if dataset_name_or_path:
        # 如果是本地路径
        if os.path.exists(dataset_name_or_path):
            from datasets import load_dataset
            ds = load_dataset("json", data_files=dataset_name_or_path, split=split)
        else:
            # 尝试从ModelScope加载
            try:
                if subset_name:
                    if isinstance(subset_name, list):
                        # 加载多个子集并合并
                        datasets = []
                        for subset in subset_name:
                            print(f"加载子集: {subset}")
                            subset_ds = MsDataset.load(dataset_name_or_path, subset_name=subset, split=split)
                            if hasattr(subset_ds, 'to_hf_dataset'):
                                subset_ds = subset_ds.to_hf_dataset()
                            datasets.append(subset_ds)
                        print(f"合并 {len(datasets)} 个子集...")
                        ds = concatenate_datasets(datasets)
                    else:
                        ds = MsDataset.load(dataset_name_or_path, subset_name=subset_name, split=split)
                else:
                    ds = MsDataset.load(dataset_name_or_path, split=split)
                # 转换为HuggingFace Dataset格式（如果还不是）
                if hasattr(ds, 'to_hf_dataset'):
                    ds = ds.to_hf_dataset()
                # 如果已经是HuggingFace Dataset，直接使用
            except Exception as e:
                print(f"从ModelScope加载失败，尝试从HuggingFace加载: {e}")
                from datasets import load_dataset
                ds = load_dataset(dataset_name_or_path, split=split)
    else:
        # 使用默认数据集，加载所有可用的子集
        default_subsets = [
            'chinese_traditional',
            'coig_pc',
            'exam',
            'finance',
            'douban',
            'human_value',
            'logi_qa',
            'ruozhiba',
            'segmentfault',
            'wiki',
            'wikihow',
            'xhs',
            'zhihu'
        ]

        # 如果指定了subset_name，使用指定的；否则使用所有默认子集
        if subset_name:
            if isinstance(subset_name, str):
                subset_list = [subset_name]
            else:
                subset_list = subset_name
        else:
            subset_list = default_subsets

        print(f"使用默认数据集: AI-ModelScope/COIG-CQIA")
        print(f"加载子集: {subset_list}")

        datasets = []
        for subset in subset_list:
            try:
                print(f"正在加载子集: {subset}...")
                subset_ds = MsDataset.load('AI-ModelScope/COIG-CQIA', subset_name=subset, split='train')
                # 转换为HuggingFace Dataset格式（如果还不是）
                if hasattr(subset_ds, 'to_hf_dataset'):
                    subset_ds = subset_ds.to_hf_dataset()
                print(f"子集 {subset} 加载完成，大小: {len(subset_ds)}")
                datasets.append(subset_ds)
            except Exception as e:
                print(f"警告: 加载子集 {subset} 失败: {e}")
                continue

        if not datasets:
            raise ValueError("所有子集加载失败，请检查数据集名称和网络连接")

        print(f"合并 {len(datasets)} 个子集...")
        ds = concatenate_datasets(datasets)
        print(f"合并后数据集总大小: {len(ds)}")

    # 检查数据集是否为空
    if len(ds) == 0:
        print("警告: 加载的数据集为空！")
        print("请检查数据集路径或名称是否正确。")
        return ds

    # 格式化数据集（使用Qwen3 chat template）
    formatted_ds = format_dataset_for_sft(ds, text_column=text_column, max_length=max_length, tokenizer=tokenizer)

    # 清理原始数据集以释放内存
    del ds
    import gc
    gc.collect()

    if len(formatted_ds) == 0:
        print("警告: 格式化后的数据集为空！")
        print("可能的原因:")
        print("1. 数据集格式不符合预期")
        print("2. 所有数据都被过滤掉了")
        print("3. 数据格式解析失败")

    return formatted_ds


@dataclass
class DataCollatorForCausalLM:
    """用于通用文本全量微调的 data collator。

    功能：
    - 使用 tokenizer 对样本中的 `text` 字段做截断和 padding
    - 构造 labels，并将 padding 位置的标签置为 -100（忽略这些位置的 loss）
    - 对非 padding 的 token 全量计算 loss，适用于通用语言建模目标
    """

    tokenizer: AutoTokenizer
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 从样本中取出文本字段
        texts = [f["text"] for f in features]

        # 统一做 tokenize / 截断 / padding
        batch = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        # labels 初始与 input_ids 一致
        labels = batch["input_ids"].clone()

        # 将 padding 位置的 label 置为 -100，避免对 padding 计算 loss
        if "attention_mask" in batch:
            pad_mask = batch["attention_mask"] == 0
            labels[pad_mask] = -100

        batch["labels"] = labels
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="基于DeepSpeed + TRL的SFT训练 (32B模型)")

    # 模型参数
    parser.add_argument("--model_name", type=str, default="/data/Qwen3-32B",
                        help="模型路径或名称")
    parser.add_argument("--output_dir", type=str, default="./outputs-sft-megatron",
                        help="输出目录")

    # 数据集参数
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集名称或路径")
    parser.add_argument("--subset_name", default=None, nargs='+',
                        help="数据集子集名称")
    parser.add_argument("--text_column", type=str, default="text",
                        help="文本列名")

    # 训练超参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率（32B模型建议使用较小的学习率）")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="梯度累积步数（32B模型需要较大的梯度累积）")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="预热步数")

    # 保存和日志
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="保存检查点的步数")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="保存的检查点数量限制")
    parser.add_argument("--eval_strategy", type=str, default="no",
                        choices=["no", "steps", "epoch"],
                        help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估步数（当eval_strategy=steps时）")

    # 精度和优化
    parser.add_argument("--bf16", action="store_true",
                        help="使用bfloat16精度（推荐用于A100/H100）")
    parser.add_argument("--fp16", action="store_true",
                        help="使用float16精度")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="启用梯度检查点以节省内存")
    parser.add_argument("--use_8bit_optimizer", action="store_true",
                        help="使用8-bit优化器")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="优化器类型，如果使用8-bit优化器，应设置为adamw_8bit或paged_adamw_8bit")
    parser.add_argument("--packing", action="store_true", default=True,
                        help="启用序列打包以提高效率（默认启用，可减少padding浪费）")
    
    # DeepSpeed专用优化选项
    parser.add_argument("--deepspeed_aggressive_memory", action="store_true",
                        help="启用DeepSpeed激进内存优化（更小的bucket size等）")
    parser.add_argument("--deepspeed_cpu_offload", action="store_true", default=True,
                        help="启用DeepSpeed CPU Offload（优化器和参数都offload到CPU）")
    parser.add_argument("--deepspeed_offload_optimizer", action="store_true", default=True,
                        help="启用优化器CPU Offload")
    parser.add_argument("--deepspeed_offload_param", action="store_true", default=True,
                        help="启用参数CPU Offload（Stage 3专用）")
    parser.add_argument("--deepspeed_reduce_bucket_size", type=int, default=None,
                        help="自定义reduce bucket size（字节，默认自动）")
    parser.add_argument("--deepspeed_stage3_max_live_params", type=int, default=None,
                        help="自定义Stage 3最大活跃参数数（默认自动）")

    # 分布式训练参数
    parser.add_argument("--use_deepspeed", action="store_true",
                        help="使用DeepSpeed进行分布式训练（推荐）")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="DeepSpeed配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="本地GPU rank（由torchrun自动设置）")

    # 其他参数
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="数据加载器工作进程数（AMD GPU建议设为0以节省内存）")
    parser.add_argument("--dataloader_pin_memory", action="store_true",
                        help="启用数据加载器pin memory")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="最大训练步数，-1表示使用num_train_epochs")

    args = parser.parse_args()

    # 在脚本开始时就设置 NCCL 环境变量（必须在任何分布式操作之前）
    os.environ.setdefault("NCCL_TIMEOUT", "1800")  # 30分钟超时
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")  # 启用异步错误处理
    os.environ.setdefault("NCCL_DEBUG", "INFO")  # 调试信息
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
    
    # AMD GPU (ROCm/HSA) 环境变量配置
    # 检测是否为AMD GPU环境（ROCm使用HIP而不是CUDA）
    is_rocm = os.environ.get("ROCM_PATH") or os.path.exists("/opt/rocm")
    if is_rocm or not torch.version.cuda:  # 如果没有CUDA版本，可能是ROCm
        # HSA运行时配置
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # 如果遇到GFX版本问题，可以取消注释并设置
        os.environ.setdefault("HSA_QUEUE_PRIORITY", "normal")
        
        # HIP内存管理优化
        if "PYTORCH_HIP_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # 限制HSA队列大小，避免资源耗尽
        os.environ.setdefault("HSA_MAX_QUEUE_SIZE", "4096")
        
        # 禁用一些可能导致资源问题的特性
        os.environ.setdefault("HSA_AMD_SDMA_COMPUTE", "0")  # 禁用SDMA计算队列
        os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "0")  # 禁用强制设备kernel参数
        
        print("检测到AMD GPU/ROCm环境，已配置HSA运行时环境变量")
    
    set_seed(args.seed)

    # 初始化分布式训练环境
    # torchrun 会自动初始化进程组，这里只需要检查并使用
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        args.local_rank = int(os.environ["RANK"])
    
    # 检查分布式训练环境
    if args.local_rank != -1 and torch.cuda.is_available():
        # 对于AMD GPU，延迟设置设备，避免HSA资源竞争
        # 先初始化进程组，然后同步，最后再设置设备
        if not dist.is_initialized():
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "29500")
            rank = int(os.environ.get("RANK", args.local_rank))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            # 使用gloo后端先初始化，然后再切换到nccl（如果需要）
            # 对于ROCm，使用gloo可能更稳定
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method=f"tcp://{master_addr}:{master_port}",
                    rank=rank,
                    world_size=world_size,
                    timeout=torch.distributed.default_pg_timeout,
                )
            except RuntimeError as e:
                # 如果nccl初始化失败，尝试gloo（ROCm的备选方案）
                if "nccl" in str(e).lower() or "HSA" in str(e).upper():
                    print(f"警告: NCCL初始化失败，尝试使用Gloo后端: {e}")
                    try:
                        dist.init_process_group(
                            backend="gloo",
                            init_method=f"tcp://{master_addr}:{master_port}",
                            rank=rank,
                            world_size=world_size,
                            timeout=torch.distributed.default_pg_timeout,
                        )
                        print("✓ 使用Gloo后端成功初始化进程组")
                    except Exception as e2:
                        print(f"错误: Gloo初始化也失败: {e2}")
                        raise
                else:
                    raise
        
        # 同步所有进程，确保所有进程都准备好后再设置设备
        if dist.is_initialized():
            try:
                # 注意：某些PyTorch版本不支持barrier的timeout参数
                dist.barrier()
            except TypeError:
                # 如果barrier不支持timeout参数，使用不带参数的版本
                dist.barrier()
            except Exception as e:
                # 如果barrier失败，可能是某个进程卡住了
                rank = dist.get_rank()
                print(f"[rank{rank}]: 警告: barrier同步失败: {e}")
                print(f"[rank{rank}]: 这可能是由于rank 0初始化失败导致的")
                # 继续执行，但在日志中记录
        
        # 现在安全地设置设备（所有进程已同步）
        try:
            torch.cuda.set_device(args.local_rank)
            # 验证设备设置是否成功
            test_tensor = torch.zeros(1, device=f"cuda:{args.local_rank}")
            del test_tensor
            torch.cuda.synchronize(args.local_rank)
        except RuntimeError as e:
            if "HSA" in str(e).upper() or "out of resources" in str(e).lower():
                print(f"警告: GPU {args.local_rank} 初始化时遇到HSA资源问题，等待后重试...")
                import time
                time.sleep(5)  # 等待5秒让其他进程完成初始化
                try:
                    torch.cuda.set_device(args.local_rank)
                    test_tensor = torch.zeros(1, device=f"cuda:{args.local_rank}")
                    del test_tensor
                    torch.cuda.synchronize(args.local_rank)
                    print(f"✓ GPU {args.local_rank} 重试初始化成功")
                except Exception as e2:
                    print(f"错误: GPU {args.local_rank} 初始化失败: {e2}")
                    raise
            else:
                raise
        
        device = torch.device(f"cuda:{args.local_rank}")
        
        # 再次同步，确保所有GPU都已正确初始化
        if dist.is_initialized():
            try:
                # 注意：某些PyTorch版本不支持barrier的timeout参数
                dist.barrier()
            except TypeError:
                dist.barrier()
            except Exception as e:
                rank = dist.get_rank()
                print(f"[rank{rank}]: 警告: GPU初始化后的barrier同步失败: {e}")
                # 不抛出异常，让训练继续，但在日志中记录问题
        
        is_main_process = dist.get_rank() == 0
        world_size = dist.get_world_size()
    else:
        is_main_process = True
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 50)
    print("基于 DeepSpeed + TRL 的32B模型SFT训练")
    print("=" * 50)
    if args.use_deepspeed:
        print("使用: DeepSpeed + TRL SFTTrainer")
    else:
        print("使用: 标准分布式训练 (DDP) + TRL SFTTrainer")
    if args.local_rank != -1:
        print(f"进程rank: {dist.get_rank()}, 总进程数: {world_size}")
    else:
        print(f"单GPU训练模式")
    print("=" * 50)

    # 加载分词器
    print(f"加载分词器: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载数据集
    print("加载数据集...")
    train_ds = load_dataset_for_sft(
        dataset_name_or_path=args.dataset,
        subset_name=args.subset_name,
        split="train",
        text_column=args.text_column,
        max_length=args.max_seq_length,
        tokenizer=tokenizer,  # 传递tokenizer以使用chat template
    )
    print(f"训练集大小: {len(train_ds)}")
    
    # 如果启用packing，设置数据集缓存以加速
    if args.packing and is_main_process:
        print("序列打包已启用，将自动优化序列长度分布以减少padding")

    # 计算全局批次大小和训练步数
    global_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    total_train_steps = args.num_train_epochs * len(train_ds) // global_batch_size if args.max_steps <= 0 else args.max_steps
    
    if is_main_process:
        print(f"全局批次大小: {global_batch_size}")
        print(f"总训练步数: {total_train_steps}")

    # 在加载模型前，确保所有进程已同步
    if args.local_rank != -1 and dist.is_initialized():
        try:
            # 注意：某些PyTorch版本不支持barrier的timeout参数
            dist.barrier()
        except TypeError:
            dist.barrier()
        except Exception as e:
            rank = dist.get_rank()
            print(f"[rank{rank}]: 警告: 模型加载前的barrier同步失败: {e}")
            # 继续执行
    
    # 加载模型
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    
    if is_main_process:
        print(f"加载模型: {args.model_name}")
        print(f"使用数据类型: {dtype}")
    
    # 对于 DeepSpeed，模型应该加载到 CPU，让 DeepSpeed 处理设备分配
    if torch.cuda.is_available():
        # 清理GPU缓存，释放可能残留的资源
        try:
            torch.cuda.empty_cache()
            if args.local_rank != -1:
                torch.cuda.synchronize(args.local_rank)
        except RuntimeError as e:
            if "HSA" not in str(e).upper():  # HSA错误可能表示设备未初始化，这是正常的
                print(f"警告: 清理GPU缓存时出错: {e}")
        import gc
        gc.collect()
    
    try:
        if is_main_process:
            print("开始加载模型权重...")
        
        # 对于 DeepSpeed，将模型加载到 CPU
        if args.use_deepspeed:
            if is_main_process:
                print("使用 DeepSpeed：模型将加载到 CPU，由 DeepSpeed 处理设备分配")
            # 注意：不要使用 device_map，因为 DeepSpeed 会自己处理设备分配
            # 使用 device_map 可能导致返回字典而不是模型对象
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
                # 不使用 device_map，直接加载到 CPU
            )
            # 确保模型所有参数都在 CPU 上
            model = model.cpu()
            # 确保模型是模型对象而不是字典
            if isinstance(model, dict):
                raise ValueError("模型加载返回了字典而不是模型对象，请检查 transformers 版本")
            for param in model.parameters():
                if param.is_cuda:
                    param.data = param.data.cpu()
            if is_main_process:
                print("✓ 模型已强制加载到 CPU，所有参数已验证在 CPU 上")
        else:
            # 非 DeepSpeed 模式，正常加载
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
            )
            model = model.to(device)
        
        if is_main_process:
            print("模型权重加载完成")
        
    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
            print(f"错误: 内存不足 (OOM) - {error_msg}")
            print("建议: 减小 batch_size 或 max_seq_length，或增加 gradient_accumulation_steps")
        raise
    
    # 启用梯度检查点（如果启用）
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if is_main_process:
                print("梯度检查点已启用")
        except Exception as e:
            if is_main_process:
                print(f"警告: 启用梯度检查点失败: {e}")
    
    # PyTorch 2.5 兼容性修复：确保 _parameters 是 OrderedDict 类型（DeepSpeed 需要）
    # 在 PyTorch 2.5 中，_parameters 从 OrderedDict 改为了 dict，导致 DeepSpeed 无法添加 _in_forward 属性
    if args.use_deepspeed:
        try:
            from collections import OrderedDict
            import torch.nn as nn
            
            # 检查 PyTorch 版本
            torch_version = torch.__version__.split('.')
            major, minor = int(torch_version[0]), int(torch_version[1])
            
            # 如果是 PyTorch 2.5+，需要修复 _parameters 类型
            if major == 2 and minor >= 5:
                if is_main_process:
                    print("检测到 PyTorch 2.5+，应用 DeepSpeed 兼容性修复...")
                
                # 递归修复所有子模块的 _parameters
                def fix_parameters_dict(module):
                    """将模块的 _parameters 从 dict 转换为 OrderedDict"""
                    if hasattr(module, '_parameters') and module._parameters is not None:
                        if not isinstance(module._parameters, OrderedDict):
                            # 保存原始参数
                            params = dict(module._parameters)
                            # 转换为 OrderedDict，保持顺序
                            module._parameters = OrderedDict(params)
                    
                    # 递归处理所有子模块
                    for child in module.children():
                        fix_parameters_dict(child)
                
                fix_parameters_dict(model)
                
                if is_main_process:
                    print("✓ DeepSpeed 兼容性修复已应用")
        except Exception as e:
            if is_main_process:
                print(f"警告: DeepSpeed 兼容性修复失败: {e}")
                print("  如果遇到 '_in_forward' 错误，请考虑降级 PyTorch 到 2.4 或更新 DeepSpeed")
    
    # 配置训练参数
    sft_cfg_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "eval_strategy": args.eval_strategy,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": False,
        "fp16": args.fp16 and not args.bf16,
        "bf16": args.bf16,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "report_to": ["none"],
        "gradient_checkpointing": args.gradient_checkpointing,
        "packing": args.packing,
        "remove_unused_columns": False,
        "dataset_text_field": args.text_column,
        "local_rank": args.local_rank,
    }

    # 如果使用8-bit优化器，设置优化器类型
    if args.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            if is_main_process:
                print("bitsandbytes已安装，可以使用8-bit优化器")
        except ImportError:
            if is_main_process:
                print("警告: 未安装bitsandbytes，8-bit优化器将无法使用")
                print("请运行: pip install bitsandbytes")
                print("将回退到标准优化器")
            args.use_8bit_optimizer = False

        if args.use_8bit_optimizer:
            if args.optim == "adamw_torch":
                sft_cfg_kwargs["optim"] = "paged_adamw_8bit"
            else:
                sft_cfg_kwargs["optim"] = args.optim
            if is_main_process:
                print("使用8-bit优化器以节省显存（可节省约50-75%优化器状态显存）")

    # 只在 max_steps > 0 时添加该参数
    if args.max_steps > 0:
        sft_cfg_kwargs["max_steps"] = args.max_steps

    # 只在 eval_strategy 为 "steps" 时添加 eval_steps
    if args.eval_strategy == "steps":
        sft_cfg_kwargs["eval_steps"] = args.eval_steps

    # 如果使用DeepSpeed，添加DeepSpeed配置
    if args.use_deepspeed:
        if args.deepspeed_config and os.path.exists(args.deepspeed_config):
            sft_cfg_kwargs["deepspeed"] = args.deepspeed_config
            if is_main_process:
                print(f"使用DeepSpeed配置文件: {args.deepspeed_config}")
        else:
            # 使用默认DeepSpeed配置（ZeRO Stage 3，最大内存节省）
            enable_optimizer_offload = args.deepspeed_offload_optimizer if hasattr(args, 'deepspeed_offload_optimizer') else True
            enable_param_offload = args.deepspeed_offload_param if hasattr(args, 'deepspeed_offload_param') else True
            
            # 根据激进内存优化选项调整参数
            if args.deepspeed_aggressive_memory:
                reduce_bucket_size = args.deepspeed_reduce_bucket_size if args.deepspeed_reduce_bucket_size else 5e6  # 5MB
                stage3_prefetch_bucket_size = 2e6  # 2MB
                stage3_param_persistence_threshold = 1e4
                stage3_max_live_parameters = args.deepspeed_stage3_max_live_params if args.deepspeed_stage3_max_live_params else 5e7  # 5千万
                stage3_max_reuse_distance = 5e7  # 5千万
                allgather_bucket_size = 5e6  # 5MB
                if is_main_process:
                    print("⚠️  启用DeepSpeed激进内存优化模式（最大化显存节省）")
            else:
                reduce_bucket_size = args.deepspeed_reduce_bucket_size if args.deepspeed_reduce_bucket_size else 2e7  # 20MB
                stage3_prefetch_bucket_size = 1e7  # 10MB
                stage3_param_persistence_threshold = 1e5
                stage3_max_live_parameters = args.deepspeed_stage3_max_live_params if args.deepspeed_stage3_max_live_params else 2e8  # 2亿
                stage3_max_reuse_distance = 2e8  # 2亿
                allgather_bucket_size = 2e7  # 20MB
            
            if is_main_process:
                print("使用默认DeepSpeed配置（ZeRO Stage 3，最大化内存节省）")
                if enable_optimizer_offload:
                    print("  ✓ 优化器CPU Offload: 已启用")
                if enable_param_offload:
                    print("  ✓ 参数CPU Offload: 已启用")
            
            # 构建ZeRO配置
            zero_config = {
                "stage": 3,
                "overlap_comm": False,  # 关闭重叠通信，避免死锁
                "contiguous_gradients": True,
                "reduce_bucket_size": int(reduce_bucket_size),
                "stage3_prefetch_bucket_size": int(stage3_prefetch_bucket_size),
                "stage3_param_persistence_threshold": int(stage3_param_persistence_threshold),
                "stage3_max_live_parameters": int(stage3_max_live_parameters),
                "stage3_max_reuse_distance": int(stage3_max_reuse_distance),
                "stage3_gather_16bit_weights_on_model_save": True,
                "allgather_partitions": True,
                "allgather_bucket_size": int(allgather_bucket_size),
                "reduce_scatter": True,
                "round_robin_gradients": False
            }
            
            # 添加优化器offload配置
            if enable_optimizer_offload:
                zero_config["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": False
                }
            
            # 添加参数offload配置（Stage 3专用）
            if enable_param_offload:
                zero_config["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": False
                }
            
            sft_cfg_kwargs["deepspeed"] = {
                "zero_optimization": zero_config,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "gradient_clipping": 1.0,
                "train_batch_size": global_batch_size,
                "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
                "bf16": {
                    "enabled": args.bf16
                },
                "fp16": {
                    "enabled": args.fp16 and not args.bf16
                },
                "wall_clock_breakdown": False,
                "steps_per_print": args.logging_steps,
            }

    # 创建SFTConfig
    sft_cfg = SFTConfig(**sft_cfg_kwargs)

    # 创建 data collator（仅在未启用packing时使用）
    # 注意：当 packing=True 时，TRL 使用 padding-free 模式，不支持自定义 data collator
    if args.packing:
        data_collator = None
        if is_main_process:
            print("序列打包已启用，将使用 TRL 内置的 padding-free data collator")
    else:
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        )
        if is_main_process:
            print("使用自定义 data collator（packing 未启用）")

    # 创建训练器
    if is_main_process:
        print("开始训练...")
        print(f"训练参数:")
        print(f"  全局批次大小: {global_batch_size}")
        print(f"  每设备批次大小: {args.per_device_train_batch_size}")
        print(f"  梯度累积步数: {args.gradient_accumulation_steps}")
        print(f"  总训练步数: {total_train_steps}")
        if args.packing:
            print(f"  序列打包: 已启用（减少padding浪费）")
    
    # 在创建 SFTTrainer 之前，确保模型对象是正确的类型（不是字典）
    if isinstance(model, dict):
        raise ValueError("错误: 模型对象是字典而不是模型实例。这可能是由于 transformers 版本问题导致的。")
    
    # 验证模型对象有必要的属性
    if not hasattr(model, 'forward') or not hasattr(model, 'parameters'):
        raise ValueError("错误: 模型对象缺少必要的属性（forward 或 parameters）。")
    
    # 构建 trainer kwargs
    trainer_kwargs = {
        "model": model,
        "args": sft_cfg,
        "train_dataset": train_ds,
        "processing_class": tokenizer,
    }
    # 仅在未启用 packing 时传递 data_collator
    if not args.packing:
        trainer_kwargs["data_collator"] = data_collator
    
    trainer = SFTTrainer(**trainer_kwargs)
    
    # DeepSpeed 会在 Trainer.train() 开始时自动初始化
    if args.use_deepspeed:
        if is_main_process:
            print("\n✓ DeepSpeed 配置已就绪")
            print("  模型当前在 CPU 上，DeepSpeed 将在训练开始时自动分片到所有 GPU")
            print("  使用 ZeRO Stage 3 + CPU Offload，最大化内存节省")
        
        # 同步所有进程
        if args.local_rank != -1 and dist.is_initialized():
            try:
                # 注意：某些PyTorch版本不支持barrier的timeout参数
                dist.barrier()
            except TypeError:
                dist.barrier()
            except Exception as e:
                rank = dist.get_rank()
                print(f"[rank{rank}]: 警告: DeepSpeed初始化前的barrier同步失败: {e}")
        
        # 清理内存，准备训练
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 开始训练
    try:
        trainer.train()
    except AttributeError as e:
        error_msg = str(e)
        if "_in_forward" in error_msg and "dict" in error_msg:
            if is_main_process:
                print("\n" + "="*50)
                print("DeepSpeed 兼容性错误诊断:")
                print("="*50)
                print("错误: AttributeError: 'dict' object has no attribute '_in_forward'")
                print("\n原因:")
                print("这是 PyTorch 2.5+ 与 DeepSpeed 的兼容性问题。")
                print("在 PyTorch 2.5 中，module._parameters 从 OrderedDict 改为了 dict，")
                print("导致 DeepSpeed 无法添加 _in_forward 属性。")
                print("\n解决方案:")
                print("1. 降级 PyTorch 到 2.4: pip install torch==2.4.*")
                print("2. 更新 DeepSpeed 到最新版本: pip install --upgrade deepspeed")
                print("3. 如果问题仍然存在，请检查 DeepSpeed GitHub 是否有相关修复")
                print("="*50)
            raise RuntimeError(f"DeepSpeed 兼容性错误: {error_msg}\n请参考上面的解决方案。") from e
        raise
    except RuntimeError as e:
        error_msg = str(e)
        if is_main_process:
            print(f"训练过程中发生运行时错误: {error_msg}")
            import traceback
            traceback.print_exc()
            
            if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                print("\n" + "="*50)
                print("内存不足 (OOM) 错误诊断:")
                print("="*50)
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        print(f"GPU {i}: 已分配 {allocated:.2f}GB / 已保留 {reserved:.2f}GB / 总计 {total:.2f}GB")
                print("\n建议解决方案:")
                print("1. 减小 per_device_train_batch_size")
                print("2. 减小 max_seq_length")
                print("3. 增加 gradient_accumulation_steps 来补偿")
                print("4. 确保启用了 gradient_checkpointing")
                print("5. 如果使用 DeepSpeed，考虑启用 CPU offload")
                print("="*50)
        raise
    
    # 保存最终模型
    if is_main_process:
        print(f"\n训练完成，保存最终模型到: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if is_main_process:
        print("模型和tokenizer保存完成")
        print("训练完成！")
    
    # 清理分布式环境
    if args.local_rank != -1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
