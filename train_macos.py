#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本 - macOS适配版本
适配macOS系统，支持MPS (Metal Performance Shaders) 和CPU训练
"""

import os
import math
import json
import torch
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import argparse
import numpy as np

# 设置日志格式
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置环境变量避免tokenizers警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 尝试导入必要的库
try:
    from qwen1_8.modeling_qwen import QWenLMHeadModel
    from qwen1_8.tokenization_qwen import QWenTokenizer
    from qwen1_8.configuration_qwen import QWenConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from transformers import BitsAndBytesConfig
    from tensorboardX import SummaryWriter
except ImportError as e:
    logger.error(f"Missing required library: {e}")
    logger.error("Please install: pip install peft bitsandbytes tensorboardX")
    exit(1)

class QwenPromptDataSet(Dataset):
    """千问提示数据集类"""
    
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip=True):
        """
        初始化数据集
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_len: 最大序列长度
            max_src_len: 最大源序列长度
            is_skip: 是否跳过超长样本
        """
        self.data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {data_path}: {e}")
            exit(1)
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_src_len = max_src_len
        self.is_skip = is_skip
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            处理后的样本数据
        """
        try:
            item = self.data[idx]
            
            # 构建输入文本
            input_text = str(item.get('input', ''))
            output_text = str(item.get('output', ''))
            
            if not input_text or not output_text:
                return None
            
            # 分词处理
            input_ids = self.tokenizer.encode(input_text, max_length=self.max_src_len, truncation=True)
            output_ids = self.tokenizer.encode(output_text, max_length=self.max_len - len(input_ids) - 1, truncation=True)
            
            # 构建完整的输入序列
            full_input_ids = input_ids + output_ids + [self.tokenizer.eod_id]
            
            # 检查序列长度
            if len(full_input_ids) > self.max_len:
                if self.is_skip:
                    return None
                else:
                    full_input_ids = full_input_ids[:self.max_len]
            
            # 构建标签，输入部分为-100（不计算损失）
            labels = [-100] * len(input_ids) + output_ids + [self.tokenizer.eod_id]
            
            if len(labels) > len(full_input_ids):
                labels = labels[:len(full_input_ids)]
            elif len(labels) < len(full_input_ids):
                labels.extend([-100] * (len(full_input_ids) - len(labels)))
            
            return {
                'input_ids': torch.tensor(full_input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.ones(len(full_input_ids), dtype=torch.long)
            }
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            return None

class DataCollator:
    """数据整理器"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.eod_id
    
    def __call__(self, features):
        """
        整理批次数据
        Args:
            features: 批次样本列表
        Returns:
            整理后的批次数据
        """
        # 过滤None值
        features = [f for f in features if f is not None]
        
        if len(features) == 0:
            return None
        
        # 获取最大长度
        max_len = max([len(f['input_ids']) for f in features])
        
        # 填充到最大长度
        input_ids = []
        labels = []
        attention_masks = []
        
        for f in features:
            pad_len = max_len - len(f['input_ids'])
            
            # 创建填充后的序列
            padded_input = torch.cat([f['input_ids'], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            padded_labels = torch.cat([f['labels'], torch.full((pad_len,), -100, dtype=torch.long)])
            
            # 创建注意力掩码：1表示真实token，0表示填充token
            attention_mask = torch.cat([
                f['attention_mask'], 
                torch.zeros(pad_len, dtype=torch.long)
            ])
            
            input_ids.append(padded_input)
            labels.append(padded_labels)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attention_masks)
        }

def find_all_linear_names(model):
    """
    找到模型中所有的全连接层名称
    Args:
        model: 模型对象
    Returns:
        全连接层名称列表
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

def print_trainable_parameters(model):
    """
    打印可训练参数信息
    Args:
        model: 模型对象
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%")

def evaluation(model, eval_dataloader, device):
    """
    评估模型在验证集上的表现
    Args:
        model: 模型对象
        eval_dataloader: 验证数据加载器
        device: 计算设备
    Returns:
        困惑度 (perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    try:
        with torch.no_grad():
            for batch in eval_dataloader:
                if batch is None:
                    continue
                    
                # 移动数据到设备
                for key in batch:
                    batch[key] = batch[key].to(device)
                
                # 前向传播
                outputs = model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                total_steps += 1
        
        if total_steps == 0:
            return float('inf')
        
        avg_loss = total_loss / total_steps
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return perplexity.item()
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return float('inf')

def save_model(model, tokenizer, output_dir, checkpoint_name):
    """
    保存模型
    Args:
        model: 模型对象
        tokenizer: 分词器
        output_dir: 输出目录
        checkpoint_name: 检查点名称
    """
    try:
        save_path = os.path.join(output_dir, checkpoint_name)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存LoRA权重
        model.save_pretrained(save_path)
        
        # 保存分词器
        tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def train():
    """训练主函数"""
    parser = argparse.ArgumentParser(description="Qwen LoRA Training on macOS")
    parser.add_argument("--train_path", type=str, default="data/train.json", help="训练数据路径")
    parser.add_argument("--test_path", type=str, default="data/test.json", help="测试数据路径")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen-1_8-chat/", help="预训练模型路径")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="评估批次大小")
    parser.add_argument("--max_len", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--max_src_len", type=int, default=512, help="最大源序列长度")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="warmup比例")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("--show_loss_step", type=int, default=10, help="显示损失步数")
    parser.add_argument("--lora_dim", type=int, default=16, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha值")
    parser.add_argument("--save_model_step", type=int, default=100, help="保存模型步数")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout率")
    parser.add_argument("--output_dir", type=str, default="./output_dir_qlora_macos", help="输出目录")
    parser.add_argument("--is_skip", action='store_true', help="是否跳过超长样本")
    parser.add_argument("--use_cpu", action='store_true', help="强制使用CPU")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 检测设备
    if args.use_cpu:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device (MPS not available)")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置tensorboard
    try:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    except ImportError:
        logger.warning("tensorboardX not available, skipping tensorboard logging")
        tb_writer = None
    
    # 加载分词器
    # 在模型加载后，确保pad token设置正确
    try:
        tokenizer = QWenTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eod_id
        logger.info("Tokenizer loaded successfully")
        logger.info(f"Pad token: {tokenizer.pad_token}, Pad token ID: {tokenizer.pad_token_id}")
        logger.info(f"EOS token: {tokenizer.eos_token}, EOS token ID: {tokenizer.eos_token_id}")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        exit(1)
    
    # 加载模型配置
    try:
        model_config = QWenConfig.from_pretrained(args.model_name_or_path)
        logger.info("Model config loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model config: {e}")
        exit(1)
    
    # 加载模型（使用4bit量化）
    logger.info("Loading model...")
    try:
        model = QWenLMHeadModel.from_pretrained(
            args.model_name_or_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            ),
            torch_dtype=torch.float16,
            device_map="auto" if device.type == "cpu" else {"": device},
            trust_remote_code=True
        )
        
        # 为kbit训练准备模型
        model = prepare_model_for_kbit_training(model)
        logger.info("Model loaded and prepared for kbit training")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Make sure you have the Qwen model files in the specified directory")
        exit(1)
    
    # 找到所有线性层
    try:
        lora_module_name = find_all_linear_names(model)
        logger.info(f"Found {len(lora_module_name)} linear modules for LoRA")
    except Exception as e:
        logger.error(f"Error finding linear modules: {e}")
        exit(1)
    
    # 配置LoRA
    try:
        lora_config = LoraConfig(
            r=args.lora_dim,
            lora_alpha=args.lora_alpha,
            target_modules=lora_module_name,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        logger.info("LoRA configuration applied successfully")
    except Exception as e:
        logger.error(f"Error applying LoRA: {e}")
        exit(1)
    
    # 打印可训练参数
    print_trainable_parameters(model)
    
    # 移动模型到设备
    if device.type != "cpu":
        model = model.to(device)
    
    # 加载数据集
    logger.info("Loading datasets...")
    try:
        train_dataset = QwenPromptDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.is_skip)
        test_dataset = QwenPromptDataSet(args.test_path, tokenizer, args.max_len, args.max_src_len, args.is_skip)
        logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        exit(1)
    
    # 创建数据加载器
    try:
        train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)
        data_collator = DataCollator(tokenizer)
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=True,
            num_workers=0  # macOS compatibility
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.per_device_eval_batch_size,
            sampler=test_sampler,
            collate_fn=data_collator,
            num_workers=0  # macOS compatibility
        )
        
        logger.info(f"Train batches: {len(train_dataloader)}, Test batches: {len(test_dataloader)}")
    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        exit(1)
    
    # 设置优化器
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 计算训练步数
        num_training_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps
        num_warmup_steps = int(args.warmup_ratio * num_training_steps)
        
        # 设置学习率调度器
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps
        )
        
        logger.info(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
    except Exception as e:
        logger.error(f"Error setting up optimizer/scheduler: {e}")
        exit(1)
    
    # 训练循环
    logger.info("Starting training...")
    model.train()
    
    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    
    try:
        for epoch in range(args.num_train_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")
            
            epoch_loss = 0.0
            epoch_steps = 0
            
            for step, batch in enumerate(train_dataloader):
                if batch is None:
                    continue
                
                # 移动数据到设备
                for key in batch:
                    batch[key] = batch[key].to(device)
                
                # 前向传播
                outputs = model(**batch)
                loss = outputs.loss
                
                # 损失缩放
                loss = loss / args.gradient_accumulation_steps
                
                # 反向传播
                loss.backward()
                
                tr_loss += loss.item()
                epoch_loss += loss.item()
                epoch_steps += 1
                
                # 梯度累积
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # 优化器步骤
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # 记录损失
                    if global_step % args.show_loss_step == 0:
                        avg_loss = (tr_loss - logging_loss) / args.show_loss_step
                        logger.info(f"Step {global_step}: loss = {avg_loss:.4f}")
                        if tb_writer:
                            tb_writer.add_scalar("train_loss", avg_loss, global_step)
                        logging_loss = tr_loss
                    
                    # 保存模型
                    if args.save_model_step and global_step % args.save_model_step == 0:
                        try:
                            ppl = evaluation(model, test_dataloader, device)
                            logger.info(f"Step {global_step}: ppl = {ppl:.4f}")
                            if tb_writer:
                                tb_writer.add_scalar("eval_ppl", ppl, global_step)
                            
                            save_model(model, tokenizer, args.output_dir, f"checkpoint-{global_step}")
                            model.train()
                        except Exception as e:
                            logger.error(f"Error during evaluation: {e}")
            
            # 每个epoch结束后评估
            try:
                ppl = evaluation(model, test_dataloader, device)
                logger.info(f"Epoch {epoch + 1} completed: ppl = {ppl:.4f}")
                if tb_writer:
                    tb_writer.add_scalar("eval_ppl", ppl, global_step)
                
                # 保存epoch模型
                save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}")
            except Exception as e:
                logger.error(f"Error during epoch evaluation: {e}")
        
        # 保存最终模型
        save_model(model, tokenizer, args.output_dir, "final")
        logger.info("Training completed!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        save_model(model, tokenizer, args.output_dir, "interrupted")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        save_model(model, tokenizer, args.output_dir, "error")
    finally:
        if tb_writer:
            tb_writer.close()

if __name__ == "__main__":
    train()