
import argparse
import json
import math
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import deepspeed
from transformers import BitsAndBytesConfig
from utils import print_trainable_parameters, print_rank_0, to_device, set_random_seed, save_model, DataCollator, \
    find_all_linear_names, evaluation
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen1_8.modeling_qwen import QWenLMHeadModel
from qwen1_8.tokenization_qwen import QWenTokenizer
from qwen1_8.configuration_qwen import QWenConfig
from utils import QwenPromptDataSet
import os

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter


def parse_args():
    """
    解析命令行参数
    
    Args:
        --model_name_or_path (str): 模型名称或路径，必需参数
        --train_path (str): 训练数据路径，默认为空字符串
        --test_path (str): 测试数据路径，默认为空字符串
        --max_len (int): 最大序列长度，默认为1024
        --max_src_len (int): 最大源序列长度，默认为256
        --is_skip (bool): 是否跳过某些处理步骤，store_true类型参数
        --per_device_train_batch_size (int): 每个设备的训练批次大小，默认为16
        --learning_rate (float): 学习率，默认为1e-3
        --weight_decay (float): 权重衰减系数，默认为0.1
        --num_train_epochs (int): 训练轮数，默认为1
        --gradient_accumulation_steps (int): 梯度累积步数，默认为1
        --warmup_ratio (float): warmup比例，默认为0.1
        --output_dir (str): 输出目录，默认为None
        --seed (int): 随机种子，默认为1234
        --local_rank (int): 本地rank，默认为-1
        --show_loss_step (int): 显示损失的步长，默认为10
        --gradient_checkpointing (bool): 是否启用梯度检查点，store_true类型参数
        --save_model_step (int): 保存模型的步长，默认为None
        --ds_file (str): DeepSpeed配置文件，默认为"ds_zero2.json"
        --lora_dim (int): LoRA维度，默认为8
        --lora_alpha (int): LoRA alpha值，默认为30
        --lora_dropout (float): LoRA dropout率，默认为0.1
        
    Returns:
        argparse.Namespace: 解析后的命令行参数对象
    """
    parser = argparse.ArgumentParser()
    # 模型配置
    parser.add_argument("--model_name_or_path", type=str, help="", required=True)
    # 数据配置
    parser.add_argument("--train_path", default="", type=str, help="")
    parser.add_argument("--test_path", default="", type=str, help="")
    parser.add_argument("--max_len", type=int, default=1024, help="")
    parser.add_argument("--max_src_len", type=int, default=256, help="")
    parser.add_argument("--is_skip", action='store_true', help="")
    # 训练配置
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="")
    parser.add_argument("--output_dir", type=str, default=None, help="")
    parser.add_argument("--seed", type=int, default=1234, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--show_loss_step", default=10, type=int, help="")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="")
    parser.add_argument("--save_model_step", default=None, type=int, help="")
    # DeepSpeed配置
    parser.add_argument("--ds_file", type=str, default="ds_zero2.json", help="")
    # QLoRA配置
    parser.add_argument("--lora_dim", type=int, default=8, help="")
    parser.add_argument("--lora_alpha", type=int, default=30, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="")

    # 添加DeepSpeed配置参数
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def train():
    """
    训练函数，用于训练Qwen模型
    
    该函数执行以下主要步骤：
    1. 解析命令行参数
    2. 初始化分布式训练环境（如果需要）
    3. 设置随机种子以确保结果可复现
    4. 加载预训练的Qwen模型和分词器
    5. 使用LoRA技术对模型进行微调配置
    6. 加载训练和测试数据集
    7. 配置DeepSpeed优化器和调度器
    8. 执行模型训练循环，包括前向传播、损失计算和反向传播
    9. 定期保存模型检查点并评估性能
    
    Args:
        无显式参数，但会从parse_args()中获取训练配置
        
    Note:
        该函数依赖多个全局配置和工具函数，包括模型定义、数据集处理、分布式训练设置等
    """
    # 解析命令行参数，获取训练配置
    args = parse_args()
    
    # 判断是多卡训练还是单卡训练，local_rank为-1表示单卡训练
    if args.local_rank == -1:
        # 单卡训练，直接使用cuda设备
        device = torch.device("cuda")
    else:
        # 多卡训练，设置当前进程使用的GPU设备
        torch.cuda.set_device(args.local_rank)
        # 创建指定GPU的cuda设备对象
        device = torch.device("cuda", args.local_rank)
        # 初始化deepspeed分布式训练环境
        deepspeed.init_distributed()
    
    # 获取当前进程的全局rank
    args.global_rank = torch.distributed.get_rank()
    
    # 设置tensorboard，只在主进程上记录训练过程中的loss以及ppl
    if args.global_rank <= 0:
        tb_write = SummaryWriter()
    
    # 设置随机种子，确保模型训练结果可复现
    set_random_seed(args.seed)
    
    # 在所有进程之间设置屏障，确保同步
    torch.distributed.barrier()
    
    # 加载千问模型分词器，从预训练模型路径加载
    tokenizer = QWenTokenizer.from_pretrained(args.model_name_or_path)
    
    # 设置分词器的pad token id为eod id
    tokenizer.pad_token_id = tokenizer.eod_id
    
    # 加载千问模型
    # 设置设备映射，使用环境变量中的LOCAL_RANK或默认为0
    device_map = {'': int(os.environ.get('LOCAL_RANK', '0'))}
    
    # 从预训练模型路径加载模型配置
    model_config = QWenConfig.from_pretrained(args.model_name_or_path)
    
    # 从预训练模型路径加载模型，并配置4bit量化参数
    model = QWenLMHeadModel.from_pretrained(args.model_name_or_path,
                                            quantization_config=BitsAndBytesConfig(
                                                load_in_4bit=True,                 # 启用4bit量化
                                                bnb_4bit_compute_dtype=model_config.torch_dtype,  # 计算数据类型
                                                bnb_4bit_use_double_quant=True,    # 使用双重量化
                                                bnb_4bit_quant_type="nf4",         # 量化类型为nf4
                                                llm_int8_threshold=6.0,            # int8阈值
                                                llm_int8_has_fp16_weight=False,    # 不使用fp16权重
                                            ),
                                            torch_dtype=model_config.torch_dtype,  # 设置模型数据类型
                                            device_map=device_map)                 # 设置设备映射
    
    # 为kbit训练准备模型
    model = prepare_model_for_kbit_training(model)
    
    # 找到模型中所有的全连接层，用于后续LoRA配置
    lora_module_name = find_all_linear_names(model)
    
    # 设置Lora配置，并生成外挂可训练参数
    # 创建LoRA配置对象，指定相关参数
    config = LoraConfig(r=args.lora_dim,                      # LoRA秩
                        lora_alpha=args.lora_alpha,           # LoRA alpha值
                        target_modules=lora_module_name,      # 目标模块名称列表
                        lora_dropout=args.lora_dropout,       # LoRA dropout率
                        bias="none",                          # 不训练bias参数
                        task_type="CAUSAL_LM",                # 任务类型为因果语言模型
                        inference_mode=False,                 # 不是推理模式
                        )
    
    # 使用peft库将LoRA配置应用到模型上
    model = get_peft_model(model, config)
    
    # 设置模型配置的torch数据类型为float32
    model.config.torch_dtype = torch.float32
    
    # 打印可训练参数
    # 遍历模型的命名参数，打印需要训练的参数名称
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print_rank_0(name, 0)
    
    # 打印模型的可训练参数统计信息
    print_trainable_parameters(model)

    # 加载模型训练所需要的数据，如果是多卡训练需要分布式加载数据
    # 创建训练数据集对象
    train_dataset = QwenPromptDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.is_skip)
    
    # 创建测试数据集对象
    test_dataset = QwenPromptDataSet(args.test_path, tokenizer, args.max_len, args.max_src_len, args.is_skip)
    
    # 根据是否为分布式训练设置不同的采样器
    if args.local_rank == -1:
        # 单卡训练使用随机采样器和顺序采样器
        train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)
    else:
        # 多卡训练使用分布式采样器
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)

    # 创建数据整理器
    data_collator = DataCollator(tokenizer)
    
    # 创建训练数据加载器
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    
    # 创建测试数据加载器
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, sampler=test_sampler, batch_size=args.per_device_train_batch_size)
    
    # 打印训练数据加载器的长度信息
    print_rank_0("len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
    
    # 打印训练数据集的长度信息
    print_rank_0("len(train_dataset) = {}".format(len(train_dataset)), args.global_rank)

    # 加载DeepSpeed配置文件，并进行修改
    # 读取DeepSpeed配置文件
    with open(args.ds_file, "r", encoding="utf-8") as fh:
        ds_config = json.load(fh)
    
    # 设置每个GPU的微批次大小
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    
    # 设置总的训练批次大小
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    
    # 设置梯度累积步数
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    
    # 配置优化器参数
    ds_config["optimizer"]["params"]["lr"] = args.learning_rate       # 学习率
    ds_config["optimizer"]["params"]["betas"] = (0.9, 0.95)          # Adam优化器的beta参数
    ds_config["optimizer"]["params"]["eps"] = 1e-8                   # Adam优化器的epsilon参数
    ds_config["optimizer"]["params"]["weight_decay"] = 0.1           # 权重衰减
    
    # 计算总的训练步数
    num_training_steps = args.num_train_epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    # 打印总的训练步数
    print_rank_0("num_training_steps = {}".format(num_training_steps), args.global_rank)
    
    # 计算warmup步数
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    # 打印warmup步数
    print_rank_0("num_warmup_steps = {}".format(num_warmup_steps), args.global_rank)
    
    # 配置学习率调度器参数
    ds_config["scheduler"]["params"]["total_num_steps"] = num_training_steps    # 总步数
    ds_config["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps     # warmup步数
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.learning_rate      # warmup最大学习率
    ds_config["scheduler"]["params"]["warmup_min_lr"] = args.learning_rate * 0.1  # warmup最小学习率

    # 设置模型gradient_checkpointing
    # 如果启用了梯度检查点
    if args.gradient_checkpointing:
        # 启用模型的梯度检查点功能
        model.gradient_checkpointing_enable()
        
        # 检查模型是否有enable_input_require_grads方法
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # 定义使输入需要梯度的函数
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
                
            # 为模型的输入嵌入层注册前向钩子
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # DeepSeed对模型进行初始化
    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, args=args, config=ds_config, dist_init_required=True)
    
    # 初始化训练损失、日志损失和最小损失
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    
    # 初始化全局步数
    global_step = 0
    
    # 模型开始训练
    # 遍历训练轮数
    for epoch in range(args.num_train_epochs):
        # 打印当前训练轮数信息
        print_rank_0("Beginning of Epoch {}/{}, Total Micro Batches {}".format(epoch + 1, args.num_train_epochs, len(train_dataloader)), args.global_rank)
        
        # 设置模型为训练模式
        model.train()
        
        # 遍历所有数据
        # 使用tqdm显示训练进度
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch"):
            # 将批次数据移动到指定设备上
            batch = to_device(batch, device)
            
            # 获取训练结果
            outputs = model(**batch, use_cache=False)
            
            # 获取损失值
            loss = outputs.loss
            
            # 损失进行回传
            model.backward(loss)
            
            # 累计训练损失
            tr_loss += loss.item()
            
            # 对模型参数进行梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 执行优化器步骤
            model.step()
            
            # 当训练步数整除累积步数时，记录训练损失值和模型保存
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 增加全局步数
                global_step += 1
                
                # 损失值记录
                # 如果达到显示损失的步数
                if global_step % args.show_loss_step == 0:
                    # 只在主进程上记录
                    if args.global_rank <= 0:
                        # 将训练损失写入tensorboard
                        tb_write.add_scalar("train_loss", (tr_loss - logging_loss) / (args.show_loss_step * args.gradient_accumulation_steps), global_step)
                        # 更新日志损失
                        logging_loss = tr_loss
                
                # 模型保存并验证测试集的PPL值
                # 如果设置了模型保存步数且达到保存条件
                if args.save_model_step is not None and global_step % args.save_model_step == 0:
                    # 在测试集上评估模型
                    ppl = evaluation(model, test_dataloader, device)
                    
                    # 只在主进程上记录和保存
                    if args.global_rank <= 0:
                        # 将ppl值写入tensorboard
                        tb_write.add_scalar("ppl", ppl, global_step)
                        # 打印ppl值
                        print_rank_0("save_model_step-{}: ppl-{}".format(global_step, ppl), args.global_rank)
                    
                    # 在主进程上保存模型
                    if args.global_rank <= 0:
                        # 保存模型
                        save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")
                    
                    # 重新设置模型为训练模式
                    model.train()
        
        # 每个Epoch对模型进行一次测试，记录测试集的损失
        # 在测试集上评估模型
        ppl = evaluation(model, test_dataloader, device)
        
        # 只在主进程上记录和保存
        if args.global_rank <= 0:
            # 将ppl值写入tensorboard
            tb_write.add_scalar("ppl", ppl, global_step)
            # 打印ppl值
            print_rank_0("save_model_step-{}: ppl-{}".format(global_step, ppl), args.global_rank)
        
        # 在主进程上保存模型
        if args.global_rank <= 0:
            # 保存模型
            save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")


if __name__ == "__main__":
    train()
