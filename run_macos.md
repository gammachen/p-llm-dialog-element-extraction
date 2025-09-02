# macOS训练命令

## 使用MPS (Apple Silicon GPU) 训练
```bash
python train_macos.py \
  --train_path data/train.json \
  --test_path data/test.json \
  --model_name_or_path Qwen-1_8-chat/ \
  --per_device_train_batch_size 2 \
  --max_len 2048 \
  --max_src_len 1560 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 4 \
  --warmup_ratio 0.03 \
  --seed 1234 \
  --show_loss_step 10 \
  --lora_dim 16 \
  --lora_alpha 64 \
  --save_model_step 100 \
  --lora_dropout 0.1 \
  --output_dir ./output_dir_qlora_macos \
  --is_skip
```

## 强制使用CPU训练（如果MPS有问题）
```bash
python train_macos.py \
  --train_path data/train.json \
  --test_path data/test.json \
  --model_name_or_path Qwen-1_8-chat/ \
  --per_device_train_batch_size 1 \
  --max_len 1024 \
  --max_src_len 512 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 8 \
  --warmup_ratio 0.03 \
  --seed 1234 \
  --show_loss_step 5 \
  --lora_dim 16 \
  --lora_alpha 64 \
  --save_model_step 50 \
  --lora_dropout 0.1 \
  --output_dir ./output_dir_qlora_cpu \
  --use_cpu \
  --is_skip
```

## 内存优化建议

对于macOS系统，建议：

1. **Apple Silicon (M1/M2/M3)**:
   - 使用MPS设备可以获得GPU加速
   - 适当减小batch_size和max_len
   - 建议使用`--per_device_train_batch_size 1-2`

2. **Intel Mac**:
   - 只能使用CPU训练
   - 显著减小batch_size和max_len
   - 建议使用`--per_device_train_batch_size 1`
   - 考虑使用`--max_len 512-1024`

3. **通用优化**:
   - 增加`--gradient_accumulation_steps`来模拟更大的batch_size
   - 减小`--max_len`和`--max_src_len`减少内存占用
   - 使用`--save_model_step`更频繁地保存检查点