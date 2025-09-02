import torch
import json
import os
import sys
from modelscope import AutoTokenizer, AutoModelForCausalLM


def read_multiline_input(prompt="请输入内容（输入END结束）:\n"):
    """
    读取多行输入，避免单行输入长度限制
    Args:
        prompt: 输入提示
    Returns:
        完整的输入文本
    """
    print(prompt)
    lines = []
    
    try:
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
    except KeyboardInterrupt:
        return None
    except EOFError:
        return None
    
    return "\n".join(lines).strip()


def read_single_line_input(prompt="请输入内容: ", max_length=1000):
    """
    读取单行输入，带有长度限制和验证
    Args:
        prompt: 输入提示
        max_length: 最大长度限制
    Returns:
        输入文本或None
    """
    try:
        print(prompt)
        text = input()
        
        if len(text) > max_length:
            print(f"警告：输入文本过长（{len(text)}字符），将被截断为前{max_length}字符")
            return text[:max_length]
        
        return text.strip()
    except KeyboardInterrupt:
        return None
    except EOFError:
        return None


def predict_with_modelscope(model, tokenizer, instruction, text, max_length=2048):
    """
    使用ModelScope的QWen模型进行对话要素抽取
    Args:
        model: QWen模型
        tokenizer: 分词器
        instruction: 提示词内容
        text: 对话内容
        max_length: 最大生成长度

    Returns:
        模型响应结果
    """
    try:
        # 检查输入长度
        if not text or not text.strip():
            return "错误：输入文本不能为空"
        
        if len(text) > 2000:
            print(f"警告：输入文本过长（{len(text)}字符），可能影响性能")
        
        # 构建输入文本
        input_text = instruction + text
        print(f"输入文本长度: {len(input_text)}字符")
        
        # 获取模型设备
        device = next(model.parameters()).device
        print(f"使用设备: {device}")
        
        # 对输入进行编码
        inputs = tokenizer(input_text, return_tensors="pt")
        print(f"编码完成，输入张量形状: {inputs['input_ids'].shape}")
        
        # 移动输入到模型所在设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 检查输入是否有效
        if inputs['input_ids'] is None or inputs['input_ids'].size(1) == 0:
            return "错误：输入编码失败"
        
        # 生成响应
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        print(f"生成完成，输出张量形状: {outputs.shape}")
        
        # 解码输出
        input_length = inputs['input_ids'].size(1)
        print(f"输入长度: {input_length}, 输出长度: {outputs.size(1)}")
        
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"预测时发生错误: {str(e)}\n详细错误:\n{error_details}"


def safe_input_handler():
    """
    安全的输入处理器
    """
    print("\n" + "=" * 60)
    print("对话要素抽取工具 - ModelScope QWen-1_8B-Chat")
    print("=" * 60)
    print("输入方式说明:")
    print("1. 单行输入：直接输入文本，按回车结束")
    print("2. 多行输入：输入'MULTI'开始多行输入，输入'END'结束")
    print("3. 文件输入：输入'FILE'从文件读取")
    print("4. 退出：输入'EXIT'或按Ctrl+C")
    print("-" * 60)
    
    while True:
        try:
            choice = input("\n请选择输入方式 (1/2/3/4): ").strip().upper()
            
            if choice in ['1', 'SINGLE', '']:
                # 单行输入
                text = read_single_line_input("请输入对话内容: ")
                if text is None:
                    continue
                return text
                
            elif choice in ['2', 'MULTI']:
                # 多行输入
                text = read_multiline_input("请输入对话内容（多行，输入END结束）:\n")
                if text is None:
                    continue
                return text
                
            elif choice in ['3', 'FILE']:
                # 文件输入
                filename = input("请输入文件路径: ").strip()
                if os.path.exists(filename):
                    try:
                        with open(filename, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        print(f"已读取文件: {filename} ({len(text)}字符)")
                        return text
                    except Exception as e:
                        print(f"读取文件失败: {str(e)}")
                else:
                    print("文件不存在")
                    
            elif choice in ['4', 'EXIT', 'QUIT']:
                return None
                
            else:
                print("无效选择，请重新输入")
                
        except (KeyboardInterrupt, EOFError):
            return None


def main():
    """
    主函数
    """
    print("正在加载ModelScope Qwen-1_8B-Chat模型...")
    
    try:
        # 设置ModelScope模型路径
        model_path = "/Users/shhaofu/.cache/modelscope/hub/models/qwen/Qwen-1_8B-Chat"
        
        print(f"模型路径: {model_path}")
        
        # 检测设备
        if torch.backends.mps.is_available():
            device = "mps"
            print("检测到MPS设备 (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = "cuda"
            print("检测到CUDA设备")
        else:
            device = "cpu"
            print("使用CPU设备")
        
        # 加载分词器和模型
        print("加载分词器和模型...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32
        ).eval()
        
        print(f"模型加载完成，使用设备: {device}")
        print("=" * 60)
        
        # 设置提示词
        instruction = """你现在是一个医疗对话要素抽取专家。
请针对下面对话内容抽取出药品名称、药物类别、医疗检查、医疗操作、现病史、辅助检查、诊断结果和医疗建议等内容，并且以json格式返回。

"""
        
        # 提供示例
        print("示例对话:")
        print("医生: 您好，宝宝4岁？咳嗽发热喉咙痛有几天了？")
        print("患者: 今天第三天")
        print("医生: 都服用什么药物？只是退烧药吗？目前体温多少？")
        print("患者: 今天喉咙才疼的")
        print("患者: 喝了个抗病毒药和消炎药")
        print("患者: 体温38.5")
        print("=" * 60)
        
        # 简单的测试
        print("测试模型...")
        test_text = "医生: 宝宝咳嗽发热3天，体温38.5度"
        response = predict_with_modelscope(model, tokenizer, instruction, test_text)
        print(f"测试结果: {response}")
        print("=" * 60)
        
        # 开始交互
        while True:
            try:
                text = safe_input_handler()
                if text is None:
                    print("程序已退出")
                    break
                
                if not text:
                    print("输入文本为空，请重新输入")
                    continue
                
                print(f"\n输入文本长度: {len(text)}字符")
                
                # 进行对话要素抽取
                print("正在处理...")
                response = predict_with_modelscope(model, tokenizer, instruction, text)
                
                print("\n对话要素抽取结果为：")
                print(response)
                print("=" * 60)
                
                # 询问是否继续
                continue_choice = input("\n是否继续？(y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', '是', '']:
                    break
                    
            except KeyboardInterrupt:
                print("\n程序已退出")
                break
            except Exception as e:
                print(f"发生错误: {str(e)}")
                continue
                
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        print("请检查以下几点:")
        print("1. 模型路径是否正确: /Users/shhaofu/.cache/modelscope/hub/models/qwen/Qwen-1_8B-Chat")
        print("2. ModelScope是否已安装: pip install modelscope")
        print("3. 是否有足够的内存加载模型")
        print("4. 网络连接是否正常")


if __name__ == '__main__':
    main()