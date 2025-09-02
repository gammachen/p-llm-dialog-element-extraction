import torch
import json
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def predict_with_qwen(model, tokenizer, instruction, text, max_length=2048):
    """
    使用QWen模型进行对话要素抽取，修复past_key_values问题
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
        
        # 获取模型设备
        device = next(model.parameters()).device
        
        # 对输入进行编码
        inputs = tokenizer(
            input_text, 
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        
        # 移动输入到模型所在设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        
        # 解码输出
        input_length = inputs['input_ids'].size(1)
        new_tokens = outputs[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        return f"预测时发生错误: {str(e)}"


def predict_with_local_model(model, tokenizer, instruction, text, max_length=2048):
    """
    使用本地QWen模型进行预测
    Args:
        model: 本地模型
        tokenizer: 本地分词器
        instruction: 提示词
        text: 输入文本
        max_length: 最大长度
    Returns:
        预测结果
    """
    try:
        if not text or not text.strip():
            return "错误：输入文本不能为空"
        
        input_text = instruction + text
        
        # 使用本地模型的chat方法
        response, history = model.chat(tokenizer, input_text, history=None)
        return response
        
    except Exception as e:
        return f"本地模型预测错误: {str(e)}"


def safe_input_handler():
    """
    安全的输入处理器
    """
    print("\n" + "=" * 60)
    print("对话要素抽取工具 - QWen2-0.5B 医疗专用")
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
    print("正在加载QWen模型...")
    
    try:
        # 使用QWen2-0.5B模型，更小且兼容性好
        model_name = "Qwen/Qwen2-0.5B-Instruct"
        
        # 检测设备，优先使用CPU避免MPS兼容性问题
        device = "cpu"  # 强制使用CPU避免MPS错误
        print("使用CPU设备（避免MPS兼容性问题）")
        
        print(f"正在加载模型: {model_name}")
        
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # 明确使用CPU
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).eval()
        
        print(f"QWen2-0.5B-Instruct模型加载完成")
        print("模型特点：0.5B参数，轻量级，适合CPU推理")
        print("=" * 60)
        
        # 设置医疗专用提示词
        instruction = """你是一名专业的医疗对话分析专家。请仔细分析以下医患对话，并提取出关键医疗信息。

需要提取的信息包括：
1. 药品名称（具体药物）
2. 药物类别（如抗病毒药、消炎药等）
3. 医疗检查（如血常规、肺部听诊等）
4. 医疗操作（如就诊、查体等）
5. 现病史（主要症状和时间）
6. 辅助检查结果
7. 诊断结果
8. 医疗建议

请以JSON格式返回，确保信息准确完整。

对话内容："""
        
        # 提供示例
        print("示例对话:")
        print("医生: 您好，宝宝4岁，咳嗽发热喉咙痛3天")
        print("患者: 体温38.5度，医生开了抗病毒药和消炎药")
        print("医生: 建议做血常规检查，肺部听诊正常")
        print("=" * 60)
        
        # 简单的测试
        print("测试模型响应...")
        test_text = "医生: 宝宝咳嗽发热3天，体温38.5度，建议服用抗病毒药物"
        response = predict_with_qwen(model, tokenizer, instruction, test_text)
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
                print("正在分析对话内容...")
                response = predict_with_qwen(model, tokenizer, instruction, text)
                
                print("\n对话要素抽取结果：")
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
        print(f"QWen2-0.5B模型加载失败: {str(e)}")
        print("尝试使用本地ModelScope模型...")
        
        # 回退到本地ModelScope模型
        try:
            from modelscope import AutoTokenizer as ModelScopeTokenizer, AutoModelForCausalLM as ModelScopeModel
            
            model_path = "/Users/shhaofu/.cache/modelscope/hub/models/qwen/Qwen-1_8B-Chat"
            print(f"正在加载本地ModelScope模型: {model_path}")
            
            tokenizer = ModelScopeTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            model = ModelScopeModel.from_pretrained(
                model_path,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32
            ).eval()
            
            print("本地ModelScope模型加载完成")
            
            # 使用本地模型预测
            while True:
                text = safe_input_handler()
                if text is None:
                    break
                
                try:
                    response = predict_with_local_model(model, tokenizer, instruction, text)
                    print(f"\n本地模型结果: {response}")
                except Exception as model_error:
                    print(f"模型预测错误: {str(model_error)}")
                    
        except Exception as fallback_error:
            print(f"所有模型加载失败: {str(fallback_error)}")
            print("请检查以下几点:")
            print("1. 网络连接是否正常")
            print("2. transformers和modelscope是否已安装")
            print("3. 是否有足够的内存和存储空间")


if __name__ == '__main__':
    main()