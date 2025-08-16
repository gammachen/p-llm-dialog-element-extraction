from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch


def predict_openai(model, instruction, text):
    """
    利用Qwen模型进行对话要素抽取实战
    Args:
        model: Qwen模型
        instruction: 提示词内容
        text: 对话内容

    Returns:

    """
    # 为CPU环境设置适当的参数
    if not torch.cuda.is_available():
        # 在CPU上禁用缓存机制以避免past_key_values相关错误
        response, history = model.chat(tokenizer, instruction + text, history=None, use_cache=False)
    else:
        response, history = model.chat(tokenizer, instruction + text, history=None)
    return response


if __name__ == '__main__':
    # 实例化Qwen-1.8B模型以及Tokenizer（使用modelscope）
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True)
    # 检查是否有可用的GPU，如果没有则使用CPU
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", 
                                                 device_map=device, 
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.float32 if device == "cpu" else None).eval()
    print('开始对对话内容进行要素抽取，输入CTRL+C，则退出')
    while True:
        # 输入提示词内容和对话内容
        instruction = input("输入的提示词内容为：")
        text = input("输入的对话内容为：")
        # 进行对话要素抽取
        response = predict_openai(model, instruction, text)
        print("对话要素抽取结果为：")
        print(response)