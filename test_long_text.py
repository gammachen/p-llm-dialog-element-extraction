#!/usr/bin/env python3
"""
测试长文本输入功能
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_qwen import read_multiline_input, read_single_line_input

def test_input_handlers():
    """测试输入处理函数"""
    print("测试输入处理函数...")
    
    # 测试单行输入
    print("\n1. 测试单行输入（带长度限制）:")
    text = read_single_line_input("请输入测试文本（限制50字符）: ", max_length=50)
    print(f"输入结果: {text}")
    print(f"长度: {len(text) if text else 0}字符")
    
    # 测试多行输入
    print("\n2. 测试多行输入:")
    text = read_multiline_input("请输入多行测试文本（输入END结束）:\n")
    print(f"输入结果: {text}")
    print(f"长度: {len(text) if text else 0}字符")

if __name__ == "__main__":
    test_input_handlers()