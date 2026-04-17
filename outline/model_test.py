#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27 11:41
# @File  : model_test.py
# @Author: johnson
# @Desc  : 加载 TRL GRPO 训练后的模型进行大纲生成推理测试

import os
import json
import time
import dotenv
import prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

dotenv.load_dotenv()

# ---------- 配置 ----------
NAME = os.getenv("ART_NAME", "web-search-outline")
BASE_MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_CACHE_DIR = "/data/xxx/model" # 请替换为你的模型缓存目录

# 指定训练好的模型路径,替换为你自己的路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_PATH = os.path.join(SCRIPT_DIR, "output/outline01-20251220-223525")

TEST_TOPICS = [
    "企业数字化转型战略规划",
    "大模型技术在教育领域的创新应用",
    "智能制造与工业4.0转型路径",
    "金融科技创新与风险管理",
    "元宇宙技术发展与商业应用",
    "碳中和目标下的绿色能源发展",
    "区块链技术在供应链中的应用",
    "远程办公模式下的团队管理",
    "短视频营销策略与内容创作",
    "云原生架构设计与实践",
    "自动驾驶技术商业化进程",
    "虚拟数字人技术与应用场景",
    "农业科技创新与智慧农业",
]


def load_model():
    """加载训练好的模型"""
    if os.path.exists(TRAINED_MODEL_PATH):
        print(f"[Model] 加载训练好的模型: {TRAINED_MODEL_PATH}")
        model_path = TRAINED_MODEL_PATH
    else:
        raise FileNotFoundError(f"模型路径不存在: {TRAINED_MODEL_PATH}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        cache_dir=MODEL_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.bfloat16
    )
    
    # 打印模型设备信息
    if hasattr(model, 'hf_device_map'):
        print(f"[GPU] 模型设备分配: {model.hf_device_map}")
    else:
        print(f"[GPU] 模型设备: {next(model.parameters()).device}")
    
    return model, tokenizer


def generate_outline(model, tokenizer, topic: str) -> str:
    """根据主题生成大纲"""
    # 构建消息
    messages = [
        {"role": "system", "content": prompt.ROLLOUT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt.ROLLOUT_USER_PROMPT.format(topic=topic)}
    ]
    
    # 应用聊天模板
    if hasattr(tokenizer, 'apply_chat_template'):
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        input_text = f"System: {prompt.ROLLOUT_SYSTEM_PROMPT}\n\nUser: {prompt.ROLLOUT_USER_PROMPT.format(topic=topic)}\n\nAssistant:"
    
    # tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码（只取生成的部分）
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    outline = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return outline


def run_test(model, tokenizer, topic: str, index: int):
    """运行单个测试用例"""
    print(f"\n{'='*70}")
    print(f"测试用例 {index}/{len(TEST_TOPICS)}: {topic}")
    print('='*70)
    
    outline = generate_outline(model, tokenizer, topic)
    
    print("\n生成的大纲:")
    print("-"*70)
    print(outline[:500] + "..." if len(outline) > 500 else outline)
    print("-"*70)
    
    return outline


def save_results_to_jsonl(results: list, output_path: str):
    """将结果保存为 JSONL 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[Save] 结果已保存到: {output_path}")


def main():
    print("[Main] 开始加载模型...")
    model, tokenizer = load_model()
    
    print(f"\n[Test] 开始测试 {len(TEST_TOPICS)} 个主题\n")
    
    results = []
    for i, topic in enumerate(TEST_TOPICS, 1):
        outline = run_test(model, tokenizer, topic, i)
        results.append({
            "id": i,
            "topic": topic,
            "outline": outline,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # 保存结果为 JSONL 文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "outline.jsonl")
    save_results_to_jsonl(results, output_path)
    
    print(f"\n{'='*70}")
    print(f"测试完成！共测试 {len(TEST_TOPICS)} 个主题")
    print(f"结果已保存到: {output_path}")
    print('='*70)


if __name__ == "__main__":
    main()
