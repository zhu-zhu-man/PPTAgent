#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/12/20
# @File  : train_trl.py
# @Author: johnson
# @Desc  : 基于 TRL GSPO 训练内容生成模型（根据大纲生成详细内容）

import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from typing import List, Dict, Any
import re
import json
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import torch
import wandb
from openai import OpenAI
from prompt import CONTENT_SYSTEM_PROMPT, CONTENT_USER_PROMPT, JUDGE_PROMPT

os.environ["WANDB_MODE"] = "offline"

# 加载 backend/.env 配置文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
dotenv.load_dotenv(os.path.join(BACKEND_DIR, '.env'))

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
USE_DEEPSEEK_JUDGE = os.getenv("USE_DEEPSEEK_JUDGE", "true").lower() == "true"

# ---------- 配置（使用 CONTENT_* 环境变量）----------
NAME = os.getenv("CONTENT_NAME", "content01")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("CONTENT_PROJECT", "content-training")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", 4096))

print(f"{NAME} - {MODEL_NAME} - {PROJECT_NAME}")
print(f"训练时传入的最大序列长度: {MAX_SEQ_LEN}")

# wandb
WANDB_PROJECT = os.getenv("WANDB_PROJECT", PROJECT_NAME)
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", f"{NAME}-{time.strftime('%Y%m%d-%H%M%S')}")

# 模型缓存目录
MODEL_CACHE_DIR = "/data/xxx/model" # 请替换为你的模型缓存目录

# 全局变量
tokenizer = None
model = None


def load_model_and_tokenizer():
    """加载模型和tokenizer"""
    global tokenizer, model
    
    if tokenizer is None:
        print(f"[Model] 加载 tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    if model is None:
        print(f"[Model] 加载模型: {MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            device_map="auto",
            cache_dir=MODEL_CACHE_DIR,
            torch_dtype=torch.bfloat16
        )
        
        if hasattr(model, 'hf_device_map'):
            print(f"[GPU] 模型设备分配: {model.hf_device_map}")
        else:
            print(f"[GPU] 模型设备: {next(model.parameters()).device}")
    
    return model, tokenizer


# ============================================================
# 从大纲中加载训练数据
# ============================================================
def load_training_data_from_jsonl(jsonl_path: str) -> List[Dict[str, str]]:
    """
    从 outline.jsonl 加载完整大纲作为训练数据
    返回格式: [{"topic": ..., "outline": ...}, ...]
    """
    all_outlines = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            topic = data.get("topic", "")
            outline = data.get("outline", "")
            
            # 清理 outline（去掉 markdown 代码块标记和末尾说明文字）
            outline = re.sub(r'^```markdown\s*', '', outline)
            outline = re.sub(r'\s*```[\s\S]*$', '', outline)  # 去掉 ``` 及其后面的所有内容
            outline = outline.strip()
            
            if topic and outline:
                all_outlines.append({
                    "topic": topic,
                    "outline": outline
                })
    
    return all_outlines


# ============================================================
# 奖励函数 1：内容质量规则奖励（针对完整文档）
# ============================================================
def content_quality_reward(completions: List[str], **kwargs) -> List[float]:
    """
    【奖励函数 1】完整文档内容质量评分（归一化到 [0, 1]）
    
    评估标准：
    - 结构完整性：保留大纲结构（0.25分）
    - 内容丰富度：足够长且有实质内容（0.25分）
    - 包含数字/数据（0.2分）
    - 无模板句（0.15分）
    - 专业性指标（0.15分）
    """
    rewards = []
    
    for content in completions:
        try:
            score = 0.0
            content = content.strip()
            
            # 1. 结构完整性（0.25分）- 检查是否保留了大纲结构
            has_h1 = bool(re.search(r'^# ', content, re.MULTILINE))
            has_h2 = bool(re.search(r'^## ', content, re.MULTILINE))
            has_h3 = bool(re.search(r'^### ', content, re.MULTILINE))
            has_bullets = bool(re.search(r'^- ', content, re.MULTILINE))
            
            structure_score = 0
            if has_h1:
                structure_score += 0.05
            if has_h2:
                structure_score += 0.08
            if has_h3:
                structure_score += 0.07
            if has_bullets:
                structure_score += 0.05
            score += min(0.25, structure_score)
            
            # 2. 内容丰富度（0.25分）- 长度不做上限限制
            length = len(content)
            if length >= 2000:
                score += 0.25  # 内容充实
            elif length >= 1500:
                score += 0.2
            elif length >= 1000:
                score += 0.15
            elif length >= 500:
                score += 0.1
            elif length >= 200:
                score += 0.05
            # 太短不给分
            
            # 3. 包含数字/数据（0.2分）
            data_patterns = re.findall(r'\d+[%％]|\d+\.\d+|\d{4}年|\d+个?月|\d+亿|\d+万', content)
            if len(data_patterns) >= 5:
                score += 0.2
            elif len(data_patterns) >= 3:
                score += 0.15
            elif len(data_patterns) >= 1:
                score += 0.1
            
            # 4. 无模板句（0.15分）
            template_patterns = [
                r'^Detailed content',
                r'^详细内容',
                r'^这是关于',
                r'^本节介绍',
                r'请根据以上',
                r'以下是.*内容',
            ]
            is_template = any(re.search(p, content, re.IGNORECASE) for p in template_patterns)
            if not is_template:
                score += 0.15
            
            # 5. 专业性指标（0.15分）- 包含专业词汇
            professional_terms = [
                '技术', '系统', '平台', '方案', '策略', '模式', '机制', '体系',
                '优化', '提升', '实现', '部署', '构建', '整合', '分析', '评估',
                '创新', '架构', '框架', '流程', '效率', '成本', '收益', '风险'
            ]
            term_count = sum(1 for term in professional_terms if term in content)
            if term_count >= 10:
                score += 0.15
            elif term_count >= 6:
                score += 0.1
            elif term_count >= 3:
                score += 0.05
            
            rewards.append(min(1.0, score))
            
        except Exception:
            rewards.append(0.0)
    
    return rewards


# ============================================================
# 奖励函数 2：DeepSeek API 评估奖励
# ============================================================
def _call_deepseek_judge(content: str) -> float:
    """调用 DeepSeek API 评估内容质量，返回归一化分数 [0, 1]"""
    try:
        base_url = DEEPSEEK_BASE_URL.replace("/chat/completions", "")
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=base_url)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(content=content)}],
            temperature=0.1,
            max_tokens=10,
        )
        score_text = response.choices[0].message.content.strip()
        match = re.search(r"(\d+(?:\.\d+)?)", score_text)
        if match:
            score = float(match.group(1))
            return min(1.0, max(0.0, score / 10.0))
        return 0.5
    except Exception as e:
        logging.warning(f"DeepSeek API 调用失败: {e}")
        return 0.5


def deepseek_judge_reward(completions: List[str], **kwargs) -> List[float]:
    """
    【奖励函数 2】DeepSeek API 评估奖励（归一化到 [0, 1]）
    """
    if not USE_DEEPSEEK_JUDGE or not DEEPSEEK_API_KEY:
        return [0.5] * len(completions)
    
    rewards = []
    for completion in completions:
        score = _call_deepseek_judge(completion)
        rewards.append(score)
    return rewards


# ============================================================
# 准备训练数据
# ============================================================
def prepare_gspo_dataset(outlines: List[Dict[str, str]]) -> Dataset:
    """
    准备 GSPO 训练数据集（基于完整大纲）
    """
    prompts = []
    
    for item in outlines:
        messages = [
            {"role": "system", "content": CONTENT_SYSTEM_PROMPT},
            {"role": "user", "content": CONTENT_USER_PROMPT.format(
                topic=item["topic"],
                outline=item["outline"]
            )}
        ]
        
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = f"System: {CONTENT_SYSTEM_PROMPT}\n\nUser: {CONTENT_USER_PROMPT.format(**item)}\n\nAssistant:"
        
        prompts.append(formatted_prompt)
    
    dataset = Dataset.from_dict({"prompt": prompts})
    return dataset


# ============================================================
# 训练主程序
# ============================================================
def main():
    global model, tokenizer
    
    # 加载模型
    print("[Main] 开始加载模型...")
    model, tokenizer = load_model_and_tokenizer()
    
    # 初始化 wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY if WANDB_ENTITY else None,
        name=WANDB_RUN_NAME,
        config={
            "project": PROJECT_NAME,
            "name": NAME,
            "base_model": MODEL_NAME,
            "max_seq_len": MAX_SEQ_LEN,
        },
        settings=wandb.Settings(start_method="thread"),
    )

    # 加载训练数据（从 outline.jsonl，加载完整大纲）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outline_jsonl_path = os.path.join(script_dir, '../outline/outline.jsonl')
    
    if not os.path.exists(outline_jsonl_path):
        raise FileNotFoundError(f"训练数据不存在: {outline_jsonl_path}")
    
    training_outlines = load_training_data_from_jsonl(outline_jsonl_path)
    print(f"[Data] 从 outline.jsonl 加载了 {len(training_outlines)} 个完整大纲")

    # 准备数据集
    train_dataset = prepare_gspo_dataset(training_outlines)
    print(f"[Data] 准备了 {len(train_dataset)} 条训练数据")

    # 输出目录
    output_dir = f"./output/{NAME}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # GSPO 配置 - 不限制输出长度
    gspo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=50,
        max_completion_length=MAX_SEQ_LEN,  # 输出不限制，使用最大序列长度
        num_generations=4,
        temperature=0.7,
        seed=42,
        bf16=True,
        report_to="wandb",
        remove_unused_columns=False,
        num_iterations=1,
        beta=0.04,
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type="dr_grpo",
    )

    # 创建训练器
    reward_funcs = [content_quality_reward]
    if USE_DEEPSEEK_JUDGE and DEEPSEEK_API_KEY:
        reward_funcs.append(deepseek_judge_reward)
        print(f"[Reward] 启用 DeepSeek API 评估奖励")
    else:
        print(f"[Reward] 仅使用规则奖励（USE_DEEPSEEK_JUDGE={USE_DEEPSEEK_JUDGE}）")
    
    trainer = GRPOTrainer(
        model=model,
        args=gspo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
    )

    print(f"[Train] 开始 GSPO 训练...")
    print(f"[Train] 输出目录: {output_dir}")
    print(f"[Train] 奖励函数: {[f.__name__ for f in reward_funcs]}")
    
    # 训练
    trainer.train()
    
    # 保存模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"[Train] 模型已保存到 {output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()

