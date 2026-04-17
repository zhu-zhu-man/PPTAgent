
import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from typing import List
import re
import json
import dotenv
import prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import torch
import wandb
from openai import OpenAI

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

# ---------- 配置（使用 OUTLINE_* 环境变量）----------
NAME = os.getenv("OUTLINE_NAME", "outline01")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("OUTLINE_PROJECT", "outline-training")
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
# 奖励函数 1：规则奖励（结构 + 格式）
# ============================================================
def rule_based_reward(completions: List[str], **kwargs) -> List[float]:
    """
    【奖励函数 1】基于规则的大纲评分（归一化到 [0, 1]）
    
    评估标准（与 ROLLOUT_SYSTEM_PROMPT 对齐）：
    - 结构：# 1个, ## 5个, 每个##下3-4个###, 每个###下3-5个要点
    - 格式：要点以动词开头、≤18字、不以句号结尾
    - 禁止：问句标题、禁止段落（引言/结语/总结/目录/参考）
    """
    rewards = []
    for md in completions:
        try:
            score = 0.0
            
            # 1. 一级标题检查（0.1分）
            h1 = re.findall(r"(?m)^# [^\n]+$", md)
            if len(h1) == 1:
                score += 0.1
            
            # 2. 二级标题检查（0.2分）
            h2_matches = list(re.finditer(r"(?m)^## [^\n]+$", md))
            if len(h2_matches) == 5:
                score += 0.2
            elif len(h2_matches) > 0:
                score += 0.1 * min(len(h2_matches), 5) / 5
            
            # 3. 三级标题检查（0.2分）
            h3_count = len(re.findall(r"(?m)^### [^\n]+$", md))
            if 15 <= h3_count <= 20:
                score += 0.2
            elif h3_count > 0:
                score += 0.1 * min(h3_count, 15) / 15
            
            # 4. 要点检查（0.3分）
            bullets = re.findall(r"(?m)^- (.+)$", md)
            if len(bullets) >= 45:
                # 检查动词开头
                verb_prefixes = ("分析", "设计", "实现", "优化", "评估", "构建", "制定", 
                                "应用", "建立", "开发", "部署", "测试", "验证", "规划",
                                "整合", "配置", "监控", "管理", "提升", "改进", "创建",
                                "定义", "执行", "推进", "确保", "支持", "提供", "解决")
                verb_count = sum(1 for b in bullets if any(b.strip().startswith(v) for v in verb_prefixes))
                verb_ratio = verb_count / len(bullets) if bullets else 0
                
                # 检查长度和句号
                valid_format = sum(1 for b in bullets if len(b.strip()) <= 18 and not b.strip().endswith(("。", ".", "．")))
                format_ratio = valid_format / len(bullets) if bullets else 0
                
                score += 0.15 * verb_ratio + 0.15 * format_ratio
            elif len(bullets) > 0:
                score += 0.1 * len(bullets) / 45
            
            # 5. 禁止内容检查（0.2分，惩罚项）
            penalty = 0.0
            # 问句标题
            if re.search(r"^#+ .*[？?]", md, re.MULTILINE):
                penalty += 0.1
            # 禁止段落
            if re.search(r"(引言|结语|总结|目录|参考|说明|免责声明)", md):
                penalty += 0.1
            
            score = max(0.0, score + 0.2 - penalty)
            rewards.append(min(1.0, score))
            
        except Exception:
            rewards.append(0.0)
    
    return rewards


# ============================================================
# 奖励函数 2：DeepSeek API 评估奖励
# ============================================================
JUDGE_PROMPT = """请评估以下大纲的质量，给出0-10的评分。

评分标准：
1. 内容质量（0-4分）：逻辑递进、专业深度、实操导向
2. 结构完整（0-3分）：层级清晰、覆盖全面
3. 格式规范（0-3分）：要点简洁、动词开头、无冗余

大纲内容：
{outline}

请仅输出一个0-10之间的数字评分，不要输出其他内容。"""


def _call_deepseek_judge(outline: str) -> float:
    """调用 DeepSeek API 评估大纲质量，返回归一化分数 [0, 1]"""
    try:
        # 从 DEEPSEEK_BASE_URL 中提取 base_url（去掉 /chat/completions 后缀）
        base_url = DEEPSEEK_BASE_URL.replace("/chat/completions", "")
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=base_url)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(outline=outline)}],
            temperature=0.1,
            max_tokens=10,
        )
        score_text = response.choices[0].message.content.strip()
        # 提取数字
        match = re.search(r"(\d+(?:\.\d+)?)", score_text)
        if match:
            score = float(match.group(1))
            return min(1.0, max(0.0, score / 10.0))  # 归一化到 [0, 1]
        return 0.5  # 解析失败返回中间值
    except Exception as e:
        logging.warning(f"DeepSeek API 调用失败: {e}")
        return 0.5  # API 失败返回中间值


def deepseek_judge_reward(completions: List[str], **kwargs) -> List[float]:
    """
    【奖励函数 2】DeepSeek API 评估奖励（归一化到 [0, 1]）
    
    调用 DeepSeek API 对大纲进行综合评估，重点关注内容质量。
    """
    if not USE_DEEPSEEK_JUDGE or not DEEPSEEK_API_KEY:
        # 未启用或无API Key，返回中间值
        return [0.5] * len(completions)
    
    rewards = []
    for completion in completions:
        score = _call_deepseek_judge(completion)
        rewards.append(score)
    return rewards


# ============================================================
# 准备训练数据
# ============================================================
def prepare_gspo_dataset(topics: List[str]) -> Dataset:
    """
    准备 GSPO 训练数据集
    GSPO 需要的数据集格式：包含 'prompt' 列
    """
    prompts = []
    
    for topic in topics:
        messages = [
            {"role": "system", "content": prompt.ROLLOUT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt.ROLLOUT_USER_PROMPT.format(topic=topic)}
        ]
        
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = f"System: {prompt.ROLLOUT_SYSTEM_PROMPT}\n\nUser: {prompt.ROLLOUT_USER_PROMPT.format(topic=topic)}\n\nAssistant:"
        
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

    # 加载训练主题
    script_dir = os.path.dirname(os.path.abspath(__file__))
    topic_json_path = os.path.join(script_dir, 'topic.json')
    assert os.path.exists(topic_json_path), f"训练数据不存在: {topic_json_path}"
    
    with open(topic_json_path, 'r', encoding='utf-8') as f:
        topic_data = json.load(f)
    topics = topic_data["topics"]
    print(f"[Data] 加载了 {len(topics)} 个训练主题")

    # 准备数据集
    train_dataset = prepare_gspo_dataset(topics)
    print(f"[Data] 准备了 {len(train_dataset)} 条训练数据")

    # 输出目录
    output_dir = f"./output/{NAME}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # GSPO 配置（Group Sequence Policy Optimization）
    # GSPO 是 GRPO 的变体，通过序列级别的重要性采样提高训练稳定性
    gspo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,  # GSPO 推荐使用更小的学习率
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=50,
        max_completion_length=MAX_SEQ_LEN // 2,
        num_generations=4,  # 每个 prompt 生成4个样本进行比较
        temperature=0.7,
        seed=42,
        bf16=True,
        report_to="wandb",
        remove_unused_columns=False,
        num_iterations=1,
        beta=0.04,  # GSPO 推荐的 KL 惩罚系数
        # GSPO 核心参数：序列级别的重要性采样（区别于 GRPO 的 token 级别）
        importance_sampling_level="sequence",
        mask_truncated_completions = False,
        loss_type = "dr_grpo",
    )

    # 创建训练器
    # reward_funcs: GSPO 会用这些函数评估模型生成的内容
    # 两个奖励函数都已归一化到 [0, 1]
    reward_funcs = [rule_based_reward]  # 奖励函数1：规则奖励
    if USE_DEEPSEEK_JUDGE and DEEPSEEK_API_KEY:
        reward_funcs.append(deepseek_judge_reward)  # 奖励函数2：DeepSeek API 评估
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
