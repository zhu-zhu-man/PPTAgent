#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27 11:41
# @File  : model_test.py
# @Author: johnson
# @Desc  : 加载 TRL GSPO 训练后的模型进行内容生成推理测试

import os
import json
import time
import re
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from prompt import CONTENT_SYSTEM_PROMPT, CONTENT_USER_PROMPT

# 加载 backend/.env 配置文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
dotenv.load_dotenv(os.path.join(BACKEND_DIR, '.env'))

# ---------- 配置 ----------
BASE_MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_CACHE_DIR = "/data/xxx/model" # 请替换为你的模型缓存目录

# 指定训练好的模型路径,替换为你自己的路径
TRAINED_MODEL_PATH = os.path.join(SCRIPT_DIR, "output/content01-20251221-093206")


# =========================
# 测试大纲数据（3个测试用例）
# =========================
TEST_OUTLINES = [
    {
        "topic": "人工智能在医疗诊断中的应用",
        "outline": """# 人工智能在医疗诊断中的应用

## 技术基础与发展现状
### 核心算法与模型架构
- 构建深度卷积神经网络进行医学影像识别
- 应用Transformer架构处理电子病历文本
- 设计多模态融合模型整合多源医疗数据
- 优化模型推理速度以满足实时诊断需求

### 数据采集与处理流程
- 建立标准化医学影像数据采集规范
- 实施数据脱敏和隐私保护机制
- 设计高效的数据标注质量控制体系

## 临床应用场景分析
### 医学影像辅助诊断
- 实现肺部CT影像的病灶自动检测
- 构建眼底图像糖尿病视网膜病变筛查系统
- 开发乳腺X光片智能分析工具

### 病理切片智能分析
- 应用AI识别肿瘤细胞形态特征
- 自动化生成病理分级报告
- 预测肿瘤分子分型和预后

## 挑战与发展趋势
### 技术瓶颈与解决方案
- 提升小样本场景下的模型泛化能力
- 增强模型可解释性以获取医生信任
- 解决跨中心数据分布差异问题

### 未来发展方向
- 推动多中心联邦学习平台建设
- 探索大模型在医疗领域的应用潜力
- 建立AI辅助诊断的法规监管框架
"""
    },
    {
        "topic": "新能源汽车产业链分析",
        "outline": """# 新能源汽车产业链分析

## 产业链上游核心环节
### 动力电池技术演进
- 分析磷酸铁锂与三元锂电池的技术路线差异
- 评估固态电池商业化进程与挑战
- 探讨电池能量密度提升的技术路径
- 研究快充技术对电池性能的影响

### 关键原材料供应格局
- 梳理锂资源全球分布与供应链风险
- 分析钴镍等材料的价格波动因素
- 评估电池回收再利用的经济可行性

## 产业链中游制造环节
### 整车制造工艺创新
- 推广一体化压铸技术降低生产成本
- 优化电池包集成设计提升空间利用率
- 实施智能化产线提高生产效率

### 核心零部件国产化
- 加速电驱系统核心技术自主研发
- 推进车规级芯片国产替代进程
- 培育国内热管理系统供应商

## 产业链下游应用场景
### 充电基础设施建设
- 规划城市公共充电网络布局
- 推广超级充电站建设标准
- 探索换电模式的商业可行性
- 发展车网互动V2G技术

### 市场发展趋势预测
- 预测2025年新能源车渗透率
- 分析下沉市场的增长潜力
- 评估出口市场的竞争态势
"""
    },
    {
        "topic": "远程办公效率提升策略",
        "outline": """# 远程办公效率提升策略

## 工作环境与基础设施
### 家庭办公空间优化
- 设计符合人体工学的办公区域布局
- 配置专业级视频会议设备与网络环境
- 建立工作与生活的物理边界隔离
- 优化照明和通风条件提升舒适度

### 数字化工具体系搭建
- 部署企业级协同办公平台
- 集成项目管理与任务追踪系统
- 建设安全可靠的VPN接入方案

## 时间管理与自我驱动
### 高效工作节奏设计
- 实施番茄工作法提升专注力
- 规划每日固定的深度工作时段
- 设定清晰的每日目标与优先级

### 自我监督机制建立
- 采用时间追踪工具量化工作效率
- 建立每周工作复盘与改进循环
- 培养自律习惯与内在驱动力
- 设计合理的休息与恢复策略

## 团队协作与沟通策略
### 异步沟通最佳实践
- 制定清晰的书面沟通规范与模板
- 合理使用异步消息减少打扰
- 建立文档驱动的知识共享体系

### 团队凝聚力维护
- 组织定期虚拟团建增强归属感
- 实施一对一远程导师计划
- 建立跨时区协作的最佳时间窗口
- 创造非正式交流的虚拟空间
"""
    }
]


def load_model():
    """加载训练好的模型"""
    model_path = TRAINED_MODEL_PATH
    
    if os.path.exists(model_path):
        print(f"[Model] 加载训练好的模型: {model_path}")
    else:
        print(f"[Model] 模型路径不存在: {model_path}，使用基础模型: {BASE_MODEL_NAME}")
        model_path = BASE_MODEL_NAME
    
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


def generate_full_content(model, tokenizer, topic: str, outline: str) -> str:
    """根据完整大纲生成完整内容"""
    # 构建消息
    messages = [
        {"role": "system", "content": CONTENT_SYSTEM_PROMPT},
        {"role": "user", "content": CONTENT_USER_PROMPT.format(
            topic=topic,
            outline=outline
        )}
    ]
    
    # 应用聊天模板
    if hasattr(tokenizer, 'apply_chat_template'):
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        input_text = f"System: {CONTENT_SYSTEM_PROMPT}\n\nUser: {CONTENT_USER_PROMPT.format(topic=topic, outline=outline)}\n\nAssistant:"
    
    # tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 生成 - 不限制长度，使用较大的 max_new_tokens
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,  # 支持生成完整长文档
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码（只取生成的部分）
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return content.strip()


def run_test(model, tokenizer, outline_data: dict, index: int, total: int):
    """运行单个测试用例（完整大纲）"""
    topic = outline_data["topic"]
    outline = outline_data["outline"]
    
    print(f"\n{'='*70}")
    print(f"测试用例 {index}/{total}")
    print(f"主题: {topic}")
    print(f"大纲长度: {len(outline)} 字符")
    print('='*70)
    
    content = generate_full_content(model, tokenizer, topic, outline)
    
    print("\n生成的完整内容:")
    print("-"*70)
    print(content[:2000] + "..." if len(content) > 2000 else content)  # 只显示前2000字
    print("-"*70)
    print(f"生成内容长度: {len(content)} 字符")
    
    return content


def save_results_to_jsonl(results: list, output_path: str):
    """将结果保存为 JSONL 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[Save] 结果已保存到: {output_path}")


def main():
    print("[Main] 开始加载模型...")
    model, tokenizer = load_model()
    
    # 直接使用完整大纲进行测试（3个测试大纲）
    print(f"\n[Test] 共有 {len(TEST_OUTLINES)} 个测试大纲\n")
    
    results = []
    for i, outline_data in enumerate(TEST_OUTLINES, 1):
        content = run_test(model, tokenizer, outline_data, i, len(TEST_OUTLINES))
        results.append({
            "id": i,
            "topic": outline_data["topic"],
            "outline": outline_data["outline"],
            "generated_content": content,
            "content_length": len(content),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # 保存结果为 JSONL 文件
    output_path = os.path.join(SCRIPT_DIR, "content.jsonl")
    save_results_to_jsonl(results, output_path)
    
    print(f"\n{'='*70}")
    print(f"测试完成！共测试 {len(TEST_OUTLINES)} 个完整大纲")
    print(f"结果已保存到: {output_path}")
    print('='*70)


if __name__ == "__main__":
    main()
