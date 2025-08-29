import json
from pathlib import Path

AGENT_FILE_PATH = Path(__file__).parent / "processed/agent_20ppl.json"
OUTPUT_PATH = Path(__file__).parent / "filtered_agents_demo.json"

def match(agent, config):
    """判断 agent 是否符合所有筛选条件"""
    for field, cond in config.items():
        value = agent.get(field, "")
        if cond is None:
            continue
        # 支持多种筛选方式
        if isinstance(cond, (list, set, tuple)):
            # 多选之一
            if value not in cond:
                return False
        elif isinstance(cond, dict):
            # 范围筛选
            if "min" in cond or "max" in cond:
                try:
                    v = float(value)
                except Exception:
                    return False
                if "min" in cond and v < cond["min"]:
                    return False
                if "max" in cond and v > cond["max"]:
                    return False
            if "contains" in cond:
                if cond["contains"] not in value:
                    return False
        else:
            # 精确匹配
            if value != cond:
                return False
    return True

def main():
    # 读取 agent 文件
    with open(AGENT_FILE_PATH, "r", encoding="utf-8") as f:
        agents = json.load(f)

    # ====== 你可以在这里自定义筛选条件 ======
    # 示例1：筛选 income 在 $50,000-$59,999 或 $60,000-$74,999 的 agent
    # filter_config = {
    #     "income": {"$50,000-$59,999", "$60,000-$74,999"},
    # }

    # 示例2：筛选 age 在 30-40 且 income 在 $50,000-$59,999 的 agent
    # filter_config = {
    #     "age": {"min": 30, "max": 40},
    #     "income": "$50,000-$59,999",
    # }

    # 示例3：筛选 Gross rent 包含 "20.0 to 24.9 percent" 且 income 在 $100,000-$124,999
    # filter_config = {
    #     "Gross rent": {"contains": "20.0 to 24.9 percent"},
    #     "income": "$100,000-$124,999",
    # }

    # 示例4：多字段组合
    filter_config = {
        "race_ethnicity": "Black or African American",
        "financial_situation": "I struggle to afford basic needs (e.g., food, housing, healthcare)",
        "health_insurance": {"with a disability, with public health insurance", "no disability, with public health insurance"},
        "income": {"$150,000-$199,999", ">$200,000"},
        "education": "Graduate or professional degree",
        "children": "Under 6 years old",
    }

    # ====== 执行筛选 ======
    filtered_agents = [
        agent for agent in agents if match(agent["agent"], filter_config)
    ]

    # 保存结果
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered_agents, f, ensure_ascii=False, indent=2)
    print(f"筛选出 {len(filtered_agents)} 个 agent，已保存到 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
