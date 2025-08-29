import json
from pathlib import Path

# 文件路径
FILTERED_USERS_PATH = Path(__file__).parent / "filtered_users_healthcare.json"
AGENT_FILE_PATH = Path(__file__).parent / "processed/agent_20ppl.json"
OUTPUT_PATH = Path(__file__).parent / "filtered_agents_healthcare.json"

def main():
    # 读取 target user ids
    with open(FILTERED_USERS_PATH, 'r', encoding='utf-8') as f:
        filtered_users = json.load(f)
    target_ids = set(filtered_users.keys())

    # 读取所有 agent
    with open(AGENT_FILE_PATH, 'r', encoding='utf-8') as f:
        agents = json.load(f)

    # 筛选出目标 agent
    filtered_agents = [agent for agent in agents if agent.get('id') in target_ids]

    # 输出到新文件
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_agents, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(filtered_agents)} agents to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
