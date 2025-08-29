import json
from scipy.stats import wasserstein_distance

def normalize_reason_vector(reason_dict, all_keys):
    vec = [reason_dict.get(k, 0) for k in all_keys]
    total = sum(vec)
    return [v / total if total > 0 else 0 for v in vec]

def filter_users_by_reason_shift(data, opinion_threshold=1.0, reason_similarity_threshold=2.0):
    all_reason_keys = set()
    for user in data.values():
        for rs in user["reasons"].values():
            all_reason_keys.update(rs.keys())
    all_reason_keys = sorted(all_reason_keys)

    selected_users = {}

    for user_id, user in data.items():
        regular_opinion = user["opinions"].get("3.4")
        regular_reason = user["reasons"].get("3.5")

        if regular_opinion is None or regular_reason is None:
            continue

        regular_reason_vec = normalize_reason_vector(regular_reason, all_reason_keys)

        for qid in ["3.6", "3.7", "3.8", "3.9"]:
            cf_opinion = user["opinions"].get(qid)
            cf_reason = user["reasons"].get(qid)

            if cf_opinion is None or cf_reason is None:
                continue

            opinion_diff = abs(cf_opinion - regular_opinion)
            cf_reason_vec = normalize_reason_vector(cf_reason, all_reason_keys)
            reason_w_dist = wasserstein_distance(regular_reason_vec, cf_reason_vec)
            reason_similarity = 1 / (1 + reason_w_dist)

            if opinion_diff < opinion_threshold and reason_similarity < reason_similarity_threshold:
                selected_users[user_id] = user
                break  # 保留整个人，只要符合任一 counterfactual 条件即可

    return selected_users

def main():
    # Step 1: 加载原始数据
    with open("./processed/survey_20ppl_healthcare_reactions.json", "r") as f:
        data = json.load(f)

    # Step 2: 筛选符合条件的用户
    filtered_users = filter_users_by_reason_shift(data)

    # Step 3: 写入新 JSON 文件
    with open("filtered_users_healthcare.json", "w") as f:
        json.dump(filtered_users, f, indent=2)

    print(f"筛选完成，共保留 {len(filtered_users)} 个用户，已保存至 filtered_users.json")

if __name__ == "__main__":
    main()
