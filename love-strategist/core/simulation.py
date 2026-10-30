import json
from openai import OpenAI

TURNING_POINTS = {
    "exclusivity_talk": "对方问：'我们到底算什么？' 你必须回应。",
    "first_conflict": "你们发生了第一次真实分歧，气氛僵住了。",
    "cooling_signal": "对方连续两次已读不回或回复极短，你需要决定下一步。",
    "life_change": "一方的生活状态发生重大变化（换城市/换工作/家庭压力）。",
}

TARGET_AGENT_PROMPT = """你扮演目标对象，基于以下画像进行回应。
当前状态扰动：{perturbation}
场景：{scene}
对方（用户）刚刚说了：{user_line}
只输出你作为目标对象的下一句回复，不要解释。"""

USER_AGENT_PROMPT = """你扮演用户，基于以下风格进行回应。
你的风格：{user_style}
场景：{scene}
对方（目标）刚刚说了：{target_line}
只输出你的下一句回复，不要解释。"""

def _chat(prompt, CFG):
    return OpenAI(base_url=CFG["llm"]["base_url"], api_key=CFG["llm"]["api_key"]).chat.completions.create(
        model=CFG["llm"]["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    ).choices[0].message.content.strip()

def simulate_turning_point(scene_key, chat, persona, CFG):
    scene = TURNING_POINTS.get(scene_key, scene_key)
    n = CFG["simulation"]["n_runs"]
    max_turns = CFG["simulation"]["max_turns"]
    outcomes = []

    user_style = "温和但容易急于确认关系，会在对方沉默时追加消息"
    perturbations = ["平静", "疲惫", "有外部压力", "心情好"]

    for i in range(n):
        perturb = perturbations[i % len(perturbations)]
        # 用户先开口（从聊天中提取最后一句用户话，或默认开场）
        user_lines = [m.text for m in chat if m.speaker == "user"]
        user_line = user_lines[-1] if user_lines else "我想和你聊聊我们的事。"

        history = []
        for turn in range(max_turns):
            target_line = _chat(
                TARGET_AGENT_PROMPT.format(
                    perturbation=perturb, scene=scene, user_line=user_line
                ), CFG
            )
            history.append(("target", target_line))
            user_line = _chat(
                USER_AGENT_PROMPT.format(
                    user_style=user_style, scene=scene, target_line=target_line
                ), CFG
            )
            history.append(("user", user_line))

        # 判定方向：最后一条目标回复的态度
        verdict_prompt = f"""根据以下模拟对话，判断目标对象最终的态度方向。
只输出 JSON：{{"direction": "clarity | repair | cooling | ambiguous", "one_line": "..."}}

{chr(10).join(f'[{s}] {t}' for s, t in history)}"""
        verdict_raw = _chat(verdict_prompt, CFG)
        try:
            v = json.loads(verdict_raw)
        except:
            v = {"direction": "ambiguous", "one_line": verdict_raw[:80]}
        v["perturbation"] = perturb
        outcomes.append(v)

    from collections import Counter
    dist = Counter(o["direction"] for o in outcomes)
    sensitive = _sensitivity(outcomes)

    lines = [f"场景：{scene}", f"模拟次数：{n}", "方向分布："]
    for k, v in dist.most_common():
        lines.append(f"  {k}: {v}/{n}")
    if sensitive:
        lines.append(f"\n⚠ 敏感点：当目标处于「{sensitive}」状态时，结论会翻转。这是真正的脆弱点，不是“结局已定”。")

    return {"distribution": "\n".join(lines), "raw": outcomes}

def _sensitivity(outcomes):
    by_perturb = {}
    for o in outcomes:
        by_perturb.setdefault(o["perturbation"], []).append(o["direction"])
    dominant = None
    for p, dirs in by_perturb.items():
        top = max(set(dirs), key=dirs.count)
        if dominant is None:
            dominant = top
        elif top != dominant and len(set(dirs)) > 1:
            return p
    return None