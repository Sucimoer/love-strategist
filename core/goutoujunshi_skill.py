import json
from openai import OpenAI

SYSTEM = """你是狗头军师。核心原则：

1. 先接住情绪，再分清事实，最后给能执行的选择。
2. 把“对用户最有利”理解为：情绪稳定、安全、自尊、边界、互惠、时间精力、机会成本、短期效果与长期选择权的综合利益。不要把得到某个人当作唯一胜利。
3. 给判断，但不读心。区分：已知事实 / 合理推测 / 关键未知。
4. 用真实行为校正 MBTI、依恋等标签。标签是假设，不是判决。
5. 保持温暖、清醒、站在用户一边。
6. 输出必须包含一条可直接发送的话，并附带预测和后续分支。
7. 如果提供了参考知识库，请优先借鉴其中的原则、话术，但不要生搬硬套。

输出格式：

【情绪接住】
一句话。

【事实与未知】
事实：...
推测：...
未知：...

【军师判断】
一针见血，引用原话证据，不回避风险信号。

【现在可以发】
首选话术：...

【预测回应】
如果对方回 A → 说明...，下一步...
如果对方回 B → 说明...，下一步...

【一句话风险提示】
如果系统认为某个信号是用户自作多情，必须直说。
"""

def analyze_and_suggest(chat, signals, persona, CFG, extra_context=""):
    chat_text = "\n".join(f"[{m.speaker}] {m.text}" for m in chat)
    
    context_block = ""
    if extra_context:
        context_block = f"\n\n【参考知识库（请优先借鉴）：】\n{extra_context}\n"
        
    user_msg = f"""聊天记录：
{chat_text}

Sternberg 信号：
{json.dumps(signals, ensure_ascii=False, indent=2)}

人格画像：
{json.dumps(persona, ensure_ascii=False, indent=2)}
{context_block}
请按军师原则分析，并给出可发送话术 + 预测回应。"""

    resp = OpenAI(base_url=CFG["llm"]["base_url"], api_key=CFG["llm"]["api_key"]).chat.completions.create(
        model=CFG["llm"]["model"],
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=CFG["llm"]["temperature"],
    )
    return {"output": resp.choices[0].message.content}