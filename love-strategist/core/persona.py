import json
from openai import OpenAI

PROMPT = """从以下聊天记录中分析【目标对象】的沟通人格。
只基于行为证据，不推测长相、生辰、面相。
输出 JSON：
{
  "mbti_guess": "如 INFP / 不确定",
  "confidence": "high | medium | low",
  "communication_style": ["特征1", "特征2"],
  "what_they_care_about": ["从聊天中看出的在意点"],
  "defense_triggers": ["可能触发防御的话题或语气"],
  "summary": "一段话画像，带原话证据"
}

聊天记录：
{chat}
"""

def build_persona(chat, CFG):
    chat_text = "\n".join(f"[{m.speaker}] {m.text}" for m in chat)
    resp = OpenAI(base_url=CFG["llm"]["base_url"], api_key=CFG["llm"]["api_key"]).chat.completions.create(
        model=CFG["llm"]["model"],
        messages=[{"role": "user", "content": PROMPT.format(chat=chat_text)}],
        temperature=0.3,
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": raw, "mbti_guess": "不确定", "confidence": "low"}