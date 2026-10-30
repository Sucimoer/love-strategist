import json
from openai import OpenAI

def _llm(CFG):
    return OpenAI(base_url=CFG["llm"]["base_url"], api_key=CFG["llm"]["api_key"])

PROMPT = """你是关系信号分析器。分析以下聊天记录中【目标对象】发出的信号，按 Sternberg 三维拆解。
只把目标对象主动说出的话当作证据，用户自己的话仅作上下文，不计入目标信号。

输出严格 JSON：
{
  "intimacy": {"score": 0-100, "evidence": ["目标原话1", "目标原话2"]},
  "passion": {"score": 0-100, "evidence": [...]},
  "commitment": {"score": 0-100, "evidence": [...]},
  "risk_flags": ["你以为在升温但可能是礼貌回避的信号，用目标原话引用"],
  "summary": "一段话，一针见血，引用证据"
}

聊天记录：
{chat}
"""

def analyze_signals(chat, CFG):
    chat_text = "\n".join(f"[{m.speaker}] {m.text}" for m in chat)
    resp = _llm(CFG).chat.completions.create(
        model=CFG["llm"]["model"],
        messages=[{"role": "user", "content": PROMPT.format(chat=chat_text)}],
        temperature=0.3,
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"summary": raw, "intimacy": {}, "passion": {}, "commitment": {}, "risk_flags": []}
    data["summary"] = data.get("summary", raw)
    return data