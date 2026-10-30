import json, os
from datetime import datetime

PATH = "data/feedback.json"

class FeedbackStore:
    def __init__(self):
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(PATH):
            json.dump([], open(PATH, "w", encoding="utf-8"), ensure_ascii=False)

    def record(self, advice, user_reply=None, target_reply=None):
        records = json.load(open(PATH, encoding="utf-8"))
        records.append({
            "time": datetime.now().isoformat(),
            "advice_snapshot": advice.get("output", "")[:600],
            "user_reply": user_reply,
            "target_reply": target_reply,
            "outcome": "unknown",
        })
        json.dump(records, open(PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return records[-1]

    def recent_voice(self, n=5):
        """取出用户最近自己写的回复，用于下一次 prompt 的语气注入。"""
        records = json.load(open(PATH, encoding="utf-8"))
        return [r["user_reply"] for r in records if r.get("user_reply")][-n:]