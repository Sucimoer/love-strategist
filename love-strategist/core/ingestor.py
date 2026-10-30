import re
from dataclasses import dataclass

@dataclass
class Message:
    speaker: str      # "user" | "target"
    text: str

def parse_chat(raw: str) -> list[Message]:
    """
    支持常见格式：
      我：xxx
      对方：xxx
      你：xxx
      Ta：xxx
    也支持纯文本，按行交替视为 user/target（首行视为 target，因为通常是对方先发）。
    """
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    msgs = []
    user_labels = {"我", "你", "user", "me", "自己"}
    target_labels = {"对方", "ta", "他", "她", "target"}

    for line in lines:
        m = re.match(r"^([^：:]{1,6})[：:]\s*(.+)$", line)
        if m:
            label, text = m.group(1).strip(), m.group(2).strip()
            if label.lower() in user_labels or label in user_labels:
                msgs.append(Message("user", text))
            elif label.lower() in target_labels or label in target_labels:
                msgs.append(Message("target", text))
            else:
                msgs.append(Message("target", text))  # 未识别标签，默认对方
        else:
            # 无标签，交替分配
            spk = "target" if len(msgs) % 2 == 0 else "user"
            msgs.append(Message(spk, line))
    return msgs