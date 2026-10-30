import os
import requests
import yaml
from flask import Flask, request, jsonify, render_template
from core.ingestor import parse_chat
from core.sternberg import analyze_signals
from core.persona import build_persona
from core.goutoujunshi_skill import analyze_and_suggest
from core.simulation import simulate_turning_point, TURNING_POINTS
from core.feedback import FeedbackStore
# 导入刚写的 rag 模块
from core.rag import load_books, add_book, delete_book, retrieve_relevant_chunks

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

@app.route("/")
def index():
    scenes = [{"key": k, "desc": v} for k, v in TURNING_POINTS.items()]
    return render_template("index.html", scenes=scenes)

# ===== 知识库接口 =====
@app.route("/api/knowledge/list", methods=["GET"])
def api_kb_list():
    books = load_books()
    return jsonify([{"id": b["id"], "title": b["title"], "length": len(b["content"])} for b in books])

@app.route("/api/knowledge/add", methods=["POST"])
def api_kb_add():
    data = request.get_json(force=True)
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    if not title or not content:
        return jsonify({"error": "标题和内容不能为空"}), 400
    book = add_book(title, content)
    return jsonify({"ok": True, "id": book["id"]})

@app.route("/api/knowledge/delete", methods=["POST"])
def api_kb_delete():
    data = request.get_json(force=True)
    book_id = data.get("id")
    delete_book(book_id)
    return jsonify({"ok": True})

@app.route("/api/knowledge/search_book", methods=["POST"])
def api_kb_search_book():
    """输入书名获取简介"""
    data = request.get_json(force=True)
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "书名不能为空"}), 400
        
    try:
        url = f"https://openlibrary.org/search.json?q={requests.utils.quote(title)}&limit=1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
        if not docs:
            return jsonify({"error": "未找到该书"}), 404
            
        first = docs[0]
        book_title = first.get("title", title)
        
        desc = f"《{book_title}》是一部著作。"
        if "key" in first:
            work_url = f"https://openlibrary.org{first['key']}.json"
            w_resp = requests.get(work_url, timeout=10)
            if w_resp.ok:
                w_data = w_resp.json()
                if "description" in w_data:
                    d = w_data["description"]
                    desc = d if isinstance(d, str) else d.get("value", desc)
                    
        return jsonify({"ok": True, "title": book_title, "description": desc})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== 核心分析接口 =====
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True)
    raw = data.get("chat", "").strip()
    user_key = data.get("api_key", "").strip() # 【新增】获取前端传来的 Key
    
    if not raw:
        return jsonify({"error": "聊天内容为空"}), 400

    # 【新增】如果前端传了 Key，就覆盖当前请求的 CFG
    local_cfg = CFG.copy()
    if user_key:
        local_cfg["llm"] = local_cfg["llm"].copy()
        local_cfg["llm"]["api_key"] = user_key

    try:
        chat = parse_chat(raw)
        signals = analyze_signals(chat, local_cfg) # 换成 local_cfg
        persona = build_persona(chat, local_cfg)   # 换成 local_cfg
        
        chat_text = "\n".join(f"[{m.speaker}] {m.text}" for m in chat)
        from core.rag import retrieve_relevant_chunks
        kb_context = retrieve_relevant_chunks(chat_text)
        
        advice = analyze_and_suggest(chat, signals, persona, local_cfg, extra_context=kb_context) # 换成 local_cfg

        fb = FeedbackStore()
        record = fb.record(advice, user_reply=None, target_reply=None)

        return jsonify({
            "signals": signals,
            "persona": persona,
            "advice": advice.get("output", ""),
            "record_id": record.get("time"),
        })
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
# (其余的 api/simulate 和 api/feedback 接口保持原样，不用改)

if __name__ == "__main__":
    app.run()