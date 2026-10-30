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
from core.rag import load_books, add_book, delete_book, retrieve_relevant_chunks

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

def get_local_cfg(data):
    user_key = data.get("api_key", "").strip()
    user_base_url = data.get("base_url", "").strip()
    user_model = data.get("model", "").strip()
    
    local_cfg = CFG.copy()
    local_cfg["llm"] = local_cfg["llm"].copy()
    
    if user_key:
        local_cfg["llm"]["api_key"] = user_key
    if user_base_url:
        local_cfg["llm"]["base_url"] = user_base_url
    if user_model:
        local_cfg["llm"]["model"] = user_model
        
    return local_cfg
@app.route("/")
def index():
    scenes = [{"key": k, "desc": v} for k, v in TURNING_POINTS.items()]
    return render_template("index.html", scenes=scenes)

# ... 前面的知识库接口保持不变 ...

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
@app.route("/api/knowledge/export", methods=["GET"])
def api_kb_export():
    """导出所有书籍数据"""
    books = load_books()
    return jsonify(books)

@app.route("/api/knowledge/import", methods=["POST"])
def api_kb_import():
    """覆盖导入书籍数据"""
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "备份数据格式错误，应为一个列表"}), 400
    
    try:
        # 直接覆盖写入本地 json
        from core.rag import save_books
        save_books(data)
        return jsonify({"ok": True, "count": len(data)})
    except Exception as e:
        return jsonify({"error": f"导入失败: {e}"}), 500

@app.route("/api/knowledge/search_book", methods=["POST"])
def api_kb_search_book():
    data = request.get_json(force=True)
    title = data.get("title", "").strip()
    if not title: return jsonify({"error": "书名不能为空"}), 400
    try:
        url = f"https://openlibrary.org/search.json?q={requests.utils.quote(title)}&limit=1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
        if not docs: return jsonify({"error": "未找到该书"}), 404
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
    if not raw: return jsonify({"error": "聊天内容为空"}), 400

    local_cfg = get_local_cfg(data)
    if not local_cfg["llm"]["api_key"] or local_cfg["llm"]["api_key"] == "sk-你的Key":
        return jsonify({"error": "请先配置有效的 API Key"}), 400

    try:
        chat = parse_chat(raw)
        signals = analyze_signals(chat, local_cfg)
        persona = build_persona(chat, local_cfg)
        
        chat_text = "\n".join(f"[{m.speaker}] {m.text}" for m in chat)
        kb_context = retrieve_relevant_chunks(chat_text)
        
        advice = analyze_and_suggest(chat, signals, persona, local_cfg, extra_context=kb_context)
        return jsonify({"signals": signals, "persona": persona, "advice": advice.get("output", "")})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

# ===== 转折点模拟接口 =====
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(force=True)
    raw = data.get("chat", "").strip()
    scene_input = data.get("scene", "").strip() # 可能是预设的 key，也可能是自定义文本
    opening_line = data.get("opening_line", "").strip()
    
    if not raw: return jsonify({"error": "聊天内容为空"}), 400

    local_cfg = get_local_cfg(data)
    if not local_cfg["llm"]["api_key"] or local_cfg["llm"]["api_key"] == "sk-你的Key":
        return jsonify({"error": "请先配置有效的 API Key"}), 400

    try:
        chat = parse_chat(raw)
        persona = build_persona(chat, local_cfg)
        
        # 如果传进来的是预设的 key，就翻译成中文描述；否则直接用自定义文本
        from core.simulation import TURNING_POINTS
        if scene_input in TURNING_POINTS:
            scene_input = TURNING_POINTS[scene_input]
            
        sim = simulate_turning_point(scene_input, chat, persona, local_cfg, opening_line=opening_line)
        return jsonify({"distribution": sim["distribution"]})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

if __name__ == "__main__":
    app.run()