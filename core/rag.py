import os
import json
import jieba
from rank_bm25 import BM25Okapi

DATA_DIR = "data"
BOOKS_FILE = os.path.join(DATA_DIR, "books.json")

def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(BOOKS_FILE):
        with open(BOOKS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def load_books():
    _ensure_dir()
    try:
        with open(BOOKS_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except Exception as e:
        print(f"读取知识库失败，可能文件损坏: {e}")
        return [] # 读取失败返回空列表，防止接口500报错

def save_books(books):
    _ensure_dir()
    with open(BOOKS_FILE, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def add_book(title, content):
    books = load_books()
    book_id = str(int(os.times().elapsed * 1000))
    chunks = chunk_text(content)
    
    book_obj = {
        "id": book_id,
        "title": title,
        "content": content,
        "chunks": chunks
    }
    books.append(book_obj)
    save_books(books)
    return book_obj

def delete_book(book_id):
    books = load_books()
    books = [b for b in books if b.get("id") != book_id]
    save_books(books)

def retrieve_relevant_chunks(query, max_chars=4000):
    books = load_books()
    if not books:
        return ""
        
    all_chunks = []
    for b in books:
        for c in b["chunks"]:
            all_chunks.append({"title": b["title"], "text": c})
            
    if not all_chunks:
        return ""
        
    tokenized_corpus = [list(jieba.cut(c["text"])) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut(query))
    doc_scores = bm25.get_scores(tokenized_query)
    scored_chunks = sorted(zip(all_chunks, doc_scores), key=lambda x: x[1], reverse=True)
    
    result_text = ""
    for chunk_info, score in scored_chunks:
        if score < 0.1:
            break
        block = f"【参考自《{chunk_info['title']}》】\n{chunk_info['text']}\n"
        if len(result_text) + len(block) > max_chars:
            break
        result_text += block + "\n"
        
    return result_text