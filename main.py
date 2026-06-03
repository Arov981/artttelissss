import os, json, asyncio, uuid, sqlite3, re, time, math, random, hashlib
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
from collections import defaultdict

import aiohttp, aiosqlite
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ============ CONFIGURATION ============
NIM_KEYS = [os.getenv(f"NIM_KEY_{i}") for i in range(1, 16) if os.getenv(f"NIM_KEY_{i}")]
OR_KEY = os.getenv("OR_KEY", "")
EL_KEY = os.getenv("EL_KEY", "")

NIM_BASE = "https://integrate.api.nvidia.com/v1"
OR_BASE = "https://openrouter.ai/api/v1"
DB_PATH = "artelis.db"

# ============ 15-KEY ROTATION ============
_key_idx = 0
_key_lock = asyncio.Lock()

async def next_key():
    global _key_idx
    async with _key_lock:
        if not NIM_KEYS:
            return OR_KEY if OR_KEY else None
        key = NIM_KEYS[_key_idx % len(NIM_KEYS)]
        _key_idx += 1
        return key

# ============ DATABASE ============
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT DEFAULT 'Learner',
                native_lang TEXT DEFAULT 'hi',
                target_lang TEXT DEFAULT 'en',
                level TEXT DEFAULT 'beginner',
                xp INTEGER DEFAULT 0,
                streak INTEGER DEFAULT 0,
                streak_freezes_used INTEGER DEFAULT 0,
                total_lessons INTEGER DEFAULT 0,
                last_practice DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                daily_goal INTEGER DEFAULT 15,
                notifications_enabled INTEGER DEFAULT 1,
                voice_enabled INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sections (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                description TEXT,
                order_num INTEGER,
                is_generated INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS units (
                id TEXT PRIMARY KEY,
                section_id TEXT,
                title TEXT,
                description TEXT,
                order_num INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (section_id) REFERENCES sections(id)
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                unit_id TEXT,
                title TEXT,
                description TEXT,
                order_num INTEGER,
                question_count INTEGER DEFAULT 17,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (unit_id) REFERENCES units(id)
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                topic_id TEXT,
                content TEXT,
                correct_answer TEXT,
                type TEXT,
                difficulty INTEGER DEFAULT 1,
                options TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_progress (
                user_id TEXT,
                question_id TEXT,
                correct INTEGER DEFAULT 0,
                attempts INTEGER DEFAULT 0,
                last_practiced TIMESTAMP,
                proficiency REAL DEFAULT 0.5,
                PRIMARY KEY (user_id, question_id),
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (question_id) REFERENCES questions(id)
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_mistakes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question_id TEXT,
                mistake_count INTEGER DEFAULT 1,
                last_mistake TIMESTAMP,
                correct_after INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (question_id) REFERENCES questions(id)
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS completed_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                topic_id TEXT,
                score INTEGER,
                xp_earned INTEGER,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            )""")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )""")
        await db.commit()
        
        cursor = await db.execute("SELECT COUNT(*) FROM users")
        if (await cursor.fetchone())[0] == 0:
            default_user_id = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO users (id, name, native_lang, target_lang, level, last_practice) VALUES (?, ?, ?, ?, ?, ?)",
                (default_user_id, "Learner", "hi", "en", "beginner", date.today().isoformat())
            )
            await db.execute("INSERT INTO user_settings (user_id) VALUES (?)", (default_user_id,))
            await db.commit()

# ============ LIFESPAN FOR PYTHON 3.6 ============
@app.on_event("startup")
async def startup_event():
    await init_db()

@app.on_event("shutdown")
async def shutdown_event():
    pass

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============ AI CALL ============
async def call_ai(prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
    for _ in range(3):
        key = await next_key()
        if not key:
            continue
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{OR_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": "openrouter/auto",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    timeout=aiohttp.ClientTimeout(total=90)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    elif resp.status == 429:
                        continue
        except:
            continue
    return ""

# ============ EXPLAIN MISTAKE ============
async def explain_mistake(user_id: str, question: str, user_answer: str, correct_answer: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT target_lang, native_lang FROM users WHERE id=?", (user_id,))
        user = await cursor.fetchone()
        if not user:
            return f"The correct answer is '{correct_answer}'. Keep practicing!"
        target_lang, native_lang = user
        
        prompt = f"""You are a friendly language tutor. Your student's native language is {native_lang}. They are learning {target_lang}.

The question was: {question}
The user answered: {user_answer}
The correct answer is: {correct_answer}

Explain why the answer was wrong in {native_lang}. Be encouraging. Provide the correct answer and a simple rule to remember. Keep it short (2-3 sentences)."""
    
    return await call_ai(prompt, max_tokens=300, temperature=0.5) or f"The correct answer is '{correct_answer}'. Keep practicing!"

# ============ BIRDBRAIN ANALYSIS ============
async def birdbrain_analysis(user_id: str, topic_id: str) -> Dict:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT level FROM users WHERE id=?", (user_id,))
        user = await cursor.fetchone()
        level = user[0] if user else "beginner"
        
        cursor = await db.execute("""
            SELECT up.correct, up.attempts, q.difficulty 
            FROM user_progress up 
            JOIN questions q ON up.question_id = q.id 
            WHERE up.user_id = ? AND q.topic_id = ? 
            ORDER BY up.last_practiced DESC LIMIT 10
        """, (user_id, topic_id))
        recent = await cursor.fetchall()
        
        if not recent:
            return {"recommended_difficulty": 1, "should_review_previous": False, "next_action": "new_lesson", "proficiency": 0.5}
        
        correct_count = sum(1 for r in recent if r[0] == 1)
        total_attempts = sum(r[1] for r in recent)
        avg_difficulty = sum(r[2] for r in recent) / len(recent) if recent else 1
        proficiency = (correct_count + 0.5) / (total_attempts + 1) if total_attempts > 0 else 0.5
        
        if proficiency > 0.8:
            new_difficulty = min(5, avg_difficulty + 1)
            next_action = "new_lesson"
        elif proficiency < 0.4:
            new_difficulty = max(1, avg_difficulty - 1)
            next_action = "review"
        else:
            new_difficulty = avg_difficulty
            next_action = "new_lesson"
        
        return {
            "recommended_difficulty": int(new_difficulty),
            "should_review_previous": proficiency < 0.5,
            "next_action": next_action,
            "proficiency": proficiency
        }

# ============ GENERATE NEXT SECTION ============
async def generate_next_section(user_id: str) -> Dict:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT target_lang, native_lang, level FROM users WHERE id=?", (user_id,))
        user = await cursor.fetchone()
        if not user:
            return {"error": "User not found"}
        target_lang, native_lang, level = user
        
        cursor = await db.execute("""
            SELECT t.title FROM completed_lessons cl
            JOIN topics t ON cl.topic_id = t.id
            WHERE cl.user_id = ? ORDER BY cl.completed_at DESC LIMIT 10
        """, (user_id,))
        completed = await cursor.fetchall()
        completed_topics = ", ".join([c[0] for c in completed]) if completed else "None yet"
        
        prompt = f"""Generate a COMPLETE section for learning {target_lang} for a {level} level learner whose native language is {native_lang}. The user has completed: {completed_topics}.

Return ONLY valid JSON with: title, description, units (4 units, each with title and 4 topics). Each topic has 17 questions covering vocabulary, grammar, reading, listening, speaking."""
        
        ai_response = await call_ai(prompt, max_tokens=4000, temperature=0.8)
        if not ai_response:
            return {"error": "Failed to generate section"}
        
        import re
        json_match = re.search(r'\{[\s\S]*\}', ai_response)
        if not json_match:
            return {"error": "Failed to parse AI response"}
        
        try:
            section_data = json.loads(json_match.group())
        except:
            return {"error": "Failed to parse AI response"}
        
        section_id = str(uuid.uuid4())
        cursor = await db.execute("SELECT COUNT(*) FROM sections WHERE user_id=?", (user_id,))
        order_num = (await cursor.fetchone())[0] + 1
        
        await db.execute(
            "INSERT INTO sections (id, user_id, title, description, order_num, is_generated) VALUES (?, ?, ?, ?, ?, ?)",
            (section_id, user_id, section_data.get("title", "New Section"), section_data.get("description", ""), order_num, 1)
        )
        
        unit_order = 1
        for unit in section_data.get("units", []):
            unit_id = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO units (id, section_id, title, description, order_num) VALUES (?, ?, ?, ?, ?)",
                (unit_id, section_id, unit.get("title", "New Unit"), unit.get("description", ""), unit_order)
            )
            topic_order = 1
            for topic in unit.get("topics", []):
                topic_id = str(uuid.uuid4())
                await db.execute(
                    "INSERT INTO topics (id, unit_id, title, description, order_num, question_count) VALUES (?, ?, ?, ?, ?, ?)",
                    (topic_id, unit_id, topic.get("title", "New Topic"), topic.get("description", ""), topic_order, 17)
                )
                for q in topic.get("questions", [])[:17]:
                    question_id = str(uuid.uuid4())
                    await db.execute(
                        "INSERT INTO questions (id, topic_id, content, correct_answer, type, difficulty, options) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (question_id, topic_id, q.get("content", ""), q.get("correct_answer", ""),
                         q.get("type", "multiple_choice"), q.get("difficulty", 1),
                         json.dumps(q.get("options", [])) if q.get("options") else None)
                    )
                topic_order += 1
            unit_order += 1
        
        await db.commit()
        return {"success": True, "section_id": section_id, "title": section_data.get("title")}

# ============ API ROUTES ============
@app.get("/api/user/{user_id}")
async def get_user(user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE id=?", (user_id,))
        user = await cursor.fetchone()
        if not user:
            user_id = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO users (id, name, native_lang, target_lang, level, last_practice) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, "Learner", "hi", "en", "beginner", date.today().isoformat())
            )
            await db.execute("INSERT INTO user_settings (user_id) VALUES (?)", (user_id,))
            await db.commit()
            cursor = await db.execute("SELECT * FROM users WHERE id=?", (user_id,))
            user = await cursor.fetchone()
        
        cursor = await db.execute("SELECT COUNT(*) FROM completed_lessons WHERE user_id=?", (user_id,))
        lessons_completed = (await cursor.fetchone())[0]
        cursor = await db.execute("SELECT COUNT(*) FROM questions")
        total_questions = (await cursor.fetchone())[0]
        
        return {
            "id": user["id"], "name": user["name"], "native_lang": user["native_lang"],
            "target_lang": user["target_lang"], "level": user["level"], "xp": user["xp"],
            "streak": user["streak"], "streak_freezes_used": user["streak_freezes_used"],
            "total_lessons": user["total_lessons"], "lessons_completed": lessons_completed,
            "progress_percent": round(lessons_completed / max(1, total_questions) * 100, 1)
        }

@app.put("/api/user/{user_id}")
async def update_user(user_id: str, request: Request):
    data = await request.json()
    async with aiosqlite.connect(DB_PATH) as db:
        for key, value in data.items():
            if key in ["name", "native_lang", "target_lang", "level"]:
                await db.execute(f"UPDATE users SET {key}=? WHERE id=?", (value, user_id))
            elif key == "daily_goal":
                await db.execute("UPDATE user_settings SET daily_goal=? WHERE user_id=?", (value, user_id))
        await db.commit()
    return {"success": True}

@app.get("/api/user/{user_id}/sections")
async def get_sections(user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM sections WHERE user_id=? ORDER BY order_num", (user_id,))
        sections = await cursor.fetchall()
        result = []
        for section in sections:
            cursor = await db.execute("SELECT * FROM units WHERE section_id=? ORDER BY order_num", (section["id"],))
            units = await cursor.fetchall()
            section_units = []
            for unit in units:
                cursor = await db.execute("SELECT * FROM topics WHERE unit_id=? ORDER BY order_num", (unit["id"],))
                topics = await cursor.fetchall()
                unit_completed = True
                for topic in topics:
                    cursor = await db.execute(
                        "SELECT COUNT(*) FROM completed_lessons WHERE user_id=? AND topic_id=?", (user_id, topic["id"])
                    )
                    completed = (await cursor.fetchone())[0] > 0
                    topic = dict(topic)
                    topic["completed"] = completed
                    if not completed:
                        unit_completed = False
                    section_units.append({"id": unit["id"], "title": unit["title"], "description": unit["description"],
                                          "order_num": unit["order_num"], "topics": [dict(t) for t in topics], "completed": unit_completed})
            result.append({"id": section["id"], "title": section["title"], "description": section["description"],
                           "order_num": section["order_num"], "is_generated": section["is_generated"],
                           "units": section_units, "completed": all(u.get("completed", False) for u in section_units)})
        return {"sections": result}

@app.post("/api/user/{user_id}/generate_next_section")
async def generate_next_section_endpoint(user_id: str):
    return await generate_next_section(user_id)

@app.get("/api/lesson/topic/{topic_id}")
async def get_topic_questions(topic_id: str, user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT title FROM topics WHERE id=?", (topic_id,))
        topic = await cursor.fetchone()
        if not topic:
            raise HTTPException(404, "Topic not found")
        cursor = await db.execute("""
            SELECT q.id, q.content, q.correct_answer, q.type, q.difficulty, q.options,
                   up.correct, up.attempts, up.proficiency
            FROM questions q
            LEFT JOIN user_progress up ON q.id = up.question_id AND up.user_id = ?
            WHERE q.topic_id = ?
            ORDER BY up.proficiency ASC, q.difficulty ASC
        """, (user_id, topic_id))
        questions = []
        for q in await cursor.fetchall():
            questions.append({
                "id": q[0], "content": q[1], "correct_answer": q[2], "type": q[3],
                "difficulty": q[4], "options": json.loads(q[5]) if q[5] else [],
                "correct": q[6] == 1 if q[6] is not None else None,
                "attempts": q[7] or 0, "proficiency": q[8] or 0.5
            })
        return {"topic_id": topic_id, "title": topic[0], "questions": questions[:17], "total_questions": 17}

@app.post("/api/lesson/answer")
async def submit_answer(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    question_id = data.get("question_id")
    user_answer = data.get("answer", "").strip().lower()
    
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT correct_answer, difficulty, content FROM questions WHERE id=?", (question_id,))
        result = await cursor.fetchone()
        if not result:
            return {"error": "Question not found"}
        correct_answer, difficulty, question_content = result
        is_correct = user_answer == correct_answer.lower()
        
        cursor = await db.execute("SELECT correct, attempts FROM user_progress WHERE user_id=? AND question_id=?", (user_id, question_id))
        existing = await cursor.fetchone()
        if existing:
            new_correct = existing[0] + (1 if is_correct else 0)
            new_attempts = existing[1] + 1
            new_proficiency = (new_correct + 0.5) / (new_attempts + 1)
            await db.execute("UPDATE user_progress SET correct=?, attempts=?, proficiency=?, last_practiced=? WHERE user_id=? AND question_id=?",
                             (new_correct, new_attempts, new_proficiency, datetime.now().isoformat(), user_id, question_id))
        else:
            new_proficiency = 0.5 + (0.1 if is_correct else -0.1)
            await db.execute("INSERT INTO user_progress (user_id, question_id, correct, attempts, proficiency, last_practiced) VALUES (?, ?, ?, ?, ?, ?)",
                             (user_id, question_id, 1 if is_correct else 0, 1, new_proficiency, datetime.now().isoformat()))
        
        xp_earned = (10 + difficulty * 2) if is_correct else 0
        if is_correct:
            await db.execute("UPDATE users SET xp = xp + ? WHERE id=?", (xp_earned, user_id))
        
        explanation = None
        if not is_correct:
            explanation = await explain_mistake(user_id, question_content, user_answer, correct_answer)
        
        await db.commit()
        return {"correct": is_correct, "xp_earned": xp_earned, "explanation": explanation, "correct_answer": correct_answer}

@app.post("/api/lesson/complete")
async def complete_topic(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    topic_id = data.get("topic_id")
    score = data.get("score", 0)
    xp_earned = data.get("xp_earned", 0)
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO completed_lessons (user_id, topic_id, score, xp_earned) VALUES (?, ?, ?, ?)",
                         (user_id, topic_id, score, xp_earned))
        await db.execute("UPDATE users SET total_lessons = total_lessons + 1, xp = xp + ? WHERE id=?", (xp_earned, user_id))
        
        cursor = await db.execute("SELECT unit_id FROM topics WHERE id=?", (topic_id,))
        unit = await cursor.fetchone()
        if unit:
            cursor = await db.execute("SELECT COUNT(*) FROM topics WHERE unit_id=? AND id NOT IN (SELECT topic_id FROM completed_lessons WHERE user_id=?)",
                                      (unit[0], user_id))
            if (await cursor.fetchone())[0] == 0:
                cursor = await db.execute("SELECT section_id FROM units WHERE id=?", (unit[0],))
                section = await cursor.fetchone()
                if section:
                    cursor = await db.execute("SELECT COUNT(*) FROM units WHERE section_id=? AND id NOT IN (SELECT unit_id FROM units WHERE id IN (SELECT unit_id FROM topics WHERE id IN (SELECT topic_id FROM completed_lessons WHERE user_id=?)))",
                                              (section[0], user_id))
                    if (await cursor.fetchone())[0] == 0:
                        return {"unit_completed": True, "section_completed": True, "next_section_generating": True}
        await db.commit()
    return {"success": True, "xp_earned": xp_earned}

@app.post("/api/streak/freeze")
async def use_streak_freeze(user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE users SET streak_freezes_used = streak_freezes_used + 1 WHERE id=?", (user_id,))
        await db.commit()
    return {"success": True}

@app.websocket("/ws/roleplay")
async def roleplay_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "message", "message": "Role play mode coming soon!"})
    except WebSocketDisconnect:
        pass

# ============ STATIC FILES ============
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/manifest.json")
async def manifest():
    return FileResponse("static/manifest.json")

@app.get("/sw.js")
async def service_worker():
    return FileResponse("static/sw.js")
