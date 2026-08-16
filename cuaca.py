"""
Latihan Memory Persisten (Fase 2 lanjutan)
=============================================
Agent sekarang bisa "MENGINGAT" antar sesi program, walau ditutup & dibuka
lagi. Ini pola yang sama nanti dipakai trading agent buat inget riwayat
trade, posisi terbuka, dan hasil analisis sebelumnya.

Caranya: kita kasih agent 3 tool BARU:
- save_memory(key, value)   -> agent menyimpan fakta ke database SQLite
- recall_memory(key)        -> agent mengambil fakta berdasarkan key
- list_memories()           -> agent melihat semua fakta yang tersimpan

SQLite dipilih karena bawaan Python (tidak perlu install tambahan apa pun)
dan file databasenya (agent_memory.db) tetap ada walau program ditutup.

Cara test persistensinya:
1. Jalankan: python3 memory_agent.py
   -> ketik: "Tolong ingat, saya lebih suka kota yang sejuk untuk liburan"
2. TUTUP program (Ctrl+C atau tunggu selesai)
3. Jalankan LAGI: python3 memory_agent.py
   -> ketik: "Kota sejuk yang cocok buat saya apa ya?"
   -> agent akan INGAT preferensi dari sesi sebelumnya!
"""

from google import genai
from google.genai import types
import sqlite3
from datetime import datetime

client = genai.Client()

DB_FILE = "agent_memory.db"


# ── SETUP DATABASE: bikin tabel kalau belum ada ──────────────────────────
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


# ── TOOL 1: simpan fakta ke database ─────────────────────────────────────
def save_memory(key: str, value: str) -> dict:
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO memories (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return {"status": "tersimpan", "key": key, "value": value}


# ── TOOL 2: ambil balik fakta berdasarkan key ────────────────────────────
def recall_memory(key: str) -> dict:
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute(
        "SELECT value, updated_at FROM memories WHERE key = ?", (key,)
    ).fetchone()
    conn.close()
    if row is None:
        return {"found": False}
    return {"found": True, "value": row[0], "updated_at": row[1]}


# ── TOOL 3: lihat semua fakta yang pernah disimpan ───────────────────────
def list_memories() -> dict:
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT key, value FROM memories").fetchall()
    conn.close()
    return {"memories": [{"key": k, "value": v} for k, v in rows]}


# ── SKEMA TOOL ────────────────────────────────────────────────────────────
save_memory_declaration = types.FunctionDeclaration(
    name="save_memory",
    description="Simpan sebuah fakta/preferensi penting supaya bisa diingat di sesi berikutnya",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Nama singkat fakta, contoh: preferensi_cuaca",
            },
            "value": {"type": "string", "description": "Isi fakta yang mau disimpan"},
        },
        "required": ["key", "value"],
    },
)

recall_memory_declaration = types.FunctionDeclaration(
    name="recall_memory",
    description="Ambil kembali fakta yang pernah disimpan sebelumnya berdasarkan key",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Nama fakta yang mau diambil"}
        },
        "required": ["key"],
    },
)

list_memories_declaration = types.FunctionDeclaration(
    name="list_memories",
    description="Lihat semua fakta yang pernah disimpan, dipakai kalau tidak tahu key persisnya",
    parameters_json_schema={"type": "object", "properties": {}},
)

all_tools = types.Tool(
    function_declarations=[
        save_memory_declaration,
        recall_memory_declaration,
        list_memories_declaration,
    ]
)

available_functions = {
    "save_memory": save_memory,
    "recall_memory": recall_memory,
    "list_memories": list_memories,
}


def run_agent(user_message: str):
    contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]

    while True:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[all_tools],
                temperature=0,
                system_instruction=(
                    "Kamu adalah asisten dengan memori persisten lewat tool "
                    "save_memory, recall_memory, dan list_memories. "
                    "ATURAN WAJIB: sebelum menjawab pertanyaan apa pun yang "
                    "berkaitan dengan preferensi, fakta personal, atau riwayat "
                    "user, SELALU panggil recall_memory atau list_memories "
                    "dulu untuk cek apakah informasinya sudah pernah "
                    "disimpan. Jangan langsung menjawab dari asumsi umum "
                    "sebelum memastikan lewat memori terlebih dahulu."
                ),
            ),
        )

        contents.append(response.candidates[0].content)
        function_calls = response.function_calls

        if not function_calls:
            print(f"\nAgent: {response.text}")
            break

        function_response_parts = []
        for call in function_calls:
            print(f"[Agent memanggil tool: {call.name}({dict(call.args)})]")
            func = available_functions.get(call.name)
            result = func(**call.args) if func else {"error": "tool tidak dikenal"}
            function_response_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": result}
                )
            )

        contents.append(types.Content(role="user", parts=function_response_parts))


if __name__ == "__main__":
    init_db()
    print("=" * 60)
    print("Coba jalankan file ini beberapa KALI TERPISAH:")
    print('  Sesi 1: "Tolong ingat, saya suka kota sejuk untuk liburan"')
    print('  Sesi 2 (setelah program ditutup & dibuka lagi):')
    print('          "Kota sejuk yang cocok buat saya apa ya?"')
    print("=" * 60)

    user_input = input(
        "\nKetik pesan kamu (atau Enter untuk pakai contoh default): "
    ).strip()
    if not user_input:
        user_input = (
            "Tolong ingat, saya lebih suka kota yang sejuk dan "
            "tidak terlalu ramai untuk liburan"
        )

    run_agent(user_input)