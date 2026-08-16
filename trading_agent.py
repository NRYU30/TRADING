"""
Fase 4 (Capstone): Trading Agent Lengkap
=============================================
Menggabungkan SEMUA yang sudah dipelajari:
- Fase 1: agent loop, tool use, memory, planning
- Fase 2: LangChain, memory persisten (SQLite)
- Fase 3: data pasar real (yfinance), indikator teknikal

PENTING - INI BUKAN AUTO-TRADING BOT:
Agent ini TIDAK terhubung ke broker sungguhan. Tool `record_trade` cuma
MENCATAT keputusan ke jurnal simulasi (paper trading) di database lokal,
dan SELALU minta konfirmasi manual dari kamu dulu sebelum tercatat.
Ini prinsip human-in-the-loop: agent boleh MENGANALISIS dan MENGUSULKAN
secara mandiri, tapi TIDAK BOLEH bertindak sendiri tanpa persetujuanmu.

Kalau nanti kamu mau connect ke broker sungguhan, ganti isi record_trade
dengan pemanggilan API broker asli - tapi WAJIB tetap pertahankan langkah
konfirmasi manual ini, atau setidaknya batas risiko (max loss) yang ketat.

Cara pakai:
1. pip install yfinance pandas -qU langchain "langchain[google-genai]"
2. python3 final_trading_agent.py
3. Ngobrol bebas, contoh: "Analisis BTC-USD dong, gimana kondisinya?"
   lalu: "Oke, catat rekomendasi itu ke jurnal simulasi"
"""

import sqlite3
from datetime import datetime

import pandas as pd
import yfinance as yf
from langchain.agents import create_agent

DB_FILE = "trading_journal.db"


# ── SETUP DATABASE (memory jangka panjang, tetap ada walau restart) ─────
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            amount REAL NOT NULL,
            price REAL NOT NULL,
            reasoning TEXT,
            timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


# ── TOOL 1: harga terkini (dari Fase 3) ──────────────────────────────────
def get_price(ticker: str) -> str:
    """Ambil harga penutupan terkini untuk sebuah aset. Contoh: BTC-USD, ETH-USD, BBCA.JK."""
    data = yf.Ticker(ticker).history(period="5d")
    if data.empty:
        return f"Ticker '{ticker}' tidak ditemukan."
    price = data["Close"].iloc[-1]
    return f"Harga {ticker} terkini: {price:.2f}"


# ── TOOL 2: sinyal tren (dari Fase 3) ────────────────────────────────────
def get_trend_signal(ticker: str) -> str:
    """Analisis tren teknikal (SMA 20 vs SMA 50) untuk sebuah aset."""
    data = yf.Ticker(ticker).history(period="3mo")
    if data.empty:
        return f"Ticker '{ticker}' tidak ditemukan."
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    latest = data.iloc[-1]
    if pd.isna(latest["SMA_50"]):
        return "Data belum cukup untuk analisis tren (butuh minimal 50 hari)."
    trend = "Uptrend" if latest["SMA_20"] > latest["SMA_50"] else "Downtrend"
    return (
        f"{ticker}: SMA_20={latest['SMA_20']:.2f}, "
        f"SMA_50={latest['SMA_50']:.2f} -> {trend}"
    )


# ── TOOL 3: lihat portofolio simulasi (memory, dari Fase 2) ─────────────
def get_portfolio() -> str:
    """Lihat ringkasan posisi yang sedang 'dipegang' di jurnal trading simulasi."""
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("SELECT ticker, action, amount FROM trades").fetchall()
    conn.close()
    if not rows:
        return "Belum ada posisi tercatat. Portofolio kosong."

    net = {}
    for ticker, action, amount in rows:
        net[ticker] = net.get(ticker, 0) + (amount if action == "BUY" else -amount)
    holdings = {t: q for t, q in net.items() if abs(q) > 1e-9}

    if not holdings:
        return "Semua posisi sudah tertutup. Portofolio kosong."
    return "Posisi saat ini: " + ", ".join(f"{t}: {q:g}" for t, q in holdings.items())


# ── TOOL 4: catat trade - WAJIB persetujuan manusia dulu ────────────────
def record_trade(ticker: str, action: str, amount: float, reasoning: str) -> str:
    """
    Catat sebuah keputusan trading (BUY atau SELL) ke jurnal simulasi.
    Ini akan MEMINTA KONFIRMASI MANUAL dari user sebelum benar-benar
    dicatat. Ini SIMULASI/paper trading, BUKAN eksekusi order sungguhan.
    """
    price_info = get_price(ticker)

    print("\n" + "!" * 60)
    print("AGENT MENGUSULKAN AKSI - PERLU PERSETUJUAN KAMU")
    print("!" * 60)
    print(f"  Ticker  : {ticker}")
    print(f"  Aksi    : {action.upper()}")
    print(f"  Jumlah  : {amount}")
    print(f"  {price_info}")
    print(f"  Alasan  : {reasoning}")
    print("!" * 60)

    confirm = input("Setujui aksi ini? Ketik 'y' untuk ya, lainnya untuk tolak: ")
    if confirm.strip().lower() != "y":
        return "DITOLAK oleh user. Trade TIDAK dicatat ke jurnal."

    data = yf.Ticker(ticker).history(period="5d")
    current_price = float(data["Close"].iloc[-1]) if not data.empty else 0.0

    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO trades (ticker, action, amount, price, reasoning, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            ticker,
            action.upper(),
            amount,
            current_price,
            reasoning,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return (
        f"DISETUJUI. Trade dicatat ke jurnal (SIMULASI): "
        f"{action.upper()} {amount} {ticker} @ {current_price:.2f}"
    )


# ── AGENT ─────────────────────────────────────────────────────────────
agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[get_price, get_trend_signal, get_portfolio, record_trade],
    system_prompt=(
        "Kamu adalah trading assistant untuk tujuan BELAJAR/SIMULASI, "
        "bukan produk finansial sungguhan.\n\n"
        "ATURAN WAJIB:\n"
        "1. SELALU gunakan tool get_price/get_trend_signal untuk data "
        "faktual - jangan pernah menebak harga atau tren dari asumsi.\n"
        "2. SELALU cek get_portfolio dulu sebelum menyarankan aksi baru, "
        "supaya tidak menyarankan beli aset yang sudah dipegang berlebihan.\n"
        "3. JANGAN panggil record_trade kecuali user secara eksplisit "
        "minta dicatat/dieksekusi/disimpan - kalau user cuma minta "
        "analisis, cukup kasih analisis dan rekomendasi dalam teks.\n"
        "4. Selalu jelaskan reasoning di balik saran kamu, dan ingatkan "
        "user bahwa ini simulasi belajar, bukan saran finansial "
        "profesional."
    ),
)


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


# Riwayat percakapan disimpan selama program berjalan (short-term memory,
# konsep dari Fase 1) - hilang kalau program ditutup, beda dari jurnal
# SQLite di atas yang persisten (long-term memory)
conversation = []


def chat(user_message: str):
    conversation.append({"role": "user", "content": user_message})
    before_count = len(conversation)  # batas: sebelum giliran ini, sesudah ini baru

    result = agent.invoke({"messages": conversation})

    new_messages = result["messages"][before_count:]  # HANYA pesan baru giliran ini

    conversation.clear()
    conversation.extend(result["messages"])

    for message in new_messages:
        msg_type = message.__class__.__name__
        if msg_type == "AIMessage" and message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "record_trade":  # record_trade sudah print sendiri
                    print(f"[Agent memanggil tool: {tc['name']}({tc['args']})]")
        elif msg_type == "AIMessage" and message.content:
            print(f"\nAgent: {extract_text(message.content)}")


if __name__ == "__main__":
    init_db()
    print("=== Trading Agent Simulasi ===")
    print("Ngobrol bebas soal analisis pasar. Ketik 'exit' untuk keluar.\n")
    print("Contoh: 'Analisis BTC-USD dong' lalu 'catat rekomendasi itu ke jurnal'\n")

    while True:
        user_input = input("\nKamu: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Sampai jumpa!")
            break
        if user_input:
            chat(user_input)