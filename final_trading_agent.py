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

UPDATE - RISK MANAGEMENT (baru ditambahkan):
- Setiap trade WAJIB punya stop_loss, dan risiko dihitung otomatis
  sebagai persentase dari modal virtual. Kalau risikonya kelewat batas,
  kamu harus ketik konfirmasi lebih tegas ("OVERRIDE"), bukan cuma "y".
- Ada batas maksimal jumlah trade per hari, untuk mencegah overtrading.
- API key sekarang dibaca dari file .env (pakai python-dotenv), bukan
  cuma dari environment variable terminal - lebih praktis & konsisten
  antar sesi terminal.

Cara pakai:
1. pip install yfinance pandas python-dotenv -qU langchain "langchain[google-genai]"
2. Buat file .env di folder yang sama isinya:
   GEMINI_API_KEY=AIzaSy...ganti-dengan-key-kamu
   GOOGLE_API_KEY=AIzaSy...ganti-dengan-key-kamu-yang-sama
3. python3 final_trading_agent.py
4. Ngobrol bebas, contoh: "Analisis BTC-USD dong, gimana kondisinya?"
   lalu: "Oke, catat rekomendasi itu ke jurnal simulasi"
"""

import sqlite3
from datetime import datetime, date

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()  # baca file .env di folder yang sama, isi jadi environment variable

DB_FILE = "trading_journal.db"

# ── KONFIGURASI RISK MANAGEMENT ──────────────────────────────────────────
VIRTUAL_CAPITAL = 10_000_000  # modal simulasi, ganti sesuai kebutuhan (Rp)
MAX_RISK_PCT_PER_TRADE = 2.0  # maksimal risiko per trade: 2% dari modal
MAX_TRADES_PER_DAY = 3  # maksimal jumlah trade yang dicatat per hari


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
            stop_loss REAL,
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


# ── HELPER: hitung berapa trade yang sudah dicatat HARI INI ────────────
def count_trades_today() -> int:
    conn = sqlite3.connect(DB_FILE)
    today_str = date.today().isoformat()
    row = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (f"{today_str}%",)
    ).fetchone()
    conn.close()
    return row[0]


# ── TOOL 4: catat trade - WAJIB persetujuan manusia + risk check ───────
def record_trade(
    ticker: str, action: str, amount: float, stop_loss: float, reasoning: str
) -> str:
    """
    Catat sebuah keputusan trading (BUY atau SELL) ke jurnal simulasi.
    WAJIB sertakan stop_loss (harga batas rugi) - dipakai untuk menghitung
    risiko sebagai persentase dari modal, bukan sekadar jumlah asal.
    Ini akan MEMINTA KONFIRMASI MANUAL dari user sebelum benar-benar
    dicatat. Ini SIMULASI/paper trading, BUKAN eksekusi order sungguhan.
    """
    # ── Batas 1: jumlah trade per hari (cegah overtrading) ──────────────
    trades_today = count_trades_today()
    if trades_today >= MAX_TRADES_PER_DAY:
        return (
            f"DITOLAK OTOMATIS: sudah {trades_today} trade tercatat hari ini "
            f"(batas maksimal {MAX_TRADES_PER_DAY}/hari). Ini untuk mencegah "
            f"overtrading - coba lagi besok, atau evaluasi dulu trade yang sudah ada."
        )

    data = yf.Ticker(ticker).history(period="5d")
    if data.empty:
        return f"Ticker '{ticker}' tidak ditemukan, trade tidak bisa dicatat."
    current_price = float(data["Close"].iloc[-1])

    # ── Batas 2: hitung risiko sebagai % dari modal virtual ─────────────
    risk_per_unit = abs(current_price - stop_loss)
    risk_amount = risk_per_unit * amount
    risk_pct = (risk_amount / VIRTUAL_CAPITAL) * 100
    over_limit = risk_pct > MAX_RISK_PCT_PER_TRADE

    print("\n" + "!" * 60)
    print("AGENT MENGUSULKAN AKSI - PERLU PERSETUJUAN KAMU")
    print("!" * 60)
    print(f"  Ticker      : {ticker}")
    print(f"  Aksi        : {action.upper()}")
    print(f"  Jumlah      : {amount}")
    print(f"  Harga saat ini : {current_price:.2f}")
    print(f"  Stop loss   : {stop_loss:.2f}")
    print(f"  Estimasi risiko: {risk_pct:.2f}% dari modal virtual "
          f"(Rp{risk_amount:,.0f} dari Rp{VIRTUAL_CAPITAL:,.0f})")
    print(f"  Alasan      : {reasoning}")
    print(f"  Trade ke-{trades_today + 1} hari ini (maks {MAX_TRADES_PER_DAY})")
    print("!" * 60)

    if over_limit:
        print(
            f"\n⚠️  PERINGATAN: risiko {risk_pct:.2f}% MELEBIHI batas aman "
            f"{MAX_RISK_PCT_PER_TRADE}% per trade!"
        )
        confirm = input(
            "Risiko di atas batas. Ketik 'OVERRIDE' (huruf besar) untuk "
            "tetap lanjut, atau apa saja lainnya untuk batal: "
        )
        approved = confirm.strip() == "OVERRIDE"
    else:
        confirm = input("Setujui aksi ini? Ketik 'y' untuk ya, lainnya untuk tolak: ")
        approved = confirm.strip().lower() == "y"

    if not approved:
        return "DITOLAK oleh user. Trade TIDAK dicatat ke jurnal."

    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO trades (ticker, action, amount, price, stop_loss, reasoning, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            ticker,
            action.upper(),
            amount,
            current_price,
            stop_loss,
            reasoning,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return (
        f"DISETUJUI. Trade dicatat ke jurnal (SIMULASI): "
        f"{action.upper()} {amount} {ticker} @ {current_price:.2f}, "
        f"stop_loss={stop_loss:.2f}, risiko={risk_pct:.2f}% dari modal."
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
        "4. Kalau memanggil record_trade, WAJIB sertakan stop_loss yang "
        "masuk akal berdasarkan analisis teknikal (misal di bawah level "
        "support terdekat untuk BUY) - jangan asal angka.\n"
        "5. Ada batas maksimal trade per hari dan batas risiko per trade "
        "(dihitung otomatis oleh sistem) - kalau user minta trade yang "
        "ditolak karena batas ini, jelaskan alasannya dengan jelas, "
        "jangan coba mengulang dengan parameter yang beda untuk 'akalin' "
        "batasnya.\n"
        "6. Selalu jelaskan reasoning di balik saran kamu, dan ingatkan "
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