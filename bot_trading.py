"""
Trading Agent Monitoring - Cek Berkala & Notifikasi Selektif
==================================================================
Script ini didesain untuk dijalankan BERKALA (misal tiap 1 jam lewat
scheduler), bukan dibiarkan jalan terus dengan while True. Tiap kali
dijalankan, dia:

1. Cek kondisi tren untuk tiap ticker di WATCHLIST
2. Bandingkan dengan kondisi TERAKHIR yang tersimpan di database
3. TIDAK ada perubahan -> diam saja, keluar (hemat biaya API & tidak
   spam notifikasi)
4. ADA perubahan (misal Downtrend -> Uptrend) -> panggil AI agent bikin
   ringkasan singkat, kirim ke Telegram
5. Simpan state baru untuk perbandingan run berikutnya

PRINSIP PENTING: keputusan "apakah perlu lapor" itu logic PASTI di kode
(bandingkan data tersimpan vs data baru) - BUKAN diserahkan ke LLM
mikir ulang tiap kali dijalankan. LLM cuma dipanggil saat beneran ada
sesuatu yang perlu dijelaskan, biar hemat biaya API juga.

Cara pakai (manual dulu, untuk testing sebelum dijadwalkan otomatis):
1. pip install yfinance pandas requests python-dotenv -qU langchain "langchain[google-genai]"
2. python3 monitor_agent.py
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

DB_FILE = "trading_journal.db"
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Ticker yang dipantau - tambah/kurangi sesuai kebutuhan
WATCHLIST = ["BTC-USD", "ETH-USD"]


# ── SETUP: tabel buat nyimpen state tren terakhir tiap ticker ───────────
def init_monitor_table():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS monitor_state (
            ticker TEXT PRIMARY KEY,
            last_trend TEXT,
            last_checked_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def get_last_trend(ticker: str):
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute(
        "SELECT last_trend FROM monitor_state WHERE ticker = ?", (ticker,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def save_trend(ticker: str, trend: str):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO monitor_state (ticker, last_trend, last_checked_at) "
        "VALUES (?, ?, ?) ON CONFLICT(ticker) DO UPDATE SET "
        "last_trend=excluded.last_trend, last_checked_at=excluded.last_checked_at",
        (ticker, trend, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


# ── Ambil tren terkini (logic PASTI, bukan tebakan LLM) ──────────────────
def get_current_trend(ticker: str):
    """Return (trend, close, sma_short, sma_long) atau None kalau data kosong."""
    data = yf.Ticker(ticker).history(period="3mo")
    if data.empty:
        return None
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    latest = data.iloc[-1]
    if pd.isna(latest["SMA_50"]):
        return None
    trend = "Uptrend" if latest["SMA_20"] > latest["SMA_50"] else "Downtrend"
    return trend, latest["Close"], latest["SMA_20"], latest["SMA_50"]


def send_telegram_message(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID belum diisi.")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Gagal kirim Telegram: {e}")
        return False


# ── Agent kecil, cuma dipanggil SAAT ada perubahan tren ─────────────────
def get_trend_signal_tool(ticker: str) -> str:
    """Analisis tren teknikal (SMA 20 vs SMA 50) untuk sebuah aset."""
    result = get_current_trend(ticker)
    if result is None:
        return f"Data untuk {ticker} tidak cukup/tidak ditemukan."
    trend, close, sma20, sma50 = result
    return f"{ticker}: Close={close:.2f}, SMA_20={sma20:.2f}, SMA_50={sma50:.2f} -> {trend}"


summary_agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[get_trend_signal_tool],
    system_prompt=(
        "Kamu adalah asisten yang bikin ringkasan SINGKAT (maksimal 3 "
        "kalimat) untuk notifikasi Telegram tentang perubahan tren aset "
        "kripto. Gunakan tool get_trend_signal_tool untuk data faktual, "
        "jangan menebak angka. Selalu tutup dengan pengingat singkat "
        "bahwa ini bukan saran finansial. Gunakan format Markdown "
        "Telegram sederhana (*bold* pakai tanda bintang)."
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


def generate_summary(ticker: str, old_trend, new_trend: str) -> str:
    if old_trend is None:
        prompt = f"Berikan ringkasan kondisi awal {ticker} yang baru mulai dipantau."
    else:
        prompt = (
            f"Tren {ticker} baru saja berubah dari {old_trend} menjadi "
            f"{new_trend}. Beri ringkasan singkat soal apa artinya ini."
        )
    result = summary_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    final_message = result["messages"][-1]
    return extract_text(final_message.content)


# ── Logic utama: cek satu ticker, putuskan lapor atau diam ──────────────
def check_ticker(ticker: str):
    print(f"Mengecek {ticker}...")
    result = get_current_trend(ticker)
    if result is None:
        print(f"  Data {ticker} tidak cukup, skip.")
        return

    new_trend = result[0]
    old_trend = get_last_trend(ticker)

    if old_trend == new_trend:
        print(f"  Tidak ada perubahan ({new_trend}). Diam, tidak kirim notif.")
        save_trend(ticker, new_trend)  # tetap update timestamp pengecekan
        return

    print(f"  PERUBAHAN TERDETEKSI: {old_trend} -> {new_trend}. Mengirim notifikasi...")
    summary = generate_summary(ticker, old_trend, new_trend)
    icon = "🟢" if new_trend == "Uptrend" else "🔴"
    message = (
        f"{icon} *{ticker}: {old_trend or 'Baru dipantau'} → {new_trend}*\n\n{summary}"
    )

    if send_telegram_message(message):
        print("  Notifikasi terkirim.")
    else:
        print("  Notifikasi GAGAL terkirim.")

    save_trend(ticker, new_trend)


if __name__ == "__main__":
    init_monitor_table()
    for ticker in WATCHLIST:
        check_ticker(ticker)
    print("\nSelesai mengecek semua watchlist.")