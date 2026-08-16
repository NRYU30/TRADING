"""
Menggabungkan Fase 2 + Fase 3: Agent dengan Data Pasar SUNGGUHAN
====================================================================
Ini titik penting: tool get_weather yang selama ini dipakai buat latihan
sekarang diganti tool yang narik DATA PASAR ASLI lewat yfinance.

Strukturnya PERSIS sama kayak langchain_agent.py sebelumnya - cuma
tool-nya yang beda isi. Ini bukti nyata kenapa kita belajar konsepnya
dulu: begitu paham pola dasarnya, ganti "isi" tool jadi apa pun
(cuaca -> kurs -> harga saham) tinggal soal ganti fungsi Python biasa.

PENTING: Ini MURNI alat belajar. Sinyal SMA yang dihasilkan BUKAN saran
finansial - jangan dipakai buat keputusan trading sungguhan tanpa
riset & pemahaman risiko lebih lanjut.

Cara pakai:
1. pip install yfinance pandas -qU langchain "langchain[google-genai]"
2. python3 market_agent.py
"""

import yfinance as yf
import pandas as pd
from langchain.agents import create_agent


# ── TOOL 1: harga terkini ────────────────────────────────────────────────
def get_current_price(ticker: str) -> str:
    """
    Ambil harga penutupan terkini untuk sebuah aset.
    Contoh ticker: BTC-USD, ETH-USD, BBCA.JK (saham Indonesia), AAPL.
    """
    data = yf.Ticker(ticker).history(period="5d")
    if data.empty:
        return f"Ticker '{ticker}' tidak ditemukan atau tidak ada data."
    latest_price = data["Close"].iloc[-1]
    latest_date = data.index[-1].strftime("%Y-%m-%d")
    return f"Harga {ticker} pada {latest_date}: {latest_price:.2f}"


# ── TOOL 2: sinyal tren berdasarkan moving average ──────────────────────
def get_trend_signal(ticker: str) -> str:
    """
    Analisis tren sederhana pakai SMA 20 hari vs SMA 50 hari untuk
    sebuah aset. Berguna untuk melihat apakah aset sedang uptrend atau
    downtrend secara teknikal.
    """
    data = yf.Ticker(ticker).history(period="3mo")
    if data.empty:
        return f"Ticker '{ticker}' tidak ditemukan atau tidak ada data."

    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    latest = data.iloc[-1]

    if pd.isna(latest["SMA_50"]):
        return "Data belum cukup untuk analisis SMA 50 hari."

    trend = "Uptrend" if latest["SMA_20"] > latest["SMA_50"] else "Downtrend"
    return (
        f"{ticker}: Close={latest['Close']:.2f}, "
        f"SMA_20={latest['SMA_20']:.2f}, SMA_50={latest['SMA_50']:.2f} "
        f"-> {trend}"
    )


# ── AGENT ─────────────────────────────────────────────────────────────
agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[get_current_price, get_trend_signal],
    system_prompt=(
        "Kamu adalah asisten analisis pasar untuk tujuan BELAJAR. "
        "Selalu gunakan tool untuk data harga & tren - jangan pernah "
        "menebak harga dari pengetahuan umum karena harga pasar berubah "
        "tiap saat. Setelah memberi data, selalu ingatkan bahwa ini "
        "bukan saran finansial dan risiko trading sepenuhnya tanggung "
        "jawab pengguna."
    ),
)


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Gimana kondisi BTC-USD sekarang? Sedang uptrend "
                        "atau downtrend? Kasih harga terkininya juga."
                    ),
                }
            ]
        }
    )

    for message in result["messages"]:
        msg_type = message.__class__.__name__
        if msg_type == "AIMessage" and message.tool_calls:
            for tc in message.tool_calls:
                print(f"[Agent memanggil tool: {tc['name']}({tc['args']})]")
        elif msg_type == "AIMessage" and message.content:
            print(f"\nAgent: {extract_text(message.content)}")