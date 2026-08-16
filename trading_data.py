"""
Fase 3, Langkah 1: Ambil Data Trading ASLI - yfinance + pandas
=================================================================
Ini beda dari semua contoh sebelumnya: sekarang datanya BENERAN, bukan
dictionary palsu kayak weather_agent. Ini yang nanti dipakai jadi isi
tool get_price / get_indicator di trading agent kamu.

Cara pakai:
1. pip install yfinance pandas
2. python3 trading_data.py

Ganti TICKER di bagian bawah buat coba aset lain:
  - "BTC-USD"  -> Bitcoin
  - "ETH-USD"  -> Ethereum
  - "BBCA.JK"  -> Bank BCA (saham Indonesia, WAJIB akhiran .JK)
  - "AAPL"     -> Apple (saham AS)
"""

import yfinance as yf
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


def get_price_history(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """
    Ambil data harga historis (OHLCV = Open, High, Low, Close, Volume)
    untuk sebuah aset dari Yahoo Finance.

    period: rentang waktu, contoh valid: "1mo", "3mo", "6mo", "1y", "5y"
    """
    asset = yf.Ticker(ticker)
    data = asset.history(period=period)
    return data


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah kolom Simple Moving Average (SMA) - rata-rata harga N hari
    terakhir. Ini indikator teknikal paling dasar buat melihat arah tren,
    karena rata-rata "meredam" naik-turun harga harian yang berisik.
    """
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    return df


def simple_trend_signal(df: pd.DataFrame) -> str:
    """
    Contoh logika BELAJAR (BUKAN saran finansial!):
    Kalau SMA jangka pendek (20 hari) ada DI ATAS SMA jangka panjang
    (50 hari), kondisi ini sering disebut 'uptrend'. Sebaliknya disebut
    'downtrend'. Konsep ini dikenal sebagai 'moving average crossover' -
    salah satu indikator paling dasar dalam analisis teknikal.
    """
    latest = df.iloc[-1]
    if pd.isna(latest["SMA_50"]):
        return "Data belum cukup untuk hitung SMA_50 (butuh minimal 50 hari data)"
    if latest["SMA_20"] > latest["SMA_50"]:
        return "Uptrend (SMA 20 di atas SMA 50)"
    else:
        return "Downtrend (SMA 20 di bawah SMA 50)"


if __name__ == "__main__":
    TICKER = "ETH-USD"  # ganti ini buat coba aset lain

    print(f"Mengambil data historis {TICKER} dari Yahoo Finance...\n")
    df = get_price_history(TICKER, period="3mo")
    df = add_moving_averages(df)

    print("5 data terakhir (harga penutupan + moving average):")
    print(df[["Close", "SMA_20", "SMA_50"]].tail())

    print(f"\nJumlah total data yang berhasil diambil: {len(df)} hari")
    print(f"Sinyal tren saat ini: {simple_trend_signal(df)}")