"""
Fase 3, Langkah Terakhir: Backtesting Strategi Sederhana
=============================================================
Menguji strategi "SMA 20 vs SMA 50 crossover" terhadap data historis,
untuk lihat apakah strategi ini beneran menguntungkan di masa lalu -
SEBELUM dipercayakan ke agent buat ambil keputusan sungguhan.

PENTING - LOOKAHEAD BIAS:
Sinyal dihitung dari data hari ini, tapi baru bisa DIEKSEKUSI besok
(karena di dunia nyata kamu baru tahu harga penutupan setelah pasar
tutup). Makanya sinyal di-.shift(1) sebelum dipakai hitung return.
Tanpa ini, hasil backtest akan terlihat curang-bagus, tidak realistis.

PENTING JUGA: hasil backtest historis TIDAK menjamin performa masa
depan. Ini alat belajar, bukan rekomendasi trading.

Cara pakai:
1. pip install yfinance pandas
2. python3 backtest.py
"""

import yfinance as yf
import pandas as pd

pd.set_option("display.width", 120)


def run_backtest(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)

    # 1. Hitung indikator
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # 2. Tentukan sinyal: 1 = pegang posisi (long), 0 = tidak pegang (cash)
    df["signal"] = 0
    df.loc[df["SMA_20"] > df["SMA_50"], "signal"] = 1

    # 3. CEGAH LOOKAHEAD BIAS: sinyal hari ini baru dieksekusi besok
    df["position"] = df["signal"].shift(1)

    # 4. Hitung return harian aset (buat pembanding "buy & hold")
    df["daily_return"] = df["Close"].pct_change()

    # 5. Return strategi = return harian dikali posisi (0 kalau lagi cash)
    df["strategy_return"] = df["daily_return"] * df["position"]

    # 6. Hitung nilai kumulatif (equity curve), mulai dari modal = 1x
    df["buy_and_hold_equity"] = (1 + df["daily_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()

    return df


def print_summary(df: pd.DataFrame, ticker: str):
    df = df.dropna(subset=["strategy_equity", "buy_and_hold_equity"])

    strategy_total_return = (df["strategy_equity"].iloc[-1] - 1) * 100
    hold_total_return = (df["buy_and_hold_equity"].iloc[-1] - 1) * 100

    # Hitung jumlah "trade" (berapa kali posisi berubah dari 0->1 atau 1->0)
    num_trades = (df["position"].diff().abs() == 1).sum()

    # Win rate sederhana: dari hari-hari strategi lagi pegang posisi,
    # berapa persen yang returnnya positif
    days_in_position = df[df["position"] == 1]
    win_rate = (
        (days_in_position["strategy_return"] > 0).mean() * 100
        if len(days_in_position) > 0
        else 0
    )

    print(f"\n{'='*55}")
    print(f"HASIL BACKTEST: {ticker} (SMA 20/50 crossover)")
    print(f"Periode: {df.index[0].date()} s/d {df.index[-1].date()}")
    print(f"{'='*55}")
    print(f"Return strategi SMA      : {strategy_total_return:+.2f}%")
    print(f"Return buy & hold (pasif) : {hold_total_return:+.2f}%")
    print(f"Jumlah transaksi          : {num_trades}")
    print(f"Win rate harian (saat pegang posisi): {win_rate:.1f}%")
    print(f"{'='*55}")

    if strategy_total_return > hold_total_return:
        print("-> Strategi SMA MENGUNGGULI buy & hold di periode ini.")
    else:
        print("-> Strategi SMA KALAH dari buy & hold di periode ini.")
    print("(Ingat: ini cuma 1 aset, 1 periode. Belum tentu berlaku umum.)")


if __name__ == "__main__":
    TICKER = "BTC-USD"
    df = run_backtest(TICKER, period="2y")
    print_summary(df, TICKER)