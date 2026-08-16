"""
Weather + Kurs Agent - Versi LangChain (penutup Fase 2)
==========================================================
Ini agent yang FUNGSINYA SAMA PERSIS dengan multi_tool_agent.py yang kamu
buat manual sebelumnya - cuma sekarang LangChain yang ngurus:
  - while-loop agent loop-nya
  - manajemen riwayat pesan (messages/contents)
  - dispatch tool by name

Bandingkan panjang & kerumitan kode ini dengan multi_tool_agent.py.
Itu bukti nyata kenapa framework kayak LangChain dipakai di project
production - bukan karena konsepnya beda, tapi karena boilerplate-nya
udah dibungkus.

Cara pakai:
1. pip install -qU langchain "langchain[google-genai]"
2. Pakai API key Gemini yang sama kayak sebelumnya (GEMINI_API_KEY).
   Kalau LangChain komplain nggak nemu key, set juga:
   export GOOGLE_API_KEY="$GEMINI_API_KEY"
3. python3 langchain_agent.py
"""

from langchain.agents import create_agent


# ── TOOL: cukup fungsi Python biasa + docstring yang jelas ──────────────
# (Docstring ini yang dibaca model buat tahu kapan harus pakai tool ini -
#  menggantikan "description" panjang yang kita tulis manual di dictionary
#  sebelumnya)
def get_weather(city: str) -> str:
    """Ambil informasi cuaca terkini untuk sebuah kota."""
    fake_data = {
        "jakarta": "32°C, cerah berawan",
        "bandung": "24°C, hujan ringan",
        "surabaya": "34°C, cerah",
        "tokyo": "18°C, berawan",
    }
    return fake_data.get(city.lower(), "data cuaca tidak ditemukan")


def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Ambil kurs tukar antara dua mata uang, contoh: IDR ke JPY."""
    fake_rates = {
        ("idr", "jpy"): 0.0095,
        ("idr", "usd"): 0.000062,
        ("jpy", "idr"): 105.3,
        ("usd", "idr"): 16100,
    }
    rate = fake_rates.get((from_currency.lower(), to_currency.lower()))
    if rate is None:
        return "kurs tidak ditemukan"
    return f"1 {from_currency.upper()} = {rate} {to_currency.upper()}"


# ── AGENT: satu pemanggilan ini menggantikan SEMUA while-loop manual kita
agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[get_weather, get_exchange_rate],
    system_prompt=(
        "Kamu adalah asisten travel yang membantu user merencanakan "
        "liburan. Selalu gunakan tool yang tersedia untuk data faktual, "
        "jangan menebak dari pengetahuan umum."
    ),
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Saya mau liburan ke Tokyo. Gimana cuacanya di sana? "
                        "Terus kalau saya tukar uang 5 juta Rupiah ke Yen, "
                        "kira-kira dapat berapa?"
                    ),
                }
            ]
        }
    )

    # Tampilkan langkah-langkah yang diambil agent (mirip print manual
    # yang kita bikin sendiri sebelumnya) - LangChain menyimpan semuanya
    # otomatis di result["messages"]
    def extract_text(content) -> str:
        """Content bisa berupa string biasa ATAU list of content blocks,
        tergantung model/provider-nya. Fungsi ini menyeragamkan keduanya
        jadi teks biasa yang enak dibaca."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return str(content)

    for message in result["messages"]:
        msg_type = message.__class__.__name__
        if msg_type == "AIMessage" and message.tool_calls:
            for tc in message.tool_calls:
                print(f"[Agent memanggil tool: {tc['name']}({tc['args']})]")
        elif msg_type == "AIMessage" and message.content:
            print(f"\nAgent: {extract_text(message.content)}")