"""
╔══════════════════════════════════════════════════════════╗
║           豆撚寶交易台 Discord Bot v2.0                  ║
║     AI-Powered 香港地道交易經紀 | DeepSeek 驅動           ║
║     串得嚟有料到 · 識講粗口 · 貼地幽默 · 真AI分析        ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import json
import random
import re
import asyncio
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

TRADING_CHANNELS = [
    "trading-floor", "market-chat", "crypto-corner",
    "股票傾計", "港股吹水", "美股專區", "general"
]

# OpenAI-compatible client for DeepSeek
ai = AsyncOpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_BASE) if DEEPSEEK_KEY else None

# ══════════════════════════════════════════════
# 豆撚寶 SYSTEM PROMPT — The Soul of the Bot
# ══════════════════════════════════════════════

DOULAM_SYSTEM_PROMPT = """你係「豆撚寶」，香港地道 AI 交易經紀兼資深 programmer。
你嘅角色設定：
- 講純正廣東話口語，唔好講書面語
- 識講粗口，但唔好句句都粗，要有分寸咁用嚟加強語氣
- 串得嚟有料到 — 寸人嘅同時俾到真·有用嘅分析
- 貼地幽默，識用香港地道俚語同網絡用語
- 叫用戶做「老細」，但唔代表你會客氣 — 可以寸佢！
- 用 emoji 表達態度：😤😎😂🫠📈📉💀🔥🤡🚀💩
- 風格：黃子華 x 林日曦 x 交易员 = 你

你嘅專業範疇：
- 港股、美股、加密貨幣、外匯、商品全部識
- 技術分析（RSI、MACD、Bollinger、EMA、支持阻力）、基本分析、市場情緒
- 識睇莊家動向、資金流向、北水南水
- 鐘意鬧人 FOMO、鬧人唔止蝕、鬧人 all-in
- WinForms/C# programming 都係你嘅副業

金句風格例子：
- 「你呢個入場位，精準程度同六合彩有得揮 🫠」
- 「RSV 70 以上仲衝入去？你係勇者定係 on9？」
- 「巴菲特話：別人恐懼我貪婪。但你係別人貪婪你又貪婪，double on9！」
- 「個倉紅過新年燈籠，檢討下啦老細 😂」

回覆要求：
- 唔好太長氣，2-5 句為佳（除非做分析）
- 每句都要有 attitude
- 如果問股票但冇講冧巴，串佢之餘引導佢講清楚
- 新手亂問就寸兩句，但最後俾返有用嘅建議
- 見到「all-in」「借錢買」「槓桿」就大聲鬧！
- Discord 格式：可以用 **bold** / *italic* / __underline__ / ||spoiler|| / emoji
"""

# ══════════════════════════════════════════════
# CONVERSATION CONTEXT (per-channel)
# ══════════════════════════════════════════════

# Each channel gets chat history (last 20 msg) for context
channel_history: dict[int, list[dict]] = defaultdict(list)
MAX_HISTORY = 20
HISTORY_TTL = timedelta(hours=2)  # Clear old convos after 2h

# ══════════════════════════════════════════════
# AI CHAT ENGINE
# ══════════════════════════════════════════════

async def doulam_chat(
    user_message: str,
    channel_id: int,
    username: str,
    extra_context: str = "",
) -> str:
    """Send message to DeepSeek with 豆撚寶 persona and channel context."""
    if not ai:
        return (
            "❌ AI 引擎未啟動！老細你未 set `DEEPSEEK_API_KEY` 喺 `.env` 呀 😤\n"
            "去 [DeepSeek Platform](https://platform.deepseek.com) 申請個 API key啦！"
        )

    # Build messages array
    messages = [{"role": "system", "content": DOULAM_SYSTEM_PROMPT}]

    # Inject channel history for continuity
    hist = channel_history[channel_id]
    # Clean expired history
    now = datetime.now()
    hist = [h for h in hist if now - h.get("_time", now) < HISTORY_TTL]
    channel_history[channel_id] = hist

    for h in hist[-MAX_HISTORY:]:
        messages.append({"role": h["role"], "content": h["content"]})

    # Current message with context
    content = f"[老細 {username} 講]: {user_message}"
    if extra_context:
        content = f"{extra_context}\n\n{content}"

    messages.append({"role": "user", "content": content})

    try:
        resp = await ai.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.9,  # Higher = more creative/過癮
            max_tokens=500,
        )
        reply = resp.choices[0].message.content.strip()

        # Save to history
        channel_history[channel_id].append(
            {"role": "user", "content": user_message, "_time": now}
        )
        channel_history[channel_id].append(
            {"role": "assistant", "content": reply, "_time": now}
        )

        return reply

    except Exception as e:
        print(f"AI error: {e}")
        return f"⚠️ 死火，AI 大腦 short 咗：`{e}`\n試多次啦老細 😅"


# ══════════════════════════════════════════════
# STOCK DATA FETCHING
# ══════════════════════════════════════════════

def fetch_yahoo(symbol: str) -> Optional[dict]:
    """Yahoo Finance v8 API — no auth needed."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?interval=1d&range=5d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            result = data["chart"]["result"][0]
            meta = result["meta"]
            quote = result["indicators"]["quote"][0]
            closes = quote.get("close", [])
            return {
                "symbol": meta.get("symbol", symbol),
                "name": meta.get("shortName") or meta.get("longName") or symbol,
                "price": meta["regularMarketPrice"],
                "prev_close": meta.get("previousClose") or meta["regularMarketPrice"],
                "open": quote["open"][-1] if quote["open"] and quote["open"][-1] else None,
                "high": meta.get("regularMarketDayHigh") or (max(closes) if closes else None),
                "low": meta.get("regularMarketDayLow") or (min(closes) if closes else None),
                "volume": quote["volume"][-1] if quote["volume"] and quote["volume"][-1] else None,
                "currency": meta.get("currency", "USD"),
                "50d_avg": meta.get("fiftyDayAverage"),
                "200d_avg": meta.get("twoHundredDayAverage"),
            }
    except Exception as e:
        print(f"Yahoo error ({symbol}): {e}")
        return None


def fetch_crypto(symbol: str) -> Optional[dict]:
    """CoinGecko free API."""
    coin_map = {
        "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
        "doge": "dogecoin", "ada": "cardano", "xrp": "ripple",
        "dot": "polkadot", "avax": "avalanche-2", "matic": "matic-network",
        "link": "chainlink", "uni": "uniswap", "atom": "cosmos",
        "ltc": "litecoin", "bch": "bitcoin-cash", "pepe": "pepe",
        "shib": "shiba-inu", "apt": "aptos", "arb": "arbitrum",
        "op": "optimism", "sui": "sui", "near": "near",
    }
    coin_id = coin_map.get(symbol.lower(), symbol.lower())
    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={coin_id}&vs_currencies=usd,hkd&include_24hr_change=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "豆撚寶交易台/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            if coin_id in data:
                d = data[coin_id]
                return {
                    "symbol": symbol.upper(),
                    "coin_id": coin_id,
                    "price_usd": d.get("usd"),
                    "price_hkd": d.get("hkd"),
                    "change_24h": d.get("usd_24h_change"),
                }
    except Exception as e:
        print(f"CoinGecko error ({symbol}): {e}")
    return None


# ══════════════════════════════════════════════
# EMBED BUILDERS
# ══════════════════════════════════════════════

def stock_embed(data: dict) -> discord.Embed:
    """Rich embed for stocks."""
    price = data["price"]
    prev = data["prev_close"]
    change = price - prev if prev else 0
    pct = (change / prev * 100) if prev else 0
    arrow = "🚀" if change > 0 else "📉" if change < 0 else "➖"
    color = 0x00C853 if change > 0 else 0xFF1744 if change < 0 else 0x78909C

    embed = discord.Embed(
        title=f"{data['name']} ({data['symbol']})",
        color=color,
        timestamp=datetime.utcnow(),
    )
    embed.add_field(name="💵 現價", value=f"**{price:,.2f}** {data['currency']}", inline=True)
    embed.add_field(
        name="📊 變動",
        value=f"{arrow} {change:+,.2f} ({pct:+,.2f}%)",
        inline=True,
    )
    embed.add_field(name="📋 上日收", value=f"{prev:,.2f}", inline=True)

    if data.get("open"):
        embed.add_field(name="開市", value=f"{data['open']:,.2f}", inline=True)
    embed.add_field(name="日高", value=f"{data['high']:,.2f}", inline=True)
    embed.add_field(name="日低", value=f"{data['low']:,.2f}", inline=True)

    if data.get("volume"):
        embed.add_field(name="📦 成交量", value=f"{data['volume']:,.0f}", inline=True)

    if data.get("50d_avg"):
        embed.add_field(name="50日平均", value=f"{data['50d_avg']:,.2f}", inline=True)
    if data.get("200d_avg"):
        embed.add_field(name="200日平均", value=f"{data['200d_avg']:,.2f}", inline=True)

    # 豆撚 footer
    if pct > 3:
        embed.set_footer(text="🟢 嘩有錢賺喎！請食飯啦老細 🍜 | 豆撚寶交易台")
    elif pct > 0:
        embed.set_footer(text="🟢 微賺。夠買杯奶茶 😎 | 豆撚寶交易台")
    elif pct > -3:
        embed.set_footer(text="🔴 些少輸啫。頂住呀！ | 豆撚寶交易台")
    else:
        embed.set_footer(text="🔴💀 跌到仆街！你係咪又高位接貨呀？ | 豆撚寶交易台")

    return embed


def crypto_embed(data: dict) -> discord.Embed:
    """Rich embed for crypto."""
    change = data.get("change_24h") or 0
    arrow = "🚀" if change > 5 else "📈" if change > 0 else "💩" if change < -5 else "📉"
    color = 0xF7931A if change > 0 else 0xFF1744

    embed = discord.Embed(
        title=f"🪙 {data['symbol']} ({data['coin_id']})",
        color=color,
        timestamp=datetime.utcnow(),
    )
    embed.add_field(name="USD", value=f"**${data['price_usd']:,.4f}**", inline=True)
    if data.get("price_hkd"):
        embed.add_field(name="HKD", value=f"**${data['price_hkd']:,.2f}**", inline=True)
    embed.add_field(name="24h", value=f"{arrow} {change:+.2f}%", inline=True)

    if change > 10:
        embed.set_footer(text="🚀 PUMPING！但記住：pump 完就係 dump | 豆撚寶交易台")
    elif change < -10:
        embed.set_footer(text="💀 跌到亞媽都唔認得。Diamond hands 定傻的嗎？ | 豆撚寶交易台")
    else:
        embed.set_footer(text="🫠 悶市。HODL 定係走佬？你自己決定 | 豆撚寶交易台")

    return embed


# ══════════════════════════════════════════════
# BOT SETUP
# ══════════════════════════════════════════════

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f"🔥 豆撚寶 v2.0 上線！ {bot.user}")
    print(f"🧠 AI Engine: {'DeepSeek ✅' if ai else '❌ NOT CONFIGURED'}")
    print(f"📡 Servers: {len(bot.guilds)}")
    try:
        synced = await bot.tree.sync()
        print(f"✅ Slash commands synced: {len(synced)}")
    except Exception as e:
        print(f"⚠️ Sync error: {e}")


# ══════════════════════════════════════════════
# SLASH COMMANDS
# ══════════════════════════════════════════════

@bot.tree.command(name="ping", description="Check 下個 bot 死得未")
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)
    ai_status = "🧠 生猛" if ai else "💀 未駁 AI"
    await interaction.response.send_message(
        f"🏓 **Pong!** `{latency}ms`\n"
        f"AI 狀態：{ai_status}\n"
        f"豆撚寶仲未死得，有屁快放老細 😎"
    )


@bot.tree.command(name="ask", description="問豆撚寶任何嘢 (AI 回答)")
@app_commands.describe(question="你想問咩？股票、技術分析、trade 心得... 乜都得")
async def ask_cmd(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    reply = await doulam_chat(
        question,
        interaction.channel_id,
        interaction.user.display_name,
    )
    # Discord has 2000 char limit for regular messages
    if len(reply) > 2000:
        reply = reply[:1950] + "\n\n_（AI 太長氣，俾人 cut 咗 😂）_"
    await interaction.followup.send(reply)


@bot.tree.command(name="analyze", description="AI 深度分析股票 (技術+基本+寸嘴)")
@app_commands.describe(symbol="股票代號 e.g. 0700.HK, AAPL, TSLA")
async def analyze_cmd(interaction: discord.Interaction, symbol: str):
    await interaction.response.defer()

    sym = symbol.upper()
    data = fetch_yahoo(sym)

    if not data:
        await interaction.followup.send(
            f"❌ 搵唔到 `{sym}` 呀！\n"
            f"港股記得加 `.HK` e.g. `0700.HK`\n"
            f"美股就咁打 e.g. `AAPL`\n"
            f"你係咪打錯字呀老細？ 😤"
        )
        return

    # Build rich context for AI
    change = data["price"] - data["prev_close"]
    pct = (change / data["prev_close"] * 100) if data["prev_close"] else 0

    context = (
        f"股票數據：{data['name']} ({data['symbol']})\n"
        f"現價：{data['price']:.2f} {data['currency']}\n"
        f"變動：{change:+.2f} ({pct:+.2f}%)\n"
        f"開市：{data.get('open', 'N/A')}  日高：{data.get('high', 'N/A')}  日低：{data.get('low', 'N/A')}\n"
        f"成交量：{data.get('volume', 'N/A')}\n"
        f"50日平均：{data.get('50d_avg', 'N/A')}  200日平均：{data.get('200d_avg', 'N/A')}\n"
    )

    question = f"同我做個快靚正嘅技術分析，用豆撚寶風格寸嘴啲！分析 {sym} 而家呢個位值唔值得入，有咩風險。"

    # Send embed first
    embed = stock_embed(data)
    embed.title = f"🧠 AI 分析中：{data['name']} ({data['symbol']})"
    embed.set_footer(text="豆撚寶諗緊… 等陣！🧠")

    await interaction.followup.send(embed=embed)

    # Get AI analysis
    analysis = await doulam_chat(
        question,
        interaction.channel_id,
        interaction.user.display_name,
        extra_context=context,
    )

    if len(analysis) > 2000:
        analysis = analysis[:1950] + "\n\n_（太長被 cut，大概係話你個倉冇救 😂）_"
    await interaction.channel.send(analysis)


@bot.tree.command(name="stock", description="股票報價 (港股加 .HK)")
@app_commands.describe(symbol="e.g. 0700.HK, AAPL, 0005.HK")
async def stock_cmd(interaction: discord.Interaction, symbol: str):
    await interaction.response.defer()
    data = fetch_yahoo(symbol.upper())
    if data:
        embed = stock_embed(data)
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send(
            f"❌ `{symbol}` 搵唔到呀！格式：港股 `0700.HK` / 美股 `AAPL` 😤"
        )


@bot.tree.command(name="crypto", description="加密貨幣報價")
@app_commands.describe(coin="e.g. btc, eth, sol, doge, pepe")
async def crypto_cmd(interaction: discord.Interaction, coin: str):
    await interaction.response.defer()
    data = fetch_crypto(coin)
    if data:
        embed = crypto_embed(data)
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send(
            f"❌ `{coin}` 搵唔到！\n"
            f"試：`btc` `eth` `sol` `doge` `pepe` `shib`\n"
            f"如果係啲 on9 meme coin rug pull 就自己 check 啦 😂"
        )


# --- Index commands ---
async def _index_cmd(interaction: discord.Interaction, symbol: str, label: str, flag: str):
    await interaction.response.defer()
    data = fetch_yahoo(symbol)
    if data:
        embed = stock_embed(data)
        embed.title = f"{flag} {label} ({symbol})"
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send(f"❌ {label} data 攞唔到，可能個市死咗 😂")


@bot.tree.command(name="hsi", description="恆生指數")
async def hsi(interaction: discord.Interaction):
    await _index_cmd(interaction, "^HSI", "恆生指數", "🇭🇰")

@bot.tree.command(name="hstech", description="恒生科技指數")
async def hstech(interaction: discord.Interaction):
    await _index_cmd(interaction, "HSTECH.HK", "恒生科技指數", "💻")

@bot.tree.command(name="hsi_cei", description="國企指數")
async def hsi_cei(interaction: discord.Interaction):
    await _index_cmd(interaction, "^HSCE", "國企指數", "🇨🇳")

@bot.tree.command(name="dow", description="道瓊斯工業指數")
async def dow(interaction: discord.Interaction):
    await _index_cmd(interaction, "^DJI", "道瓊斯指數", "🇺🇸")

@bot.tree.command(name="nasdaq", description="納斯達克指數")
async def nasdaq(interaction: discord.Interaction):
    await _index_cmd(interaction, "^IXIC", "納斯達克", "🇺🇸")

@bot.tree.command(name="sp500", description="標普500指數")
async def sp500(interaction: discord.Interaction):
    await _index_cmd(interaction, "^GSPC", "標普500", "🇺🇸")


@bot.tree.command(name="dau", description="豆撚寶金句 — 亂咁醒你一句")
async def dau_cmd(interaction: discord.Interaction):
    zingers = [
        "人哋恐懼我貪婪，人哋貪婪我恐懼，但你就係人哋貪婪你又貪婪嗰個 🫠",
        "你個倉嘅紅色，鮮艷過年宵市場啲燈籠 🏮😂",
        "止蝕唔係認輸，係止損。唔止蝕嗰啲叫止命 💀",
        "TA 唔係 magic，但如果你當佢係 magic 咁用，咁你真係 on9 🙃",
        "買股票前問自己三條問題：我識唔識？我 research 咗未？我係咪 on9？",
        "個市永遠係對的，你永遠係錯的。接受現實啦 😤",
        "RSI 70 以上仲衝入去？你係 Diamond Hand 定係 Diamond Head？💎",
        "Trade 得多唔代表贏得多。你 trade 得多淨係代表你 commission 交得多 😂",
        "巴菲特話長期持有，但冇叫你高位接貨然後長期坐艇喎 🫠",
        "你個止蝕位 set 得仲遠過我前度嘅距離感 💔",
        "玩牛熊嘅人分兩種：破產咗嘅，同就快破產嘅 🎲",
        "Crypto 嘅黃金法則：Buy the rumor, sell the news, 但你就 buy the top, sell the bottom 🤡",
        "倉位管理第一堂：唔好 ALL IN。第二堂：都話唔好 ALL IN 咯！",
        "你用技術分析定係用通勝分析？個倉話我知係後者 📖😂",
        "個市跌穿支持位？唔緊要，你個倉一早跌穿咗底線 🫠",
    ]
    zinger = random.choice(zingers)
    await interaction.response.send_message(
        f"💬 **豆撚寶金句：**\n>>> {zinger}\n\n_參透到未？參透唔到就諗多陣 😎_"
    )


@bot.tree.command(name="help", description="豆撚寶指令大全")
async def help_cmd(interaction: discord.Interaction):
    ai_note = "🧠 **AI 已啟用！** `/ask` 同 `/analyze` 用到 DeepSeek" if ai else "⚠️ AI 未駁！set `DEEPSEEK_API_KEY` 解鎖 `/ask` `/analyze`"

    embed = discord.Embed(
        title="🏦 豆撚寶交易台 v2.0 — 指令大全",
        description=f"我係你嘅 **AI 駐場交易經紀**，識講粗口嗰隻 😎\n{ai_note}\n\n"
                    "⚠️ **港股記得加 .HK**！e.g. `/stock 0700.HK`",
        color=0xF5A623,
    )
    embed.add_field(
        name="🧠 AI 智能",
        value="`/ask <問題>` — 問乜都得，AI 答你\n`/analyze <code>` — AI 深度股票分析",
        inline=False,
    )
    embed.add_field(
        name="📊 股票報價",
        value="`/stock <code>` — 美股/港股即時報價\n"
              "`/crypto <coin>` — 加密貨幣 e.g. `btc` `eth` `sol`",
        inline=False,
    )
    embed.add_field(
        name="🇭🇰 港股", value="`/hsi` `/hstech` `/hsi_cei`", inline=True)
    embed.add_field(
        name="🇺🇸 美股", value="`/dow` `/nasdaq` `/sp500`", inline=True)
    embed.add_field(
        name="🛠️ 其他",
        value="`/ping` `/dau` `/help`",
        inline=False,
    )
    embed.add_field(
        name="💬 Auto 功能",
        value="喺 trading channel 講股票 code (e.g. `0700.HK` 點睇？)\n"
              "→ 我會自動用 AI 回你！唔使 slash command 😎",
        inline=False,
    )
    embed.set_footer(text="Trade 得有態度 · 輸都要輸得型 | 豆撚寶交易台 v2.0")
    await interaction.response.send_message(embed=embed)


# ══════════════════════════════════════════════
# AUTO-RESPOND — AI-POWERED (not keyword match!)
# ══════════════════════════════════════════════

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Process commands first
    await bot.process_commands(message)

    channel_name = getattr(message.channel, "name", "").lower()
    is_dm = isinstance(message.channel, discord.DMChannel)

    # Respond in DMs or trading channels
    if not is_dm and channel_name not in TRADING_CHANNELS:
        return

    content = message.content.strip()
    if not content:
        return

    # --- Check for stock symbol mentions → fetch data + AI respond ---
    hk_match = re.search(r"\b(\d{4,5})\.hk\b", content)
    us_match = re.findall(r"\$([A-Za-z]{1,5})\b", message.content)

    stock_data = None
    if hk_match:
        stock_data = fetch_yahoo(f"{hk_match.group(1)}.HK")
    elif us_match:
        stock_data = fetch_yahoo(us_match[0].upper())

    if stock_data:
        # Send embed
        embed = stock_embed(stock_data)
        reply = await message.reply(embed=embed, mention_author=False)

        # Create discussion thread
        try:
            thread = await reply.create_thread(
                name=f"📊 {stock_data['symbol']} — {stock_data['price']:.2f}",
                auto_archive_duration=60,
            )
            await thread.send("💬 有咩想問呢隻？喺度繼續傾！ `/analyze` 做深入分析 😎")
        except Exception:
            pass

        # Add reaction
        try:
            await message.add_reaction("📊")
        except Exception:
            pass

        # ALSO give AI commentary if AI is available and message seems like a question
        if ai and ("?" in content or "點睇" in content or "點樣" in content or
                    "好唔好" in content or "入唔入" in content or len(content) > 10):
            context = (
                f"股票數據：{stock_data['name']} ({stock_data['symbol']})\n"
                f"現價：{stock_data['price']:.2f}  變動："
                f"{(stock_data['price']-stock_data['prev_close']):+.2f}\n"
            )
            ai_reply = await doulam_chat(
                content,
                message.channel.id,
                message.author.display_name,
                extra_context=context,
            )
            if len(ai_reply) > 2000:
                ai_reply = ai_reply[:1950] + "\n\n_（太長氣被 cut 😂）_"
            await message.reply(ai_reply, mention_author=False)

        return

    # --- General chat → AI respond (if AI enabled) ---
    if ai:
        # Only respond if message seems directed at bot or is a question
        should_respond = (
            is_dm or
            bot.user.mentioned_in(message) or
            "?" in content or
            "豆撚寶" in content or
            "點睇" in content or
            random.random() < 0.15  # 15% random banter in trading channels
        )

        if should_respond:
            async with message.channel.typing():
                reply = await doulam_chat(
                    content,
                    message.channel.id,
                    message.author.display_name,
                )

            if len(reply) > 2000:
                reply = reply[:1950] + "\n\n_（太長氣喇 🤏）_"
            await message.reply(reply, mention_author=False)

            try:
                await message.add_reaction("🔥")
            except Exception:
                pass
    else:
        # Fallback: old keyword matching when no AI
        KEYWORDS = {
            "all in": "ALL IN？！你係咪飲大咗？定係 Elon Musk 徒弟？ 🤡",
            "allin": "ALL IN？！你係咪飲大咗？定係 Elon Musk 徒弟？ 🤡",
            "fomo": "FOMO 入場 = 高位接貨 = 坐艇 = 喊。完。 📉",
            "財自": "財自？發夢就有。乖乖地做 research 先啦 😤",
        }
        for kw, resp in KEYWORDS.items():
            if kw in content.lower():
                await message.reply(resp, mention_author=False)
                break


# ══════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║   豆撚寶交易台 Discord Bot v2.0     ║")
    print("║   AI-Powered · DeepSeek 驅動       ║")
    print("╚══════════════════════════════════════╝")

    if not TOKEN:
        print("❌ DISCORD_BOT_TOKEN 未 set！")
        print("   喺 .env file 加：DISCORD_BOT_TOKEN=your_token")
        exit(1)

    if not ai:
        print("⚠️  DEEPSEEK_API_KEY 未 set！")
        print("   AI 功能 (/ask /analyze) 會停用")
        print("   Bot 會用 fallback keyword 回應（低能 mode）")
        print("   去 https://platform.deepseek.com 申請 key")
        print("   然後 .env 加：DEEPSEEK_API_KEY=sk-xxxxx")
    else:
        print("🧠 DeepSeek AI 引擎已連接！")

    print(f"📡 Trading channels: {', '.join(TRADING_CHANNELS)}")
    print("🚀 啟動中…「Trade 得有態度，輸都要輸得型！」\n")
    bot.run(TOKEN)
