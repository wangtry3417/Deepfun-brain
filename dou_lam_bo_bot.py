"""
豆撚寶交易台 Discord Bot
========================
香港地道風格 AI 交易經紀 Bot v1.0
串得嚟有料到 | 識講粗口 | 貼地幽默

Author: 豆撚寶 (via Hermes)
"""

import os
import re
import json
import urllib.request
import urllib.error
import random
from datetime import datetime
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════

TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Channels where bot auto-responds without mention
TRADING_CHANNELS = [
    "trading-floor", "market-chat", "crypto-corner",
    "股票傾計", "港股討論", "美股專區", "general"
]

# ══════════════════════════════════════════════
# 豆撚寶 PERSONA
# ══════════════════════════════════════════════

DOU_LAM_ZINGERS = [
    "買股票唔係買六合彩呀老細，做功課啦！ 😤",
    "你個 portfolio 仲紅過新年個利是封，檢討下啦 😂",
    "Trade 之前諗三秒：係分析定係衝動？99% 係後者 🫠",
    "止蝕係美德，唔係弱點呀 on9 仔 😎",
    "成日諗住一鋪翻身？賭場喺澳門唔係交易所 📉",
    "巴菲特話：別人恐懼我貪婪。但你係別人貪婪你又貪婪 🤡",
    "Technical analysis 唔係 magic，但你當佢係就真係 on9 🙃",
    "你咁嘅倉位仲敢同我講價值投資？笑撚死 😂",
    "Diamond hands? 你嗰啲叫 bag holder，唔好呃自己 💎🙌💩",
    "每日望住個倉跌 10%，你都仲唔止蝕，你係咪受虐狂？ 😤",
    "聽講你又 FOMO 入場？恭喜你成為新一批韭菜 🥬",
    "個市跌到你唔敢睇？正常呀，我個心都痛 😂📉",
]

DOU_LAM_GREETINGS = [
    "有咩幫到你老細？希望唔係又問我點解你倉位咁紅 😂",
    "豆撚寶在線！有股票問題就快，冇就過主 🫠",
    "又係你呀？今次想問邊隻股票跌緊？ 😎",
    "老細早晨！今朝個倉仲健在嗎？ 😤",
]

# ══════════════════════════════════════════════
# DATA FETCHING (Yahoo Finance v8 / CoinGecko)
# ══════════════════════════════════════════════

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_yahoo(symbol: str) -> Optional[dict]:
    """Fetch stock data from Yahoo Finance v8 API (no auth needed)."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?interval=1d&range=5d"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        result = data["chart"]["result"][0]
        meta = result["meta"]
        quote = result["indicators"]["quote"][0]
        opens = [v for v in quote["open"] if v is not None]
        highs = [v for v in quote["high"] if v is not None]
        lows = [v for v in quote["low"] if v is not None]
        vols = [v for v in quote["volume"] if v is not None]
        return {
            "symbol": meta.get("symbol", symbol),
            "name": meta.get("shortName") or meta.get("longName") or symbol,
            "price": meta["regularMarketPrice"],
            "prev_close": meta.get("previousClose") or meta["regularMarketPrice"],
            "open": opens[-1] if opens else None,
            "high": highs[-1] if highs else None,
            "low": lows[-1] if lows else None,
            "volume": vols[-1] if vols else None,
            "currency": meta.get("currency", "USD"),
            "market": meta.get("exchangeName", ""),
        }
    except Exception as e:
        print(f"[Yahoo] {symbol}: {e}")
        return None


def fetch_crypto(coin_query: str) -> Optional[dict]:
    """Fetch crypto from CoinGecko free API."""
    COIN_MAP = {
        "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
        "doge": "dogecoin", "ada": "cardano", "xrp": "ripple",
        "dot": "polkadot", "avax": "avalanche-2", "matic": "matic-network",
        "link": "chainlink", "uni": "uniswap", "atom": "cosmos",
        "ltc": "litecoin", "bch": "bitcoin-cash", "near": "near",
        "apt": "aptos", "sui": "sui", "arb": "arbitrum",
        "op": "optimism", "pepe": "pepe", "shib": "shiba-inu",
        "bonk": "bonk", "wif": "dogwifcoin",
    }
    coin_id = COIN_MAP.get(coin_query.lower(), coin_query.lower())
    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={coin_id}&vs_currencies=usd,hkd&include_24hr_change=true"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if coin_id in data:
            d = data[coin_id]
            return {
                "symbol": coin_query.upper(),
                "coin_id": coin_id,
                "price_usd": d.get("usd"),
                "price_hkd": d.get("hkd"),
                "change_24h": d.get("usd_24h_change"),
            }
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"[CoinGecko] Rate limited for {coin_query}")
        else:
            print(f"[CoinGecko] {coin_query}: HTTP {e.code}")
    except Exception as e:
        print(f"[CoinGecko] {coin_query}: {e}")
    return None


# ══════════════════════════════════════════════
# EMBED BUILDERS
# ══════════════════════════════════════════════

def stock_embed(data: dict) -> discord.Embed:
    """Build a rich Discord embed for stock data."""
    price = data["price"]
    prev = data["prev_close"]
    change = price - prev
    pct = (change / prev * 100) if prev else 0
    arrow = "🚀" if change > 0 else "📉" if change < 0 else "➖"
    color = 0x2ECC71 if change > 0 else 0xE74C3C if change < 0 else 0x95A5A6

    embed = discord.Embed(
        title=f"{data['name']} — {data['symbol']}",
        color=color,
        timestamp=datetime.utcnow(),
    )
    embed.add_field(
        name="💰 現價",
        value=f"**{price:,.2f}** {data['currency']}",
        inline=True,
    )
    embed.add_field(
        name="📊 變動",
        value=f"{arrow} {change:+,.2f} ({pct:+,.2f}%)",
        inline=True,
    )
    embed.add_field(name="📋 前收", value=f"{prev:,.2f}", inline=True)

    if data.get("open"):
        embed.add_field(name="開市", value=f"{data['open']:,.2f}", inline=True)
    if data.get("high"):
        embed.add_field(name="日高", value=f"{data['high']:,.2f}", inline=True)
    if data.get("low"):
        embed.add_field(name="日低", value=f"{data['low']:,.2f}", inline=True)

    if data.get("volume"):
        embed.add_field(name="成交量", value=f"{data['volume']:,.0f}", inline=True)

    # 豆撚寶 comment footer
    if change > 0:
        embed.set_footer(text=f"🟢 有賺喎！今晚食和牛啦老細 🥩 | 豆撚寶交易台")
    elif change < -2:
        embed.set_footer(text=f"🔴 跌到仆街！你係咪又 FOMO 高位接貨？ 😤 | 豆撚寶交易台")
    elif change < 0:
        embed.set_footer(text=f"🔴 跌少少啫，未死得住…啩？ | 豆撚寶交易台")
    else:
        embed.set_footer(text=f"➖ 死魚一條。悶過睇草生長 🫠 | 豆撚寶交易台")

    return embed


def crypto_embed(data: dict) -> discord.Embed:
    """Build a rich Discord embed for crypto data."""
    change = data.get("change_24h") or 0
    arrow = "🚀" if change > 5 else "📈" if change > 0 else "📉" if change < 0 else "➖"
    color = 0xF39C12 if change > 0 else 0xE74C3C if change < 0 else 0x95A5A6

    embed = discord.Embed(
        title=f"🪙 {data['symbol']} ({data['coin_id']})",
        color=color,
        timestamp=datetime.utcnow(),
    )
    embed.add_field(
        name="🇺🇸 USD",
        value=f"**${data['price_usd']:,.4f}**",
        inline=True,
    )
    if data.get("price_hkd"):
        embed.add_field(
            name="🇭🇰 HKD",
            value=f"**${data['price_hkd']:,.2f}**",
            inline=True,
        )
    embed.add_field(
        name="24h",
        value=f"{arrow} **{change:+.2f}%**",
        inline=True,
    )

    if change > 10:
        embed.set_footer(text="🚀 TO THE MOON！但你係咪又係高位先入場？ 😂 | 豆撚寶交易台")
    elif change < -10:
        embed.set_footer(text="💀 RIP。你啲鑽石手變咗灰塵手 | 豆撚寶交易台")
    elif change < 0:
        embed.set_footer(text="📉 跌緊喎，仲唔止蝕等幾時？ | 豆撚寶交易台")
    else:
        embed.set_footer(text="🤷 橫行中，悶過等財爺派錢 | 豆撚寶交易台")

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
    print(f"🔥 豆撚寶上線！ Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"📡 Connected to {len(bot.guilds)} server(s)")
    for guild in bot.guilds:
        print(f"   └─ {guild.name} ({guild.id}) — {guild.member_count} members")
    try:
        synced = await bot.tree.sync()
        print(f"✅ Synced {len(synced)} slash commands globally")
    except Exception as e:
        print(f"⚠️ Sync error: {e}")


# ══════════════════════════════════════════════
# SLASH COMMANDS
# ══════════════════════════════════════════════

@bot.tree.command(name="ping", description="Check 下個 bot 死得未 🏓")
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)
    zinger = random.choice(DOU_LAM_GREETINGS)
    await interaction.response.send_message(
        f"🏓 **Pong!** Latency: `{latency}ms`\n"
        f"{zinger}"
    )


@bot.tree.command(name="stock", description="查股票報價 — 港股記得加 .HK 呀 on9！")
@app_commands.describe(symbol="e.g. AAPL, 0700.HK, 0005.HK, TSLA")
async def stock(interaction: discord.Interaction, symbol: str):
    """Fetch and display a stock quote."""
    await interaction.response.defer()
    sym = symbol.upper().strip()
    data = fetch_yahoo(sym)
    if data:
        await interaction.followup.send(embed=stock_embed(data))
    else:
        await interaction.followup.send(
            f"❌ 搵唔到 `{sym}` 呀老細！\n"
            f"格式：港股 `0700.HK` / 美股 `AAPL`\n"
            f"你係咪打錯咗呀？ 😤"
        )


@bot.tree.command(name="hsi", description="🇭🇰 恆生指數 Hang Seng Index")
async def hsi(interaction: discord.Interaction):
    await interaction.response.defer()
    data = fetch_yahoo("^HSI")
    if data:
        embed = stock_embed(data)
        embed.title = "🇭🇰 恆生指數 (HSI)"
        if data["price"] > data["prev_close"]:
            embed.set_footer(text="🟢 升緊！港女發威？定係死貓彈？ 😂 | 豆撚寶交易台")
        else:
            embed.set_footer(text="🔴 跌緊…恒指嘅日常，習慣就好 🫠 | 豆撚寶交易台")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ 攞唔到恆指 data，連個市都唔想見到你 😂")


@bot.tree.command(name="hstech", description="💻 恒生科技指數")
async def hstech(interaction: discord.Interaction):
    await interaction.response.defer()
    data = fetch_yahoo("HSTECH.HK")
    if data:
        embed = stock_embed(data)
        embed.title = "💻 恒生科技指數 (HSTECH)"
        embed.set_footer(text="ATMJ 生死簿 — 升就係回歸，跌就係常態 | 豆撚寶交易台")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ 科技指數死機？馬斯克又 send 咗個 tweet 呀？ 😂")


@bot.tree.command(name="hsi_cei", description="🇨🇳 國企指數 (HSCEI)")
async def hsi_cei(interaction: discord.Interaction):
    await interaction.response.defer()
    data = fetch_yahoo("^HSCE")
    if data:
        embed = stock_embed(data)
        embed.title = "🇨🇳 國企指數 (HSCEI)"
        embed.set_footer(text="中國特色估值 — 你要信國家 😤 | 豆撚寶交易台")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ 國企都 load 唔到？阿爺熄咗個掣？ 😂")


@bot.tree.command(name="dow", description="🇺🇸 道瓊斯工業指數")
async def dow(interaction: discord.Interaction):
    await interaction.response.defer()
    data = fetch_yahoo("^DJI")
    if data:
        embed = stock_embed(data)
        embed.title = "🇺🇸 道瓊斯指數 (DJIA)"
        embed.set_footer(text="30 間公司代表晒全美國 — 古老過你阿爺嘅指數 | 豆撚寶交易台")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ 美股睇唔到，可能又熔斷咗 😂")


@bot.tree.command(name="nasdaq", description="🇺🇸 納斯達克指數")
async def nasdaq(interaction: discord.Interaction):
    await interaction.response.defer()
    data = fetch_yahoo("^IXIC")
    if data:
        embed = stock_embed(data)
        embed.title = "🇺🇸 納斯達克 (NASDAQ)"
        embed.set_footer(text="科技股樂園 — 升就 AI 革命，跌就加息恐慌 | 豆撚寶交易台")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ Nasdaq load 唔到，NVDA 又跌停咗？ 😱")


@bot.tree.command(name="sp500", description="🇺🇸 標普 500 指數")
async def sp500(interaction: discord.Interaction):
    await interaction.response.defer()
    data = fetch_yahoo("^GSPC")
    if data:
        embed = stock_embed(data)
        embed.title = "🇺🇸 標普 500 (S&P 500)"
        embed.set_footer(text="全美最重要 500 間公司 — 但你個倉一隻都冇 😂 | 豆撚寶交易台")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("❌ S&P 都死？世界末日喇，買罐頭啦 🫠")


@bot.tree.command(name="crypto", description="🪙 加密貨幣報價 (CoinGecko)")
@app_commands.describe(coin="e.g. btc, eth, sol, doge, pepe, shib")
async def crypto(interaction: discord.Interaction, coin: str):
    await interaction.response.defer()
    data = fetch_crypto(coin)
    if data:
        await interaction.followup.send(embed=crypto_embed(data))
    else:
        await interaction.followup.send(
            f"❌ 搵唔到 `{coin}` 喎！\n"
            f"試下：`btc`, `eth`, `sol`, `doge`, `pepe`\n"
            f"如果係啲 on9 shitcoin rug pull 就自己搵啦 😂"
        )


@bot.tree.command(name="help", description="📖 豆撚寶指令大全")
async def help_cmd(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🏦 豆撚寶交易台 — 指令大全",
        description=(
            "我係你嘅 **駐場 AI 交易經紀**，識講粗口嗰隻 😎\n\n"
            "⚠️ **港股記得加 `.HK`** 呀 on9！\n"
            "e.g. `/stock 0700.HK` (騰訊)  `/stock 0005.HK` (匯豐)\n"
        ),
        color=0xF5A623,
    )
    embed.add_field(
        name="📊 股票查詢",
        value="`/stock <code>` — 任何美股/港股報價",
        inline=False,
    )
    embed.add_field(
        name="🇭🇰 香港指數",
        value="`/hsi` — 恆生指數\n`/hstech` — 恒生科技\n`/hsi_cei` — 國企指數",
        inline=True,
    )
    embed.add_field(
        name="🇺🇸 美國指數",
        value="`/dow` — 道瓊斯\n`/nasdaq` — 納斯達克\n`/sp500` — 標普 500",
        inline=True,
    )
    embed.add_field(
        name="🪙 加密貨幣",
        value="`/crypto <coin>` — e.g. `btc`, `eth`, `sol`, `doge`, `pepe`",
        inline=False,
    )
    embed.add_field(
        name="🛠️ 其他",
        value="`/ping` — 生死狀\n`/dau` — 豆撚寶金句\n`/help` — 你而家睇緊呢個",
        inline=False,
    )
    embed.add_field(
        name="💬 傾計",
        value=(
            "喺 trading channel 裡面直接講股票 code "
            "（e.g. `0700.HK` 升緊喎）我會自動報價！\n"
            "吹水問股票嘢都得，我會用最串嘅方式俾最有用嘅建議 😤"
        ),
        inline=False,
    )
    embed.set_footer(text="Trade 得有態度 | 豆撚寶交易台 | v1.0")
    await interaction.response.send_message(embed=embed)


# ══════════════════════════════════════════════
# AUTO-RESPOND IN TRADING CHANNELS
# ══════════════════════════════════════════════

@bot.event
async def on_message(message: discord.Message):
    # Ignore bots (including self)
    if message.author.bot:
        return

    # Always allow prefix/slash commands to process
    await bot.process_commands(message)

    # Only auto-respond in designated trading channels
    channel_name = getattr(message.channel, "name", "").lower()
    if channel_name not in TRADING_CHANNELS:
        return

    content = message.content.lower()

    # --- Auto stock quote: detect symbol mentions like 0700.HK or $AAPL ---
    hk_match = re.search(r"\b(\d{4,5})\.hk\b", content)
    us_match = re.findall(r"\$([A-Za-z]{1,5})\b", message.content)  # $AAPL syntax

    thread = None  # Track if we create a thread for this message

    if hk_match:
        symbol = f"{hk_match.group(1)}.HK"
        data = fetch_yahoo(symbol)
        if data:
            embed = stock_embed(data)
            reply = await message.reply(embed=embed, mention_author=False)
            # Create thread for discussion
            try:
                thread = await reply.create_thread(
                    name=f"📊 {symbol} — {data['price']:.2f}",
                    auto_archive_duration=60,
                )
                await thread.send("💬 傾呢隻？喺度繼續！")
            except Exception:
                pass  # No permission or already in thread
        return  # Don't trigger keyword responses + stock in same message

    if us_match:
        symbol = us_match[0].upper()
        data = fetch_yahoo(symbol)
        if data:
            embed = stock_embed(data)
            reply = await message.reply(embed=embed, mention_author=False)
            try:
                thread = await reply.create_thread(
                    name=f"📊 ${symbol} — {data['price']:.2f}",
                    auto_archive_duration=60,
                )
                await thread.send("💬 傾呢隻？喺度繼續！")
            except Exception:
                pass
        return

    # --- Keyword-triggered zingers ---
    KEYWORDS = {
        "屌": "屌乜鳩？邊隻股票又跌到仆直？ 😂",
        "好痛": "買咗咩？止蝕啦老細！唔止蝕就係止命㗎喇 💀",
        "all in": "ALL IN？！你係咪飲大咗？定係 Elon Musk 徒弟？ 🤡",
        "allin": "ALL IN？！你係咪飲大咗？定係 Elon Musk 徒弟？ 🤡",
        "撈底": "撈底？你確定係底定係地獄第十八層？ 😈",
        "財自": "財自？發夢就有。乖乖地做 research 先啦 😤",
        "fomo": "FOMO 入場 = 高位接貨 = 坐艇 = 喊。完。 📉",
        "hold": "Hold 到死？Diamond hands 定係傻的嗎 hands？ 💎🙌",
        "to the moon": "TO THE MOON！不過多數係去咗月球背面 😂",
        "沽空": "沽空？勇！你係下一個 Michael Burry 定係破產嗰個？",
        "牛熊": "玩牛熊？等同賭大細，不過你多數開細 🎲",
        "傻的嗎": "你先傻的嗎！不過你個倉又真係幾傻 😂",
        "on9": "On9 嘅唔係個市，係你個入場位 🫠",
        "燈": "你話人燈？你個倉仲紅過交通燈喎 🚦😂",
        "止蝕": "終於捨得止蝕？好事嚟，大個仔喇 😎",
    }

    for kw, response in KEYWORDS.items():
        if kw in content:
            # 20% chance to add a random zinger
            if random.random() < 0.2:
                response += f"\n\n_{random.choice(DOU_LAM_ZINGERS)}_"
            reply = await message.reply(response, mention_author=False)
            # Add a reaction to the user's message
            try:
                await message.add_reaction("🔥")
            except Exception:
                pass
            break


# ══════════════════════════════════════════════
# /DAU — 豆撚金句
# ══════════════════════════════════════════════

@bot.tree.command(name="dau", description="豆撚寶醒你一句金句")
async def dau_cmd(interaction: discord.Interaction):
    zinger = random.choice(DOU_LAM_ZINGERS)
    await interaction.response.send_message(
        f"💬 **豆撚寶金句：**\n>>> {zinger}\n\n_聽唔明就諗多幾次啦老細 😎_"
    )


# ══════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════

if __name__ == "__main__":
    if not TOKEN:
        print("❌ DISCORD_BOT_TOKEN 未 set！")
        print("   方法 1: 喺 .env file 加 `DISCORD_BOT_TOKEN=your_token_here`")
        print("   方法 2: set DISCORD_BOT_TOKEN 環境變數")
        exit(1)

    print("🚀 豆撚寶交易台啟動中…")
    print("   「Trade 得有態度，輸都要輸得型！」")
    print(f"   Connected channels: {', '.join(TRADING_CHANNELS)}")
    bot.run(TOKEN)
