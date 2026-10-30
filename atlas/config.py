"""Loyiha konfiguratsiyasi: aktivlar, yangilik manbalari, yo'llar, model giperparametrlari."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("ATLAS_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.environ.get("ATLAS_DATA", ROOT / "data"))
PRICES_DIR = DATA_DIR / "prices"
NEWS_DIR = DATA_DIR / "news"
EMB_DIR = DATA_DIR / "embeddings"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
NEWS_DB = DATA_DIR / "news.sqlite"

for _d in (PRICES_DIR, NEWS_DIR, EMB_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Asset:
    symbol: str          # ichki nom (URL-da ishlatiladi)
    yf: str              # Yahoo Finance tikeri
    name: str            # ko'rsatiladigan nom (o'zbekcha)
    group: str           # forex | crypto | equity | index | energy | agri | metal | rates
    binance: str | None = None  # kripto real-time uchun
    keywords: tuple[str, ...] = ()


# Yangilik -> aktiv bog'lanishi uchun kalit so'zlar (inglizcha manbalar uchun)
ASSETS: list[Asset] = [
    # Forex
    Asset("EURUSD", "EURUSD=X", "EUR/USD", "forex", keywords=("euro", "ecb", "eurozone", "lagarde", "eur/usd")),
    Asset("GBPUSD", "GBPUSD=X", "GBP/USD", "forex", keywords=("pound", "sterling", "bank of england", "boe", "gilt", "uk inflation")),
    Asset("USDJPY", "USDJPY=X", "USD/JPY", "forex", keywords=("yen", "bank of japan", "boj", "japan")),
    Asset("AUDUSD", "AUDUSD=X", "AUD/USD", "forex", keywords=("aussie", "rba", "australia", "iron ore")),
    Asset("USDCHF", "USDCHF=X", "USD/CHF", "forex", keywords=("franc", "snb", "swiss")),
    Asset("USDCAD", "USDCAD=X", "USD/CAD", "forex", keywords=("loonie", "bank of canada", "canada")),
    Asset("DXY", "DX-Y.NYB", "Dollar indeksi", "forex", keywords=("dollar", "greenback", "fed", "fomc", "powell", "treasury")),
    # Kripto
    Asset("BTC", "BTC-USD", "Bitcoin", "crypto", "btcusdt", ("bitcoin", "btc", "crypto", "etf", "halving", "sec")),
    Asset("ETH", "ETH-USD", "Ethereum", "crypto", "ethusdt", ("ethereum", "ether", "eth", "crypto", "defi")),
    Asset("BNB", "BNB-USD", "BNB", "crypto", "bnbusdt", ("binance", "bnb", "crypto")),
    Asset("XRP", "XRP-USD", "XRP", "crypto", "xrpusdt", ("ripple", "xrp", "crypto")),
    Asset("SOL", "SOL-USD", "Solana", "crypto", "solusdt", ("solana", "sol", "crypto")),
    # Indekslar va aksiyalar
    Asset("SPX", "^GSPC", "S&P 500", "index", keywords=("s&p", "stocks", "wall street", "equities", "fed", "earnings")),
    Asset("NDX", "^IXIC", "Nasdaq", "index", keywords=("nasdaq", "tech stocks", "ai", "chip")),
    Asset("DJI", "^DJI", "Dow Jones", "index", keywords=("dow", "industrials", "wall street")),
    Asset("AAPL", "AAPL", "Apple", "equity", keywords=("apple", "iphone", "tim cook")),
    Asset("MSFT", "MSFT", "Microsoft", "equity", keywords=("microsoft", "azure", "openai")),
    Asset("NVDA", "NVDA", "Nvidia", "equity", keywords=("nvidia", "gpu", "chip", "semiconductor", "ai")),
    Asset("AMZN", "AMZN", "Amazon", "equity", keywords=("amazon", "aws", "retail")),
    Asset("TSLA", "TSLA", "Tesla", "equity", keywords=("tesla", "musk", "electric vehicle", "ev")),
    Asset("GOOGL", "GOOGL", "Alphabet", "equity", keywords=("google", "alphabet", "youtube")),
    Asset("META", "META", "Meta", "equity", keywords=("meta", "facebook", "instagram", "zuckerberg")),
    Asset("JPM", "JPM", "JPMorgan", "equity", keywords=("jpmorgan", "bank", "banks", "dimon")),
    Asset("XOM", "XOM", "ExxonMobil", "equity", keywords=("exxon", "oil major")),
    # Energiya
    Asset("CL", "CL=F", "Neft (WTI)", "energy", keywords=("oil", "crude", "opec", "wti", "brent", "eia", "refinery", "saudi")),
    Asset("NG", "NG=F", "Tabiiy gaz", "energy", keywords=("natural gas", "lng", "gas prices", "heating")),
    # Qishloq xo'jaligi
    Asset("WHEAT", "ZW=F", "Bug'doy", "agri", keywords=("wheat", "grain", "harvest", "usda", "drought", "ukraine", "black sea", "wasde")),
    Asset("CORN", "ZC=F", "Makkajo'xori", "agri", keywords=("corn", "maize", "ethanol", "usda", "wasde", "grain")),
    Asset("SOY", "ZS=F", "Soya", "agri", keywords=("soybean", "soy", "china imports", "usda", "brazil")),
    Asset("COFFEE", "KC=F", "Qahva", "agri", keywords=("coffee", "arabica", "brazil frost")),
    Asset("SUGAR", "SB=F", "Shakar", "agri", keywords=("sugar", "cane", "india monsoon")),
    Asset("COCOA", "CC=F", "Kakao", "agri", keywords=("cocoa", "ivory coast", "ghana")),
    Asset("COTTON", "CT=F", "Paxta", "agri", keywords=("cotton", "textile")),
    # Metallar
    Asset("GOLD", "GC=F", "Oltin", "metal", keywords=("gold", "bullion", "safe haven", "central bank buying", "geopolitical")),
    Asset("SILVER", "SI=F", "Kumush", "metal", keywords=("silver", "precious metals")),
    Asset("COPPER", "HG=F", "Mis", "metal", keywords=("copper", "china demand", "mining")),
    # Foiz stavkalari
    Asset("US10Y", "^TNX", "AQSH 10Y obligatsiya", "rates", keywords=("treasury", "yield", "bond", "fed", "inflation")),
]

ASSET_BY_SYMBOL = {a.symbol: a for a in ASSETS}
GROUPS = ["forex", "crypto", "index", "equity", "energy", "agri", "metal", "rates"]

# Yangiliklarni birinchilardan chiqaradigan, ochiq RSS manbalari.
# Eslatma: ba'zi saytlar RSS'ni o'zgartirib turadi; kollektor xatoga chidamli.
NEWS_FEEDS: dict[str, str] = {
    # Rasmiy (eng tez va eng ishonchli makro manbalar)
    "fed_press": "https://www.federalreserve.gov/feeds/press_all.xml",
    "fed_monetary": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "ecb_press": "https://www.ecb.europa.eu/rss/press.html",
    "bls_releases": "https://www.bls.gov/feed/bls_latest.rss",
    "eia_today": "https://www.eia.gov/rss/todayinenergy.xml",
    "usda_news": "https://www.usda.gov/rss/latest-releases.xml",
    "sec_press": "https://www.sec.gov/news/pressreleases.rss",
    # Moliyaviy yangiliklar
    "cnbc_markets": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "cnbc_economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "marketwatch_top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "investing_news": "https://www.investing.com/rss/news.rss",
    "fxstreet_news": "https://www.fxstreet.com/rss/news",
    "investing_commod": "https://www.investing.com/rss/news_11.rss",
    "investing_forex": "https://www.investing.com/rss/news_1.rss",
    # Kripto
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "theblock": "https://www.theblock.co/rss.xml",
    # Tovar bozorlari
    "oilprice": "https://oilprice.com/rss/main",
    "mw_markets": "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
}

# Voqea turlari (regex kalit so'zlar). Modelga alohida embedding sifatida beriladi.
EVENT_TYPES: dict[str, tuple[str, ...]] = {
    "none": (),
    "fomc": ("fomc", "federal open market committee", "rate decision", "fed holds", "fed raises", "fed cuts", "powell"),
    "cpi": ("cpi", "consumer price index", "inflation rate", "core inflation"),
    "nfp": ("nonfarm", "non-farm", "payrolls", "jobs report", "unemployment rate"),
    "gdp": ("gdp", "gross domestic product", "recession"),
    "ecb": ("ecb", "lagarde", "european central bank"),
    "earnings": ("earnings", "quarterly results", "revenue", "eps", "guidance"),
    "opec": ("opec", "output cut", "production cut", "oil supply"),
    "usda": ("usda", "wasde", "crop report", "harvest", "planting"),
    "geopolitics": ("war", "sanction", "missile", "invasion", "conflict", "tariff", "attack"),
    "crypto_reg": ("sec", "etf approval", "regulation", "ban", "lawsuit", "hack", "exploit"),
}
EVENT_NAMES = list(EVENT_TYPES)


@dataclass
class ModelConfig:
    lookback: int = 60               # narx shoxi: oxirgi 60 kun
    news_days: int = 7               # yangilik shoxi: oxirgi 7 kun
    horizons: tuple[int, ...] = (1, 5, 20)
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    emb_dim: int = 64                # FinBERT 768 -> PCA 64
    asset_emb: int = 16
    hidden: int = 64
    layers: int = 2                  # Transformer qatlamlari
    heads: int = 4                   # attention boshlari
    ensemble: int = 3                # seedlar soni (servis modeli)
    warmup_epochs: int = 1
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 512
    epochs: int = 30
    patience: int = 5
    use_news: bool = True
    use_season: bool = True
    use_market: bool = True
    seed: int = 42
    # Baholash
    holdout_days: int = 365          # oxirgi 1 yil (kalendar kun): tegilmagan final holdout
    wf_folds: int = 5
    wf_test_days: int = 182          # har bir fold testi ~6 oy (kalendar kun)
    extra: dict = field(default_factory=dict)


# Bozor yopilishi (decision_time) UTC bo'yicha. Kunlik yopilish narxi shu vaqtda
# ma'lum deb hisoblaymiz; shu vaqtdan KEYIN mavjud bo'lgan yangilik ishlatilmaydi.
DECISION_HOUR_UTC = 20  # 20:00 UTC ~ NYSE yopilishi (16:00 ET qishda 21:00, yozda 20:00) -> konservativ: 20:00
