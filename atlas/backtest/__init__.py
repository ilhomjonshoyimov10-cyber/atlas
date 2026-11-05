"""Professional backtest: faqat model ko'rmagan (out-of-sample) bashoratlar ustida, real xarajat, risk boshqaruvi
va statistik ishonchlilik testlari bilan. Batafsil: engine.py (savdo simulyatsiyasi), metrics.py (ko'rsatkichlar),
stats.py (bootstrap, PSR/DSR, tasodifiy signal testi), export.py (sayt va API uchun hisobot)."""
from .engine import STRATEGIES, Market, load_market, simulate  # noqa: F401
from .export import COSTS, THRESHOLDS, build_report, clean, robustness  # noqa: F401
