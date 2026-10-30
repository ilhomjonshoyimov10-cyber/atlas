import { useEffect, useRef } from "react";

/** TradingView'ning bepul "Advanced Real-Time Chart" vidjeti — TradingView ma'lumotlari bilan jonli grafik,
 *  indikatorlar va chizish asboblari. Belgilar TradingView nomlashiga moslangan (spot/CFD; ATLAS esa
 *  Yahoo fyucherslaridan foydalanadi, shuning uchun narx bir oz farq qilishi mumkin). */
export const TV_SYMBOLS: Record<string, string> = {
  EURUSD: "FX:EURUSD", GBPUSD: "FX:GBPUSD", USDJPY: "FX:USDJPY", AUDUSD: "FX:AUDUSD", USDCHF: "FX:USDCHF",
  USDCAD: "FX:USDCAD", DXY: "TVC:DXY",
  BTC: "BINANCE:BTCUSDT", ETH: "BINANCE:ETHUSDT", BNB: "BINANCE:BNBUSDT", XRP: "BINANCE:XRPUSDT", SOL: "BINANCE:SOLUSDT",
  SPX: "FOREXCOM:SPXUSD", NDX: "NASDAQ:IXIC", DJI: "FOREXCOM:DJI",
  AAPL: "NASDAQ:AAPL", MSFT: "NASDAQ:MSFT", NVDA: "NASDAQ:NVDA", AMZN: "NASDAQ:AMZN", TSLA: "NASDAQ:TSLA",
  GOOGL: "NASDAQ:GOOGL", META: "NASDAQ:META", JPM: "NYSE:JPM", XOM: "NYSE:XOM",
  CL: "TVC:USOIL", NG: "OANDA:NATGASUSD", WHEAT: "OANDA:WHEATUSD", CORN: "OANDA:CORNUSD", SOY: "OANDA:SOYBNUSD",
  COFFEE: "ICEUS:KC1!", SUGAR: "OANDA:SUGARUSD", COCOA: "ICEUS:CC1!", COTTON: "ICEUS:CT1!",
  GOLD: "OANDA:XAUUSD", SILVER: "OANDA:XAGUSD", COPPER: "OANDA:XCUUSD", US10Y: "TVC:US10Y",
};

export default function TradingViewWidget({ symbol }: { symbol: string }) {
  const box = useRef<HTMLDivElement>(null);
  const dark = typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.innerHTML = '<div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>';
    const s = document.createElement("script");
    s.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    s.async = true;
    s.type = "text/javascript";
    s.innerHTML = JSON.stringify({
      autosize: true, symbol: TV_SYMBOLS[symbol] ?? symbol, interval: "D", timezone: "Asia/Tashkent",
      theme: dark ? "dark" : "light", style: "1", locale: "ru", allow_symbol_change: true,
      hide_side_toolbar: false, withdateranges: true, calendar: false, support_host: "https://www.tradingview.com",
    });
    el.appendChild(s);
    return () => { el.innerHTML = ""; };
  }, [symbol, dark]);
  return (
    <div className="tv-wrap">
      {/* vidjet konteynerga "height:100%" yozib qo'yadi — shuning uchun balandlik tashqi qobiqda beriladi */}
      <div className="tv-frame"><div ref={box} className="tradingview-widget-container tv-box" /></div>
      <p className="hint">Jonli grafik va asboblar: <a href={`https://www.tradingview.com/symbols/${(TV_SYMBOLS[symbol] ?? symbol).replace(":", "-")}/`}
        target="_blank" rel="noreferrer">TradingView</a>. Bu grafik TradingView ma'lumotlari ({TV_SYMBOLS[symbol] ?? symbol}) asosida;
        ATLAS prognozi va texnik darajalar "Shamlar" rejimida ko'rinadi.</p>
    </div>
  );
}
