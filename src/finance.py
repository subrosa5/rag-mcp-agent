"""finance.py — get_stock_quote: рыночная котировка без API-ключа.

Источник — Yahoo Finance chart endpoint (тот же, на котором построена
популярная библиотека yfinance): не требует ключа/регистрации, поэтому
и подошёл сюда — вы не хотели выдавать свои платные ключи (OpenAI/
Anthropic) под демо-фичу, а это ничего не просит взамен.

Известное ограничение: Yahoo иногда блокирует запросы с IP дата-центров/
облаков (anti-scraping) — с обычной сети (дома/офис) работает без проблем,
но с облачного раннера (Render и т.п.) может отвечать отказом. Для реального
прода тут был бы платный провайдер с SLA (Finnhub, Alpha Vantage,
Twelve Data) — это тот же trade-off, что уже есть в проекте между local и
Cohere эмбеддингами (config.py): бесплатно-и-неформально vs
платно-но-надёжно.
"""

import json
import math
import re
import urllib.error
import urllib.request

_QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def get_stock_quote(ticker: str) -> dict:
    """Текущая цена и дневное изменение по биржевому тикеру (например AAPL,
    TSLA, MSFT). Возвращает {"error": ...}, если тикер не найден или Yahoo
    недоступен — агент должен уметь честно сказать "не смог получить
    котировку", а не выдумать число."""
    if not isinstance(ticker, str) or not re.fullmatch(r"[A-Za-z0-9.^=-]{1,16}", ticker.strip()):
        return {"error": "Invalid ticker"}
    ticker = ticker.strip().upper()
    req = urllib.request.Request(
        _QUOTE_URL.format(ticker=ticker),
        headers={"User-Agent": "Mozilla/5.0"},  # без UA Yahoo сразу отдаёт 429
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, ValueError):
        return {"error": f"Не удалось получить котировку {ticker}"}

    try:
        result = data["chart"]["result"][0]
        meta = result["meta"]
        price = meta["regularMarketPrice"]
        prev_close = meta["chartPreviousClose"]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (price, prev_close)):
            return {"error": "Invalid price"}
        if not isinstance(meta.get("regularMarketTime"), int):
            return {"error": "Quote timestamp missing"}
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0
        return {
            "ticker": ticker,
            "price": price,
            "currency": meta.get("currency"),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "market_time_unix": meta.get("regularMarketTime"),
        }
    except (KeyError, IndexError, TypeError, ZeroDivisionError):
        return {"error": f"Тикер {ticker} не найден или Yahoo Finance вернул неожиданный формат"}


if __name__ == "__main__":
    t = input("Тикер: ")
    print(get_stock_quote(t))
