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
import re
import urllib.error
import urllib.request

_QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# Тикер приходит от LLM (аргумент tool_call), а значит в конечном счёте —
# от пользователя или от текста документа (indirect injection). Без
# проверки он подставлялся в URL как есть: "AAPL/../../v7/finance/..." или
# "AAPL?callback=..." уводили запрос на другой путь того же хоста. Хост и
# схема зашиты в _QUOTE_URL, так что это не полноценный SSRF, но принцип
# тот же: аргумент инструмента — недоверенный ввод. Разрешаем только формат
# реальных тикеров Yahoo: AAPL, BRK-B, SBER.ME, ^GSPC, EURUSD=X.
_TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")


def get_stock_quote(ticker: str) -> dict:
    """Текущая цена и дневное изменение по биржевому тикеру (например AAPL,
    TSLA, MSFT). Возвращает {"error": ...}, если тикер не найден или Yahoo
    недоступен — агент должен уметь честно сказать "не смог получить
    котировку", а не выдумать число."""
    ticker = ticker.strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        return {"error": f"Некорректный тикер: {ticker[:20]!r}"}
    # noqa ниже: S310 предупреждает о file:// и прочих схемах в URL — здесь
    # схема зашита (https), а подставляемая часть проверена регуляркой выше.
    req = urllib.request.Request(  # noqa: S310  # nosec B310
        _QUOTE_URL.format(ticker=ticker),
        headers={"User-Agent": "Mozilla/5.0"},  # без UA Yahoo сразу отдаёт 429
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310  # nosec B310
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": f"Не удалось получить котировку {ticker}: {e}"}

    try:
        result = data["chart"]["result"][0]
        meta = result["meta"]
        price = meta["regularMarketPrice"]
        prev_close = meta["chartPreviousClose"]
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
