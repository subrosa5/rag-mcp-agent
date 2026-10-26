import pytest

from rag_agent.tools import TOOL_FUNCTIONS, TOOLS_SCHEMA, finance
from rag_agent.tools.calculator import calculate


class TestCalculator:
    @pytest.mark.parametrize("expr, expected", [
        ("2 + 2", "4"),
        ("12 * (3 + 4)", "84"),
        ("-5 + 3", "-2"),
        ("2 ** 10", "1024"),
        ("7 / 2", "3.5"),
    ])
    def test_arithmetic(self, expr, expected):
        assert calculate(expr) == expected

    @pytest.mark.parametrize("expr", [
        "__import__('os').system('id')",
        "open('/etc/passwd').read()",
        "(1).__class__",
        "[x for x in range(10)]",
        "a + 1",
        "lambda: 1",
    ])
    def test_rejects_anything_but_arithmetic(self, expr):
        # главный инвариант безопасности: строка от LLM никогда не выполняется как код
        assert calculate(expr).startswith("Ошибка")

    @pytest.mark.parametrize("expr", [
        "9**9**9**9",          # огромный показатель
        "10**1001",            # показатель выше лимита
        "((10**50+1)**2)**2",  # повторное возведение: основание растёт на каждом шаге
    ])
    def test_resource_exhaustion_is_bounded(self, expr):
        assert calculate(expr).startswith("Ошибка")

    def test_length_limit(self):
        assert calculate("1+" * 150 + "1").startswith("Ошибка")

    def test_division_by_zero_is_error_not_crash(self):
        assert calculate("1/0").startswith("Ошибка")


class TestFinance:
    @pytest.mark.parametrize("ticker", [
        "AAPL/../../v7/finance/quote",
        "AAPL?callback=x",
        "AAPL#frag",
        "AA PL",
        "",
        "A" * 30,
    ])
    def test_malformed_ticker_never_reaches_network(self, ticker, monkeypatch):
        def no_network(*a, **kw):
            raise AssertionError("сетевой вызов с непроверенным тикером")
        monkeypatch.setattr(finance.urllib.request, "urlopen", no_network)
        assert "error" in finance.get_stock_quote(ticker)

    @pytest.mark.parametrize("ticker", ["aapl", "BRK-B", "SBER.ME", "^GSPC", "EURUSD=X"])
    def test_valid_tickers_pass_validation(self, ticker, monkeypatch):
        seen = {}

        def fake_urlopen(req, timeout):
            seen["url"] = req.full_url
            raise finance.urllib.error.URLError("offline")
        monkeypatch.setattr(finance.urllib.request, "urlopen", fake_urlopen)
        finance.get_stock_quote(ticker)
        assert seen["url"].startswith("https://query1.finance.yahoo.com/v8/finance/chart/")
        assert seen["url"].endswith(ticker.upper())


def test_every_schema_entry_has_an_implementation():
    names = {t["function"]["name"] for t in TOOLS_SCHEMA}
    assert names == set(TOOL_FUNCTIONS)
