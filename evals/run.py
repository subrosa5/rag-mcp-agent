"""run.py — eval-харнес: golden set → метрики по слоям → отчёт → гейт для CI.

Без этого проект — игрушка ("вроде отвечает"). С ним любое изменение
(чанкинг, модель, промпт) превращается в таблицу "было → стало", а CI
не даёт смёржить то, что ухудшило качество. ragas не тянем: тяжёлая
зависимость, по умолчанию требует OpenAI-ключ, и прячет внутри то, что
здесь видно явно (метрики — в evals/metrics.py, ~150 строк).

Запуск:
    uv run python -m evals.run                         # RAG-пайплайн, весь golden set
    uv run python -m evals.run --pipeline full         # guardrail → rewriter → agent → critic
    uv run python -m evals.run --category multi_hop --limit 5
    uv run python -m evals.run --gate                  # exit 1, если метрика ниже порога (для CI)
    uv run python -m evals.run --save-baseline         # зафиксировать текущие цифры как точку отсчёта
    uv run python -m evals.run --resume evals/reports/<прерванный>.jsonl   # продолжить с места остановки
    uv run python -m evals.run --rejudge evals/reports/<отчёт>.jsonl       # переоценить судьёй, не генерируя заново

Строки отчёта пишутся в .jsonl по одной, сразу после каждого вопроса:
бесплатный тариф Groq ограничен 200K токенов в день на модель, и прогон
полного пайплайна (~5 вызовов на вопрос) упирается в лимит посреди набора.
Раньше это роняло процесс и теряло весь прогон; теперь дневной лимит
останавливает прогон аккуратно, а --resume продолжает с того же места.

Retrieval-метрики считаются по hybrid_retrieve(вопрос) напрямую — это
оценка СЛОЯ поиска, одинаковая для всех пайплайнов. Генерация оценивается
по тому, что реально ответил выбранный пайплайн.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import groq

from evals import metrics
from rag_agent.agents.main_agent import run_agent_verbose
from rag_agent.agents.orchestrator import run_pipeline
from rag_agent.generation.rag import answer as rag_answer
from rag_agent.generation.rag import format_context
from rag_agent.observability import tracing
from rag_agent.retrieval.hybrid import hybrid_retrieve

ROOT = Path(__file__).parent
DATASET = ROOT / "datasets" / "golden.jsonl"
REPORTS = ROOT / "reports"
THRESHOLDS = ROOT / "thresholds.json"

# метрики, по которым строится гейт и сравнение с baseline (больше = лучше)
SUMMARY_METRICS = ["hit_at_k", "recall_at_k", "mrr", "key_fact_recall", "faithfulness", "correctness"]


def load_dataset(category: str | None, limit: int | None, ids: list[str] | None) -> list[dict]:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if category:
        rows = [r for r in rows if r["category"] == category]
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    return rows[:limit] if limit else rows


def _trace_context(trace: list[dict]) -> str:
    """Для агентных пайплайнов контекст — это то, что вернули инструменты:
    именно на этом агент обязан основывать ответ."""
    return "\n\n".join(step["result"] for step in trace) if trace else ""


def run_one(item: dict, pipeline: str) -> dict:
    question = item["question"]
    expected = item["expected_sources"]

    t0 = time.perf_counter()
    chunks = hybrid_retrieve(question)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    retrieved = list(dict.fromkeys(c["source"] for c in chunks))  # порядок ранжирования, без дублей

    t0 = time.perf_counter()
    if pipeline == "rag":
        out = rag_answer(question, chunks=chunks)
        answer, context, extra = out["answer"], format_context(chunks), {}
    elif pipeline == "agent":
        tracing.reset()
        out = run_agent_verbose(question, verbose=False)
        answer, context, extra = out["answer"], _trace_context(out["trace"]), {"tools": [s["tool"] for s in out["trace"]]}
    else:
        out = run_pipeline(question, verbose=False)
        answer, context = out["answer"], _trace_context(out["trace"])
        extra = {"tools": [s["tool"] for s in out["trace"]], "blocked": out["blocked"], "revisions": out["revisions"]}
    pipeline_ms = (time.perf_counter() - t0) * 1000
    stats = tracing.get_stats()

    row = {
        "id": item["id"],
        "category": item["category"],
        "question": question,
        "answer": answer,
        "retrieved_sources": retrieved,
        "expected_sources": expected,
        # у ловушек без ответа нет "правильного" документа — retrieval-метрики N/A
        "hit_at_k": metrics.hit_at_k(expected, retrieved) if expected else None,
        "recall_at_k": metrics.recall_at_k(expected, retrieved) if expected else None,
        "mrr": metrics.reciprocal_rank(expected, retrieved) if expected else None,
        "key_fact_recall": metrics.key_fact_recall(item["key_facts"], answer),
        "reference_answer": item["reference_answer"],
        "context": context,
        "retrieval_ms": round(retrieval_ms, 1),
        "pipeline_ms": round(pipeline_ms, 1),
        "llm_calls": stats["total_llm_calls"],
        "tokens": stats["total_tokens"],
        "cost_usd": stats["total_cost_usd"],
        **extra,
    }
    return judge(row)


def judge(row: dict) -> dict:
    """Вердикты судьи — отдельно от генерации, чтобы --rejudge мог переоценить
    сохранённые ответы новым промптом/моделью судьи без повторной генерации."""
    faith = metrics.judge_faithfulness(row["context"], row["answer"])
    correct = metrics.judge_correctness(row["question"], row["reference_answer"], row["answer"])
    return row | {
        "faithfulness": 1.0 if faith["verdict"] == "PASS" else 0.0,
        "faithfulness_reason": faith["reason"],
        "correctness": metrics.CORRECTNESS_SCORE.get(correct["verdict"], 0.0),
        "correctness_verdict": correct["verdict"],
        "correctness_reason": correct["reason"],
    }


def _is_daily_quota(err: groq.RateLimitError) -> bool:
    """Минутный лимит SDK переживает сам (ретраи в llm.py); дневной — нет,
    ждать его сброса часами внутри прогона бессмысленно."""
    return "per day" in str(err)


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def summarize(rows: list[dict]) -> dict:
    summary: dict = {m: _mean([r[m] for r in rows]) for m in SUMMARY_METRICS}
    latencies = sorted(r["retrieval_ms"] + r["pipeline_ms"] for r in rows)
    summary["latency_p50_ms"] = round(statistics.median(latencies), 1)
    summary["latency_p95_ms"] = round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 1)
    summary["cost_per_query_usd"] = round(statistics.fmean(r["cost_usd"] for r in rows), 6)
    summary["llm_calls_per_query"] = round(statistics.fmean(r["llm_calls"] for r in rows), 2)
    summary["n"] = len(rows)
    summary["by_category"] = {
        cat: {m: _mean([r[m] for r in rows if r["category"] == cat]) for m in SUMMARY_METRICS}
        | {"n": sum(r["category"] == cat for r in rows)}
        for cat in sorted({r["category"] for r in rows})
    }
    return summary


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def print_report(summary: dict, baseline: dict | None) -> None:
    print("\n| метрика | значение | baseline | Δ |\n|---|---|---|---|")
    for m in SUMMARY_METRICS:
        cur, base = summary[m], (baseline or {}).get(m)
        delta = f"{cur - base:+.2f}" if cur is not None and base is not None else ""
        print(f"| {m} | {_fmt(cur)} | {_fmt(base)} | {delta} |")
    print(f"\nn={summary['n']}  latency p50={summary['latency_p50_ms']:.0f}ms p95={summary['latency_p95_ms']:.0f}ms  "
          f"cost/query=${summary['cost_per_query_usd']:.5f}  llm calls/query={summary['llm_calls_per_query']}")
    print("\n| категория | n | " + " | ".join(SUMMARY_METRICS) + " |")
    print("|---|---|" + "---|" * len(SUMMARY_METRICS))
    for cat, vals in summary["by_category"].items():
        print(f"| {cat} | {vals['n']} | " + " | ".join(_fmt(vals[m]) for m in SUMMARY_METRICS) + " |")


def check_gate(summary: dict, pipeline: str) -> list[str]:
    thresholds = json.loads(THRESHOLDS.read_text()).get(pipeline, {})
    failures = []
    for metric, minimum in thresholds.items():
        value = summary.get(metric)
        if value is None or value < minimum:
            failures.append(f"{metric}={_fmt(value)} < {minimum}")
    return failures


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", choices=["rag", "agent", "full"], default="rag")
    ap.add_argument("--category")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--gate", action="store_true", help="exit 1, если метрика ниже порога из thresholds.json")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--resume", type=Path, help="продолжить прерванный прогон (его .jsonl)")
    ap.add_argument("--rejudge", type=Path, help="переоценить ответы из .jsonl судьёй, без генерации")
    args = ap.parse_args()

    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    complete = True

    if args.rejudge:
        source = _load_rows(args.rejudge)
        pipeline = args.rejudge.stem.split("_", 1)[1].removesuffix(".rejudged")
        report_path = REPORTS / f"{stamp}_{pipeline}.rejudged.jsonl"
        rows = []
        with report_path.open("w") as out:
            for i, old in enumerate(source, 1):
                try:
                    row = judge(old)
                except groq.RateLimitError as e:
                    if not _is_daily_quota(e):
                        raise
                    complete = False
                    print(f"\nдневной лимит судьи исчерпан на {old['id']} — переоценка остановлена")
                    break
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(source)}] {old['correctness_verdict']} → {row['correctness_verdict']}  {row['id']}", flush=True)
    else:
        pipeline = args.pipeline
        items = load_dataset(args.category, args.limit, args.ids)
        if args.resume:
            report_path = args.resume
            pipeline = report_path.stem.split("_", 1)[1]
            rows = _load_rows(report_path)
            done = {r["id"] for r in rows}
            items = [it for it in items if it["id"] not in done]
            print(f"продолжаю {report_path}: готово {len(done)}, осталось {len(items)}")
        else:
            report_path = REPORTS / f"{stamp}_{pipeline}.jsonl"
            rows = []
        with report_path.open("a") as out:
            for i, item in enumerate(items, 1):
                try:
                    row = run_one(item, pipeline)
                except groq.RateLimitError as e:
                    if not _is_daily_quota(e):
                        raise
                    complete = False
                    print(f"\nдневной лимит Groq исчерпан на {item['id']} — прогон остановлен.\n"
                          f"продолжить после сброса лимита: uv run python -m evals.run --resume {report_path}")
                    break
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                mark = {1.0: "✓", 0.5: "~", 0.0: "✗"}[row["correctness"]]
                print(f"[{i}/{len(items)}] {mark} {item['id']} {item['category']:<12} {item['question'][:60]}", flush=True)

    if not rows:
        print("нет ни одной строки для отчёта")
        return 1
    summary = summarize(rows)
    baseline_path = ROOT / f"baseline_{pipeline}.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    print_report(summary, baseline)
    print(f"\nотчёт: {report_path}" + ("" if complete else "  (ЧАСТИЧНЫЙ)"))

    if args.save_baseline:
        if complete:
            baseline_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            print(f"baseline сохранён: {baseline_path}")
        else:
            print("baseline НЕ сохранён: прогон неполный")

    if args.gate:
        failures = check_gate(summary, pipeline)
        if not complete:
            failures.append("прогон неполный (дневной лимит LLM)")
        if failures:
            print("\nGATE FAILED:\n  " + "\n  ".join(failures))
            return 1
        print("\nGATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
