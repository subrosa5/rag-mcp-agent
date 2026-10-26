"""judge_calibration.py — насколько LLM-судье можно верить.

Цифра "faithfulness 0.92" ничего не стоит, если судья сам ошибается в каждом
пятом случае. Единственный честный способ это узнать — разметить часть
ответов руками и посчитать согласие человека с судьёй.

Шаг 1 — выгрузить выборку из отчёта на разметку:
    uv run python -m evals.judge_calibration export evals/reports/<отчёт>.json --n 40

    В файле evals/datasets/judge_labels.jsonl заполнить руками поля
    human_faithfulness ("PASS"/"FAIL") и human_correctness ("CORRECT"/"PARTIAL"/"WRONG").
    Вердиктов судьи в файле НЕТ нарочно — иначе разметчик подсознательно
    с ними соглашается (anchoring), и калибровка меряет сама себя.

Шаг 2 — посчитать согласие:
    uv run python -m evals.judge_calibration score

Что смотреть: accuracy (доля совпадений) и Cohen's kappa — согласие с
поправкой на случайное. Kappa < 0.6 — судье нельзя доверять без доработки
промпта или смены модели; 0.6–0.8 — приемлемо; > 0.8 — хорошо.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

LABELS = Path(__file__).parent / "datasets" / "judge_labels.jsonl"
FIELDS = {
    "faithfulness": ("judge_faithfulness", "human_faithfulness"),
    "correctness": ("judge_correctness", "human_correctness"),
}


def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def export(report_path: Path, n: int, seed: int) -> None:
    report = json.loads(report_path.read_text())
    rows = report["rows"]
    # стратифицируем по вердикту судьи: если взять случайные 40 из выборки,
    # где судья почти всё считает PASS, FAIL'ов на проверку не останется вовсе
    rng = random.Random(seed)  # noqa: S311  # nosec B311 — выборка для разметки, не криптография
    by_verdict: dict[str, list[dict]] = {}
    for r in rows:
        by_verdict.setdefault(r["correctness_verdict"], []).append(r)
    sample: list[dict] = []
    while len(sample) < min(n, len(rows)):
        for bucket in by_verdict.values():
            if bucket and len(sample) < n:
                sample.append(bucket.pop(rng.randrange(len(bucket))))
    with LABELS.open("w") as f:
        for r in sample:
            f.write(json.dumps({
                "report": str(report_path), "id": r["id"], "question": r["question"],
                "context": r["context"], "answer": r["answer"],
                "human_faithfulness": None, "human_correctness": None,
            }, ensure_ascii=False) + "\n")
    print(f"{len(sample)} примеров → {LABELS}. Заполни human_* поля и запусти `score`.")


def score() -> None:
    labels = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]
    reports: dict[str, dict] = {}
    for lab in labels:
        if lab["report"] not in reports:
            rows = json.loads(Path(lab["report"]).read_text())["rows"]
            reports[lab["report"]] = {r["id"]: r for r in rows}
        row = reports[lab["report"]][lab["id"]]
        lab["judge_faithfulness"] = "PASS" if row["faithfulness"] == 1.0 else "FAIL"
        lab["judge_correctness"] = row["correctness_verdict"]

    for metric, (judge_key, human_key) in FIELDS.items():
        pairs = [(lab[judge_key], lab[human_key]) for lab in labels if lab[human_key]]
        if not pairs:
            print(f"{metric}: нет ручной разметки")
            continue
        judge, human = zip(*pairs, strict=True)
        acc = sum(j == h for j, h in pairs) / len(pairs)
        print(f"\n{metric}: n={len(pairs)}  accuracy={acc:.2f}  kappa={cohens_kappa(list(judge), list(human)):.2f}")
        print("  расхождения (судья → человек):", dict(Counter(f"{j}→{h}" for j, h in pairs if j != h)) or "нет")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("report", type=Path)
    ex.add_argument("--n", type=int, default=40)
    ex.add_argument("--seed", type=int, default=0)
    sub.add_parser("score")
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.report, args.n, args.seed)
    else:
        score()


if __name__ == "__main__":
    main()
