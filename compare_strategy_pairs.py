import argparse
import csv
import html
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from daemon_coin import (
    BestPair,
    GAConfig,
    Strategy,
    choose_target_index,
    find_best_pairs,
    load_config_file,
    success_condition_from_name,
)


def pct(value: float) -> str:
    return f"{value:.2%}"


def strategy_label(strategy: dict) -> str:
    if strategy["strategy_type"] == "first-pattern":
        pattern = "".join(str(bit) for bit in strategy["pattern"])
        return (
            f"pattern {pattern}, occurrence {strategy.get('occurrence_number', 1)}, "
            f"offset {strategy['index_offset']}, fallback {strategy['fallback_index']}"
        )
    return (
        f"gene table, observed length {strategy['observed_length']}, "
        f"{len(strategy['genes'])} genes"
    )


def load_experiment_config(output_dir: Path) -> GAConfig:
    config_path = output_dir / "experiment_config.yaml"
    if not config_path.exists():
        return load_config_file(Path("config.toml"))

    values = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if not line or ": " not in line:
            continue
        key, raw_value = line.split(": ", 1)
        values[key] = json.loads(raw_value)
    return GAConfig(**values)


def latest_generation(output_dir: Path) -> int:
    snapshots_dir = output_dir / "population_snapshots"
    generations = []
    for path in snapshots_dir.glob("population_A_generation_*.json"):
        generations.append(int(path.stem.rsplit("_", 1)[1]))
    if not generations:
        raise ValueError(f"No population snapshots found in {snapshots_dir}")
    return max(generations)


def load_population(output_dir: Path, player_type: str, generation: int) -> list[Strategy]:
    path = (
        output_dir
        / "population_snapshots"
        / f"population_{player_type}_generation_{generation:04d}.json"
    )
    with path.open(encoding="utf-8") as population_file:
        return [Strategy(**strategy) for strategy in json.load(population_file)]


def load_score_matrix(output_dir: Path, generation: int) -> list[list[float]]:
    path = (
        output_dir
        / "score_matrices"
        / f"score_matrix_generation_{generation:04d}.csv"
    )
    with path.open(newline="", encoding="utf-8") as score_file:
        rows = list(csv.reader(score_file))
    return [[float(value) for value in row[1:]] for row in rows[1:]]


def strategy_by_id(population: list[Strategy], strategy_id: str) -> Strategy:
    for strategy in population:
        if strategy.id == strategy_id:
            return strategy
    raise ValueError(f"Strategy not found: {strategy_id}")


def observed_bits(strategy: Strategy, sequence: list[int]) -> list[int]:
    if strategy.observed_length == 0:
        return sequence
    return sequence[: strategy.observed_length]


def minimum_sequence_length(
    pairs: list[tuple[Strategy, Strategy]],
    configured_sequence_length: int,
) -> int:
    length = configured_sequence_length
    for strategy_a, strategy_b in pairs:
        length = max(
            length,
            strategy_a.observed_length,
            strategy_b.observed_length,
            strategy_a.target_choice_range,
            strategy_b.target_choice_range,
            strategy_a.fallback_index + 1,
            strategy_b.fallback_index + 1,
        )
    return length


def evaluate_pairs_on_shared_trials(
    pairs: list[tuple[Strategy, Strategy]],
    trials: int,
    sequence_length: int,
    success_condition_name: str,
    seed: int | None,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    rng = random.Random(seed)
    success_condition = success_condition_from_name(success_condition_name)
    wins = [0 for _pair in pairs]
    pairwise_difference_sums = [
        [0 for _pair in pairs]
        for _pair in pairs
    ]
    pairwise_difference_square_sums = [
        [0 for _pair in pairs]
        for _pair in pairs
    ]

    run_shared_trials_batch(
        pairs,
        trials,
        sequence_length,
        success_condition,
        rng,
        wins,
        pairwise_difference_sums,
        pairwise_difference_square_sums,
    )

    return wins, pairwise_difference_sums, pairwise_difference_square_sums


def run_shared_trials_batch(
    pairs: list[tuple[Strategy, Strategy]],
    trials: int,
    sequence_length: int,
    success_condition,
    rng: random.Random,
    wins: list[int],
    pairwise_difference_sums: list[list[int]],
    pairwise_difference_square_sums: list[list[int]],
) -> None:

    for _trial in range(trials):
        sequence_a = [rng.randrange(2) for _index in range(sequence_length)]
        sequence_b = [rng.randrange(2) for _index in range(sequence_length)]
        pair_results = []

        for pair_index, (strategy_a, strategy_b) in enumerate(pairs):
            target_in_b = choose_target_index(strategy_a, observed_bits(strategy_a, sequence_a))
            target_in_a = choose_target_index(strategy_b, observed_bits(strategy_b, sequence_b))
            survived = int(success_condition(sequence_a[target_in_a], sequence_b[target_in_b]))
            pair_results.append(survived)
            wins[pair_index] += survived

        for left_index, left_result in enumerate(pair_results):
            for right_index, right_result in enumerate(pair_results):
                difference = left_result - right_result
                pairwise_difference_sums[left_index][right_index] += difference
                pairwise_difference_square_sums[left_index][right_index] += (
                    difference * difference
                )


def interval_label(sigma: float) -> str:
    if sigma == 1.96:
        return "95% CI"
    return f"{sigma:g}-sigma interval"


def normal_ci(rate: float, trials: int, sigma: float = 1.96) -> tuple[float, float]:
    standard_error = math.sqrt(rate * (1 - rate) / trials)
    return rate - sigma * standard_error, rate + sigma * standard_error


def difference_ci(
    difference_sum: int,
    difference_square_sum: int,
    trials: int,
    sigma: float = 1.96,
) -> tuple[float, float, float]:
    mean_difference = difference_sum / trials
    mean_square = difference_square_sum / trials
    variance = max(mean_square - mean_difference * mean_difference, 0)
    standard_error = math.sqrt(variance / trials)
    return (
        mean_difference,
        mean_difference - sigma * standard_error,
        mean_difference + sigma * standard_error,
    )


def comparison_rows(
    pairs: list[BestPair],
    concrete_pairs: list[tuple[Strategy, Strategy]],
    wins: list[int],
    pairwise_difference_sums: list[list[int]],
    pairwise_difference_square_sums: list[list[int]],
    trials: int,
    sigma: float = 1.96,
) -> list[dict]:
    rows = []
    for index, (pair, (strategy_a, strategy_b), win_count) in enumerate(
        zip(pairs, concrete_pairs, wins),
        start=1,
    ):
        rate = win_count / trials
        ci_low, ci_high = normal_ci(rate, trials, sigma)
        pairwise = []
        for other_index, other_pair in enumerate(pairs):
            mean_diff, diff_low, diff_high = difference_ci(
                pairwise_difference_sums[index - 1][other_index],
                pairwise_difference_square_sums[index - 1][other_index],
                trials,
                sigma,
            )
            pairwise.append(
                {
                    "against_rank": other_index + 1,
                    "against_strategy_A": other_pair.strategy_A_id,
                    "against_strategy_B": other_pair.strategy_B_id,
                    "score_difference": mean_diff,
                    "ci95_low": diff_low,
                    "ci95_high": diff_high,
                    "sigma": sigma,
                    "statistically_better": diff_low > 0,
                    "statistically_worse": diff_high < 0,
                }
            )
        rows.append(
            {
                "rank": index,
                "strategy_A": pair.strategy_A_id,
                "strategy_B": pair.strategy_B_id,
                "ga_score": pair.score,
                "checked_score": rate,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "sigma": sigma,
                "wins": win_count,
                "trials": trials,
                "strategy_A_definition": asdict(strategy_a),
                "strategy_B_definition": asdict(strategy_b),
                "pairwise_comparisons": pairwise,
            }
        )
    return rows


def annotate_checked_ranks(rows: list[dict]) -> None:
    rows.sort(key=lambda row: row["checked_score"], reverse=True)
    for checked_rank, row in enumerate(rows, start=1):
        row["checked_rank"] = checked_rank


def pair_key(row: dict) -> tuple[str, str]:
    return row["strategy_A"], row["strategy_B"]


def comparison_against(row: dict, other: dict) -> dict:
    other_key = pair_key(other)
    for comparison in row["pairwise_comparisons"]:
        comparison_key = (
            comparison["against_strategy_A"],
            comparison["against_strategy_B"],
        )
        if comparison_key == other_key:
            return comparison
    raise ValueError("Pairwise comparison not found")


def row_is_statistically_better(row: dict, other: dict) -> bool:
    return comparison_against(row, other)["statistically_better"]


def annotate_certainty_groups(rows: list[dict]) -> None:
    if not rows:
        return

    group = 1
    rows[0]["certainty_group"] = group
    rows[0]["certainty_group_label"] = "Group 1"

    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]
        if row_is_statistically_better(previous, current):
            group += 1
        current["certainty_group"] = group
        current["certainty_group_label"] = f"Group {group}"


def certainty_ranking_is_complete(rows: list[dict]) -> bool:
    if len(rows) < 2:
        return True
    for index in range(len(rows) - 1):
        if not row_is_statistically_better(rows[index], rows[index + 1]):
            return False
    return True


def current_best_is_separated(rows: list[dict]) -> bool:
    if len(rows) < 2:
        return True
    best = rows[0]
    for comparison in best["pairwise_comparisons"]:
        if (
            comparison["against_strategy_A"] == best["strategy_A"]
            and comparison["against_strategy_B"] == best["strategy_B"]
        ):
            continue
        if not comparison["statistically_better"]:
            return False
    return True


def add_run_metadata(
    rows: list[dict],
    sequential: bool,
    stopping_reason: str,
    batches_completed: int,
) -> None:
    for row in rows:
        row["sequential"] = sequential
        row["stopping_reason"] = stopping_reason
        row["batches_completed"] = batches_completed


def sequential_comparison_rows(
    pairs: list[BestPair],
    concrete_pairs: list[tuple[Strategy, Strategy]],
    max_trials: int,
    batch_size: int,
    sequence_length: int,
    success_condition_name: str,
    seed: int | None,
    sigma: float = 1.96,
    progress: bool = False,
) -> list[dict]:
    rng = random.Random(seed)
    success_condition = success_condition_from_name(success_condition_name)
    wins = [0 for _pair in concrete_pairs]
    pairwise_difference_sums = [
        [0 for _pair in concrete_pairs]
        for _pair in concrete_pairs
    ]
    pairwise_difference_square_sums = [
        [0 for _pair in concrete_pairs]
        for _pair in concrete_pairs
    ]
    trials_completed = 0
    batches_completed = 0
    rows = []
    stopping_reason = "max_trials_reached"

    while trials_completed < max_trials:
        current_batch_size = min(batch_size, max_trials - trials_completed)
        run_shared_trials_batch(
            concrete_pairs,
            current_batch_size,
            sequence_length,
            success_condition,
            rng,
            wins,
            pairwise_difference_sums,
            pairwise_difference_square_sums,
        )
        trials_completed += current_batch_size
        batches_completed += 1
        rows = comparison_rows(
            pairs,
            concrete_pairs,
            wins,
            pairwise_difference_sums,
            pairwise_difference_square_sums,
            trials_completed,
            sigma,
        )
        annotate_checked_ranks(rows)
        annotate_certainty_groups(rows)
        if progress:
            print(
                "batch="
                f"{batches_completed} trials={trials_completed} "
                f"best={rows[0]['strategy_A']}+{rows[0]['strategy_B']} "
                f"score={rows[0]['checked_score']:.4%} "
                f"certainty_group={rows[0]['certainty_group']} "
                f"complete={certainty_ranking_is_complete(rows)}",
                flush=True,
            )
        if certainty_ranking_is_complete(rows):
            stopping_reason = "full_ranking_statistically_separated"
            break

    add_run_metadata(
        rows,
        sequential=True,
        stopping_reason=stopping_reason,
        batches_completed=batches_completed,
    )
    return rows


def write_outputs(output_dir: Path, rows: list[dict]) -> None:
    json_path = output_dir / "top_pair_statistical_check.json"
    csv_path = output_dir / "top_pair_statistical_check.csv"
    html_path = output_dir / "top_pair_statistical_check.html"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(rows, json_file, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "rank",
                "checked_rank",
                "certainty_group",
                "strategy_A",
                "strategy_B",
                "ga_score",
                "checked_score",
                "ci95_low",
                "ci95_high",
                "sigma",
                "wins",
                "trials",
                "sequential",
                "stopping_reason",
                "batches_completed",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row[key]
                    for key in writer.fieldnames
                }
            )

    html_path.write_text(render_html_report(rows), encoding="utf-8")


def render_pairwise_summary(row: dict) -> str:
    items = []
    for comparison in row["pairwise_comparisons"]:
        if (
            comparison["against_strategy_A"] == row["strategy_A"]
            and comparison["against_strategy_B"] == row["strategy_B"]
        ):
            continue
        if comparison["statistically_better"]:
            verdict = "statistically better"
        elif comparison["statistically_worse"]:
            verdict = "statistically worse"
        else:
            verdict = "not clearly different"
        label = (
            f"{comparison['against_strategy_A']} + "
            f"{comparison['against_strategy_B']}"
        )
        items.append(
            "<li>"
            f"vs <code>{html.escape(label)}</code>: "
            f"{html.escape(verdict)}; difference {pct(comparison['score_difference'])} "
            f"({interval_label(comparison.get('sigma', 1.96))} "
            f"{pct(comparison['ci95_low'])} to {pct(comparison['ci95_high'])})"
            "</li>"
        )
    return "\n".join(items)


def render_html_report(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("Cannot render an empty comparison report")
    for checked_rank, row in enumerate(rows, start=1):
        row.setdefault("checked_rank", checked_rank)
    annotate_certainty_groups(rows)
    trials = rows[0]["trials"]
    best = rows[0]
    label = interval_label(best.get("sigma", 1.96))
    sequential_text = ""
    if "sequential" in best:
        mode = "sequential" if best["sequential"] else "fixed trial"
        sequential_text = (
            f"<p>Mode: {html.escape(mode)}. "
            f"Stopping reason: {html.escape(best['stopping_reason'])}. "
            f"Batches completed: {best['batches_completed']}.</p>"
        )
    pair_rows = "\n".join(
        f"""
        <tr>
          <td>{row['checked_rank']}</td>
          <td>{row['certainty_group_label']}</td>
          <td><code>{html.escape(row['strategy_A'])}</code></td>
          <td><code>{html.escape(row['strategy_B'])}</code></td>
          <td>{pct(row['checked_score'])}</td>
          <td>{pct(row['ci95_low'])} to {pct(row['ci95_high'])}</td>
          <td>{row['rank']}</td>
          <td>{pct(row['ga_score'])}</td>
        </tr>
        """
        for row in rows
    )
    detail_sections = "\n".join(
        f"""
        <article>
          <h3>#{row['checked_rank']} {html.escape(row['strategy_A'])} + {html.escape(row['strategy_B'])}</h3>
          <p>
            Certainty rank: <strong>{html.escape(row['certainty_group_label'])}</strong>.
            Fresh-check score: <strong>{pct(row['checked_score'])}</strong>
            with {html.escape(interval_label(row.get('sigma', 1.96)))} {pct(row['ci95_low'])} to {pct(row['ci95_high'])}.
            Original GA rank {row['rank']} scored {pct(row['ga_score'])}.
          </p>
          <div class="grid">
            <div>
              <h4>Player A</h4>
              <p>{html.escape(strategy_label(row['strategy_A_definition']))}</p>
            </div>
            <div>
              <h4>Player B</h4>
              <p>{html.escape(strategy_label(row['strategy_B_definition']))}</p>
            </div>
          </div>
          <h4>Pairwise Checks</h4>
          <ul>
            {render_pairwise_summary(row)}
          </ul>
        </article>
        """
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Top Strategy Pair Statistical Check</title>
  <style>
    body {{
      color: #202124;
      font-family: Arial, sans-serif;
      line-height: 1.5;
      margin: 32px auto;
      max-width: 1100px;
      padding: 0 20px;
    }}
    h1, h2, h3, h4 {{ line-height: 1.2; }}
    code {{
      background: #f1f3f4;
      border-radius: 4px;
      padding: 2px 4px;
    }}
    section {{
      border-top: 1px solid #dadce0;
      margin-top: 28px;
      padding-top: 20px;
    }}
    table {{
      border-collapse: collapse;
      margin: 12px 0;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid #edf0f2;
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    article {{
      border: 1px solid #dadce0;
      border-radius: 8px;
      margin: 16px 0;
      padding: 16px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
  </style>
</head>
<body>
  <h1>Top Strategy Pair Statistical Check</h1>
  <p>
    The top distinct GA pairs were re-evaluated on {trials} fresh shared Monte
    Carlo trials. Shared trials reduce comparison noise because every pair is
    tested against the same generated sequences.
  </p>
  {sequential_text}

  <section>
    <h2>Result</h2>
    <p>
      The best checked pair is
      <code>{html.escape(best['strategy_A'])}</code> +
      <code>{html.escape(best['strategy_B'])}</code>
      with <strong>{pct(best['checked_score'])}</strong>.
      Pairs in the same certainty group are not separated at the selected
      interval width.
    </p>
  </section>

  <section>
    <h2>Ranking</h2>
    <table>
      <tr>
        <th>Checked rank</th>
        <th>Certainty rank</th>
        <th>Strategy A</th>
        <th>Strategy B</th>
        <th>Checked score</th>
        <th>{html.escape(label)}</th>
        <th>GA rank</th>
        <th>GA score</th>
      </tr>
      {pair_rows}
    </table>
  </section>

  <section>
    <h2>Pair Details</h2>
    {detail_sections}
  </section>
</body>
</html>
"""


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-check the top distinct GA strategy pairs with fresh shared "
            "Monte Carlo trials."
        )
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--generation", type=int)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--trials", type=int, default=100000)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Evaluate in batches and stop when the checked winner is "
            "statistically better than every other checked pair."
        ),
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        help="Maximum trials for --sequential. Defaults to --trials.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="Trials per sequential batch.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.96,
        help="Interval width in standard errors. Use 5 for five-sigma checks.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one progress line after each sequential batch.",
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(list(argv))
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.max_trials is not None and args.max_trials <= 0:
        parser.error("--max-trials must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.sigma <= 0:
        parser.error("--sigma must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    output_dir = args.output_dir
    generation = args.generation
    if generation is None:
        generation = latest_generation(output_dir)

    config = load_experiment_config(output_dir)
    population_a = load_population(output_dir, "A", generation)
    population_b = load_population(output_dir, "B", generation)
    score_matrix = load_score_matrix(output_dir, generation)
    top_pairs = find_best_pairs(population_a, population_b, score_matrix, args.top)
    concrete_pairs = [
        (
            strategy_by_id(population_a, pair.strategy_A_id),
            strategy_by_id(population_b, pair.strategy_B_id),
        )
        for pair in top_pairs
    ]
    sequence_length = minimum_sequence_length(concrete_pairs, config.sequence_length)
    if args.sequential:
        max_trials = args.max_trials if args.max_trials is not None else args.trials
        rows = sequential_comparison_rows(
            top_pairs,
            concrete_pairs,
            max_trials,
            args.batch_size,
            sequence_length,
            config.success_condition,
            args.seed,
            args.sigma,
            args.progress,
        )
    else:
        wins, pairwise_difference_sums, pairwise_difference_square_sums = (
            evaluate_pairs_on_shared_trials(
                concrete_pairs,
                args.trials,
                sequence_length,
                config.success_condition,
                args.seed,
            )
        )
        rows = comparison_rows(
            top_pairs,
            concrete_pairs,
            wins,
            pairwise_difference_sums,
            pairwise_difference_square_sums,
            args.trials,
            args.sigma,
        )
        annotate_checked_ranks(rows)
        annotate_certainty_groups(rows)
        add_run_metadata(
            rows,
            sequential=False,
            stopping_reason="fixed_trials_completed",
            batches_completed=1,
        )
    write_outputs(output_dir, rows)

    print(f"Generation checked: {generation}")
    print(f"Interval: {interval_label(args.sigma)}")
    if args.sequential:
        sequential_max_trials = (
            args.max_trials if args.max_trials is not None else args.trials
        )
        print(f"Sequential max trials: {sequential_max_trials}")
        print(f"Batch size: {args.batch_size}")
        print(f"Stopping reason: {rows[0]['stopping_reason']}")
        print(f"Trials completed: {rows[0]['trials']}")
    else:
        print(f"Trials per pair: {args.trials}")
    for row in rows:
        print(
            f"#{row['checked_rank']} {row['strategy_A']} + {row['strategy_B']} "
            f"certainty_group={row['certainty_group']} "
            f"checked={row['checked_score']:.4%} "
            f"interval=[{row['ci95_low']:.4%}, {row['ci95_high']:.4%}] "
            f"ga_rank={row['rank']} ga={row['ga_score']:.4%}"
        )
    print(f"Wrote: {output_dir / 'top_pair_statistical_check.json'}")
    print(f"Wrote: {output_dir / 'top_pair_statistical_check.csv'}")
    print(f"Wrote: {output_dir / 'top_pair_statistical_check.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
