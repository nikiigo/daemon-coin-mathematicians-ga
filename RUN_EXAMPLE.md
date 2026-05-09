# Example Run: 100 Generations With Occurrence Seeds

This document records one complete experiment and the follow-up statistical
check used to compare the best generated strategy pairs.

## GA Run

Command:

```bash
python daemon_coin.py \
  --config config.toml \
  --generations 100 \
  --output-dir ga_output_100_generations_200seed_occurrence1_2_10000_trials_childlimit4
```

Important configuration:

- `population_size_A = 200`
- `population_size_B = 200`
- `trials_per_pair = 10000`
- `sequence_length = 1000`
- `max_children_per_strategy = 4`
- `first_pattern_occurrence_min = 1`
- `first_pattern_occurrence_max = 2`

The seed population contains 200 strategies per side: 100 first-occurrence
strategies and 100 second-occurrence strategies.

Output directory:

```text
ga_output_100_generations_200seed_occurrence1_2_10000_trials_childlimit4
```

## GA Result

Final generation best pair:

| Field | Value |
| --- | --- |
| Generation | 100 |
| Strategy A | `A-g0097-000001` |
| Strategy B | `B-g0073-000011` |
| GA score | `66.51%` |

Best pair seen during the full GA run:

| Field | Value |
| --- | --- |
| Generation | 34 |
| Strategy A | `A-g0012-000010` |
| Strategy B | `B-g0007-000045` |
| GA score | `68.04%` |

The GA score is useful for search, but it is still a Monte Carlo estimate. The
top pairs are close enough that a separate statistical check is needed before
claiming one pair is better than another.

## Statistical Check

Command:

```bash
python compare_strategy_pairs.py \
  ga_output_100_generations_200seed_occurrence1_2_10000_trials_childlimit4 \
  --sequential \
  --max-trials 10000000 \
  --batch-size 100000 \
  --sigma 5 \
  --seed 1 \
  --progress
```

This rechecked the top 3 distinct final-generation pairs with fresh shared
Monte Carlo trials. Shared trials mean all checked pairs are evaluated on the
same generated sequences, reducing noise in pair-vs-pair comparisons.

The check used a 5-sigma interval. A pair is considered better than another
only when the 5-sigma interval for their score difference is entirely above
zero.

## Statistical Check Result

The run reached the full trial budget:

| Field | Value |
| --- | --- |
| Trials completed | `10,000,000` |
| Batch size | `100,000` |
| Stopping reason | `max_trials_reached` |
| Interval | `5-sigma` |

Checked scores:

| Checked rank | Certainty group | Strategy A | Strategy B | Checked score | 5-sigma interval |
| --- | --- | --- | --- | ---: | --- |
| 1 | 1 | `A-g0097-000001` | `B-g0073-000011` | `66.6777%` | `66.6032%` to `66.7523%` |
| 2 | 1 | `A-g0052-000029` | `B-g0074-000028` | `66.6744%` | `66.5999%` to `66.7489%` |
| 3 | 1 | `A-g0097-000001` | `B-g0074-000028` | `66.6715%` | `66.5970%` to `66.7461%` |

Pairwise score differences:

| Pair comparison | Difference | 5-sigma interval | Result |
| --- | ---: | --- | --- |
| Rank 1 minus Rank 2 | `0.00335%` | `-0.09522%` to `0.10192%` | Not separated |
| Rank 1 minus Rank 3 | `0.00619%` | `-0.07287%` to `0.08525%` | Not separated |
| Rank 2 minus Rank 3 | `0.00284%` | `-0.07621%` to `0.08189%` | Not separated |

## Interpretation

The ordinary checked ranking says pair 1 scored slightly higher than pair 2,
and pair 2 scored slightly higher than pair 3. However, the differences are
tiny: about `0.003` to `0.006` percentage points.

At 5-sigma certainty, those gaps are not large enough to establish a strict
ranking. All three pairs are in `certainty_group = 1`, which means the honest
statistical conclusion is:

```text
The three checked pairs are statistically tied at 5-sigma certainty.
```

This does not mean the pairs are exactly equal. It means that this experiment,
even with 10 million shared trials, does not provide enough evidence to say one
of the three is better than the others at the requested certainty level.

## Generated Reports

The statistical check wrote:

- `top_pair_statistical_check.json`
- `top_pair_statistical_check.csv`
- `top_pair_statistical_check.html`

The HTML report is available at:

```text
ga_output_100_generations_200seed_occurrence1_2_10000_trials_childlimit4/top_pair_statistical_check.html
```
