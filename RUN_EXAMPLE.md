# Example Run: 200 Generations With Manifest

This document records one complete experiment and the follow-up statistical
check used to compare the best generated strategy pairs.

## GA Run

Command:

```bash
python daemon_coin.py \
  --generations 200 \
  --output-dir ga_output_200_generations_manifest_run
```

Important configuration:

- `population_size_A = 244`
- `population_size_B = 244`
- `trials_per_pair = 10000`
- `sequence_length = 1000`
- `max_children_per_strategy = 4`
- `block_lookup_block_size_min = 1`
- `block_lookup_block_size_max = 4`
- `fsm_state_count_min = 2`
- `fsm_state_count_max = 6`

The seed population contains first-pattern, block-lookup, and FSM strategies.
FSM seeds include simple finite-state machines, accepting-state pattern
detectors, and the best FSMs observed in a previous run.

Output directory:

```text
ga_output_200_generations_manifest_run
```

Run metadata from `experiment_manifest.json`:

| Field | Value |
| --- | --- |
| Completed generation | `200` |
| Elapsed time | `2432.408655` seconds |
| Git commit | `45b8da2c5a9f91c14fd3ee72dc4531a62231dd64` |
| Config SHA-256 | `5454e3219f8057edb6105cbdde57e80ce9ed7ebcd4b097855dc9186a0ae039b6` |

## GA Result

Final generation best pair:

| Field | Value |
| --- | --- |
| Generation | 200 |
| Strategy A | `A-g0171-000020` |
| Strategy B | `B-g0118-000028` |
| GA score | `70.89%` |

Best pair seen during the full GA run:

| Field | Value |
| --- | --- |
| Generation | 137 |
| Strategy A | `A-g0074-000109` |
| Strategy B | `B-g0116-000012` |
| GA score | `71.38%` |

The final-generation winner was a block-lookup pair. The GA also used
first-pattern and FSM seeds, but the strongest final strategies in this run
were still 3-bit block lookup strategies.

Final best strategy A:

```text
strategy_type = block-lookup
block_size = 3
skip_blocks = [0]
lookup_offsets = [0, 0, 1, 0, 2, 2, 1, 2]
fallback_policy = none
```

Final best strategy B:

```text
strategy_type = block-lookup
block_size = 3
skip_blocks = [0]
lookup_offsets = [0, 0, 1, 0, 2, 2, 1, 1]
fallback_policy = random-index
```

In plain language, both strategies split their own sequence into 3-bit blocks.
They skip the `000` block. For the first other block they see, they use the
block as a lookup key and choose a position inside that block. Player A gives
up if every complete block is skipped. Player B chooses a random index in that
fallback case.

## Statistical Check

Command:

```bash
python compare_strategy_pairs.py \
  ga_output_200_generations_manifest_run \
  --sigma 3 \
  --seed 1
```

This rechecked the top 3 distinct final-generation pairs with `100000` fresh
shared Monte Carlo trials. Shared trials mean every checked pair is tested on
the same generated random sequences, which makes pair-to-pair comparisons less
noisy.

The check used a 3-sigma interval. In this report, a pair is considered clearly
better only when the 3-sigma interval for its score difference against another
pair is entirely above zero.

## Statistical Check Result

| Field | Value |
| --- | --- |
| Generation checked | 200 |
| Trials per pair | `100000` |
| Interval | `3-sigma` |
| Mode | Fixed trial count |

Checked scores:

| Checked rank | Certainty group | Strategy A | Strategy B | Checked score | 3-sigma interval |
| --- | --- | --- | --- | ---: | --- |
| 1 | 1 | `A-g0171-000020` | `B-g0118-000028` | `70.0000%` | `69.5653%` to `70.4347%` |
| 2 | 1 | `A-g0171-000020` | `B-g0153-000001` | `70.0000%` | `69.5653%` to `70.4347%` |
| 3 | 1 | `A-g0171-000020` | `B-g0180-000018` | `70.0000%` | `69.5653%` to `70.4347%` |

All three checked pairs produced exactly the same fresh-check score. They are
also in the same certainty group, so the statistical check does not separate
them.

## Plain-Language Interpretation

The GA estimated the final best pair at `70.89%`. When the top final pairs were
retested with fresh random trials, they scored `70.0000%`. That difference is
normal: both numbers are Monte Carlo estimates, so they move from run to run.

The 3-sigma interval says the true score is very likely within roughly:

```text
69.57% to 70.43%
```

The score is clearly above random guessing at `50%`. It is also clearly above
`2/3`, or about `66.67%`. It is not clearly above `70%` in this check because
the interval crosses `70%`.

The three top final-generation pairs should be treated as equivalent for this
experiment. A forced rank order would be misleading because the fresh check
found no measurable difference between them.

## Generated Reports

The GA run wrote:

- `experiment_manifest.json`
- `experiment_report.html`
- `best_pair.json`
- `best_pairs.csv`
- `population_snapshots/`
- `score_matrices/`

The statistical check wrote:

- `top_pair_statistical_check.json`
- `top_pair_statistical_check.csv`
- `top_pair_statistical_check.html`

The HTML reports are available at:

```text
ga_output_200_generations_manifest_run/experiment_report.html
ga_output_200_generations_manifest_run/top_pair_statistical_check.html
```
