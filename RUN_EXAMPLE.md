# Example Run: 100 Generations With FSM Seeds

This document records one complete experiment and the follow-up statistical
check used to compare the best generated strategy pairs.

## GA Run

Command:

```bash
python daemon_coin.py \
  --generations 100 \
  --output-dir ga_output_100_generations_best_fsm_seed
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
ga_output_100_generations_best_fsm_seed
```

## GA Result

Final generation best pair:

| Field | Value |
| --- | --- |
| Generation | 100 |
| Strategy A | `A-g0033-000002` |
| Strategy B | `B-g0039-000064` |
| GA score | `70.18%` |

Best pair seen during the full GA run:

| Field | Value |
| --- | --- |
| Generation | 94 |
| Strategy A | `A-g0056-000007` |
| Strategy B | `B-g0044-000008` |
| GA score | `71.40%` |

The final-generation winner was a block-lookup pair, not an FSM pair. The GA
did try FSM seeds, but in this run the strongest final strategies were still
3-bit block lookup strategies.

Final best strategy A:

```text
strategy_type = block-lookup
block_size = 3
skip_blocks = [7]
lookup_offsets = [0, 0, 1, 0, 2, 2, 1, 0]
fallback_policy = random-index
```

Final best strategy B:

```text
strategy_type = block-lookup
block_size = 3
skip_blocks = [7]
lookup_offsets = [2, 0, 1, 0, 2, 2, 1, 0]
fallback_policy = random-index
```

In plain language, both strategies split their own sequence into 3-bit blocks.
They skip the `111` block. For the first other block they see, they use the
block as a lookup key and choose a position inside that block. The two players
use almost the same table; the main difference is how they handle block `000`.

## Statistical Check

Command:

```bash
python compare_strategy_pairs.py \
  ga_output_100_generations_best_fsm_seed \
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
| Generation checked | 100 |
| Trials per pair | `100000` |
| Interval | `3-sigma` |
| Mode | Fixed trial count |

Checked scores:

| Checked rank | Certainty group | Strategy A | Strategy B | Checked score | 3-sigma interval |
| --- | --- | --- | --- | ---: | --- |
| 1 | 1 | `A-g0033-000002` | `B-g0039-000064` | `69.8720%` | `69.4367%` to `70.3073%` |
| 2 | 1 | `A-g0047-000003` | `B-g0039-000064` | `69.8720%` | `69.4367%` to `70.3073%` |
| 3 | 1 | `A-g0077-000001` | `B-g0039-000064` | `69.8720%` | `69.4367%` to `70.3073%` |

All three checked pairs produced exactly the same fresh-check score. They are
also in the same certainty group, so the statistical check does not separate
them.

## Plain-Language Interpretation

The GA estimated the final best pair at `70.18%`. When the top final pairs were
retested with fresh random trials, they scored `69.8720%`. That difference is
normal: both numbers are Monte Carlo estimates, so they move a little from run
to run.

The 3-sigma interval says the true score is very likely within roughly:

```text
69.44% to 70.31%
```

The score is clearly above random guessing at `50%`. It is also clearly above
`2/3`, or about `66.67%`. It is not clearly above `70%` in this check because
the interval crosses `70%`.

The three top final-generation pairs should be treated as equivalent for this
experiment. A forced rank order would be misleading because the fresh check
found no measurable difference between them.

## Generated Reports

The statistical check wrote:

- `top_pair_statistical_check.json`
- `top_pair_statistical_check.csv`
- `top_pair_statistical_check.html`

The HTML report is available at:

```text
ga_output_100_generations_best_fsm_seed/top_pair_statistical_check.html
```
