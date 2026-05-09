# daemon-coin-mathematicians-ga

Genetic algorithm experiments for the daemon coin coordination problem.

## Mathematical Problem

There is an infinite random binary sequence.

One player sees the entire subsequence of bits at even positions, while the
other player sees the entire subsequence of bits at odd positions.

Before seeing the sequences, the players are allowed to agree on a common
strategy. After the sequences are revealed, no communication is possible.

Each player must independently choose an index in the other player's sequence.
If the bit values at the chosen positions are equal, both players survive. If
not, both die.

The simulator represents the even and odd subsequences as two separate random
binary sequences. It estimates strategy quality with Monte Carlo trials and
evolves strategy populations over generations.

## Current Strategy Model

The GA currently seeds and evolves four strategy families: `first-pattern`,
`block-lookup`, `fsm`, and `gene-table`. The current `config.toml` uses
`first-pattern` only as the default family for random population slots when a
side is not fully supplied by seed strategies.

A `first-pattern` strategy:

1. Observes its own generated bit sequence.
2. Searches for occurrence number `occurrence_number` of `pattern`.
3. If that occurrence exists, chooses `occurrence_index + index_offset`, clamped into the sequence bounds.
4. If that occurrence is absent, chooses `fallback_index`.

`occurrence_number = 1` means first occurrence. `occurrence_number = 2` means
second occurrence.

`observed_length = 0` means unbounded observation within the finite simulated
`sequence_length`. A positive `observed_length` means the strategy only searches
the first N bits.

A `block-lookup` strategy splits its own sequence into fixed-size blocks, skips
configured block values, then uses the first non-skipped block as a lookup key
for a local offset inside that block. An `fsm` strategy runs a finite-state
machine over its own sequence and either chooses immediately when it reaches an
accepting state or emits the target index from its final state. A `gene-table`
strategy maps an observed bit prefix directly to a target index.

The canonical triple-block strategy is represented with `block_size = 3`,
skipped blocks `000` and `111`, and lookup offsets:

```text
001 -> 0
010 -> 1
011 -> 0
100 -> 2
101 -> 2
110 -> 1
```

## Configuration

Runtime settings are read from `config.toml` by default:

```bash
python daemon_coin.py
```

Use another config file or override common values from the CLI:

```bash
python daemon_coin.py --config config.toml --generations 50 --trials-per-pair 1000
```

Current important defaults in `config.toml`:

```toml
population_size_A = 244
population_size_B = 244
trials_per_pair = 10000
sequence_length = 1000
generations = 20
max_children_per_strategy = 4
default_strategy_type = "first-pattern"

disable_observed_length_mutation = true
disable_fallback_index_mutation = true

first_pattern_occurrence_min = 1
first_pattern_occurrence_max = 2

block_lookup_block_size_min = 1
block_lookup_block_size_max = 4
block_lookup_fallback_policy = "random-index"

fsm_state_count_min = 2
fsm_state_count_max = 6
fsm_output_index_min = 0
fsm_output_index_max = 10
fsm_index_offset_min = -10
fsm_index_offset_max = 10
fsm_fallback_index_min = 0
fsm_fallback_index_max = 10
```

`population_size_A` and `population_size_B` are maximum population sizes. If
seed strategies are configured for a side, generation 0 is exactly the expanded
seed population and validation rejects seed totals above that side's maximum. If
no seeds are configured for a side, that side is randomly filled to its maximum.

With 244 strategies per side and 10000 trials, each generation evaluates about
595,360,000 A/B pair outcomes before duplicate-strategy optimization. Lower
`trials_per_pair` for faster exploratory runs.

## Selection

Each strategy's `survival_ratio` is its best observed score with any strategy on
the opposite side. Reproduction happens inside each side separately: A children
come from A parents, and B children come from B parents. The first parent and
any compatible crossover or gene-join partners are selected with probability
weighted by `survival_ratio`, with a small floor so zero-score strategies remain
selectable.

Each parent tracks `children_produced`. When a strategy has helped produce
`max_children_per_strategy` children, it dies and is removed from the active
population. For crossover and gene-join children, every participating parent
gets one child counted.

## Seed Strategies

Generation 0 can be seeded in `config.toml`.

```toml
[[ga.initial_population_A]]
id = "A-first-0"
population = 4
strategy_type = "first-pattern"
observed_length = 0
fallback_index = 0
index_offset = 0
occurrence_number = 1
pattern = [0]

[[ga.initial_population_B]]
id = "B-first-01"
population = 4
strategy_type = "first-pattern"
observed_length = 0
fallback_index = 0
index_offset = 1
occurrence_number = 2
pattern = [0, 1]
```

Seed fields:

- `population`: number of generation-0 copies to create from the seed.
- `strategy_type`: `first-pattern`, `gene-table`, `block-lookup`, or `fsm`.
- `observed_length`: `0` means unbounded; omit it to sample from `first_pattern_observed_length_min/max`.
- `fallback_index`: used when the pattern is absent; omit it to sample from `first_pattern_fallback_index_min/max`.
- `index_offset`: added to the requested occurrence index; omit it to sample from `first_pattern_index_offset_min/max`.
- `occurrence_number`: which pattern occurrence to use; `1` first, `2` second. Omit it for seeded first-pattern strategies to use `1`.
- `pattern`: list of bits to search for.
- `block_size`: block width for a `block-lookup` strategy.
- `skip_blocks`: integer block values to ignore. With `block_size = 3`, `0` is `000` and `7` is `111`.
- `lookup_offsets`: one local offset per block value. Each value must be in `0..block_size-1`.
- `fallback_policy`: `random-index`, `first-index`, `last-index`, `random-block`, or `none`.
- `fsm_state_count`: number of finite-state-machine states.
- `fsm_start_state`: state used before reading observed bits.
- `fsm_outputs`: target index emitted by each state.
- `fsm_transitions`: one `[next_on_0, next_on_1]` row per state.
- `fsm_accepting_states`: states that immediately choose a target.
- `fsm_index_offset`: added to acceptance index and state output.
- `fsm_fallback_index`: used when no accepting state is reached.

Canonical triple-block seed example:

```toml
[[ga.initial_population_A]]
id = "A-canonical-triple"
population = 1
strategy_type = "block-lookup"
observed_length = 0
block_size = 3
skip_blocks = [0, 7]
lookup_offsets = [0, 0, 1, 0, 2, 2, 1, 0]
fallback_policy = "random-index"
```

You can also inject that seed from the CLI:

```bash
python daemon_coin.py --strategy-family block-lookup --block-size 3 --seed-known-triple-strategy --allow-asymmetric
```

The current config has 71 seed blocks per player: 7 FSM seeds, 4 block-lookup
seeds, 30 first-occurrence seeds, and 30 matching second-occurrence seeds. Their
`population` values sum to exactly 244 initial strategies per player, equal to
the configured maximum. If that sum is lower than the maximum, the initial
seeded population stays lower; it is not automatically filled with random
strategies.

## Mutation Controls

These flags freeze parts of first-pattern strategies during mutation:

```toml
disable_observed_length_mutation = true
disable_fallback_index_mutation = true
```

When disabled, mutation may still change pattern bits, pattern length, and
`index_offset` according to the configured mutation probabilities.

## Outputs

Each run writes artifacts to `output_dir`:

- `best_pair.json`: final generation best pair.
- `best_pairs.csv`: best pair per generation.
- `best_A_strategy.json` and `best_B_strategy.json`: final best individual strategies.
- `generation_stats.csv`: summary statistics per generation.
- `experiment_manifest.json`: audit metadata for the run, including command,
  git commit when available, config hash, completed generation, runtime,
  population sizes, trials, and seed.
- `experiment_report.html`: plain-language HTML report with the final
  generation's top 3 distinct strategy pairs.
- `population_snapshots/`: full population JSON per generation.
- `score_matrices/`: A/B score matrix per generation.

## Tests

Run the unit suite with:

```bash
python -m unittest discover -s tests
```

## Pair Comparison Utility

After a GA run, re-check the top distinct final-generation pairs with fresh
shared Monte Carlo trials:

```bash
python compare_strategy_pairs.py ga_output_100_generations_200seed_occurrence1_2_10000_trials_childlimit4 --trials 100000 --seed 1
```

The utility writes:

- `top_pair_statistical_check.json`
- `top_pair_statistical_check.csv`
- `top_pair_statistical_check.html`

It reports each pair's checked score, standard error, confidence interval,
z-scores against 50%, 66.67%, and 70% baselines, and pairwise score-difference
intervals against the other checked pairs. It also assigns a `certainty_group`:
pairs in the same group are not statistically separated at the selected
interval width, so the honest result may be a partial ranking rather than a
forced 1/2/3 order.

For a more reliable check, run sequentially. This evaluates in batches and stops
when the full checked ranking is statistically separated, or when the trial
budget is exhausted:

```bash
python compare_strategy_pairs.py ga_output_100_generations_200seed_occurrence1_2_10000_trials_childlimit4 --sequential --max-trials 1000000 --batch-size 100000 --seed 1
```

For a five-sigma certainty threshold, add `--sigma 5`:

```bash
python compare_strategy_pairs.py ga_output_100_generations_200seed_occurrence1_2_10000_trials_childlimit4 --sequential --max-trials 5000000 --batch-size 100000 --sigma 5 --seed 1
```

See [RUN_EXAMPLE.md](RUN_EXAMPLE.md) for a complete recorded run, including
100 GA generations and a 10,000,000-trial 5-sigma comparison.

## Notes

- Results are Monte Carlo estimates, not formal proofs.
- Larger populations and more trials reduce noise but increase runtime.
- The evaluator deduplicates identical strategy definitions during scoring, so
  seed copies can weight selection without repeating all equivalent pair
  calculations.
