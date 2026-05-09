# daemon-coin-mathematicians-ga

Genetic algorithm experiments for the daemon coin coordination problem.

Two players see separate random binary sequences. Each player must choose an
index in the other player's hidden sequence. A pair survives when the selected
bits match. The program estimates strategy quality with Monte Carlo trials and
evolves strategy populations over generations.

## Current Strategy Model

The main strategy type is `first-pattern`.

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

The code also retains a `gene-table` strategy type for lookup-table experiments,
but the default config uses `first-pattern`.

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
population_size_A = 200
population_size_B = 200
trials_per_pair = 10000
sequence_length = 1000
generations = 20
max_children_per_strategy = 4

disable_observed_length_mutation = true
disable_fallback_index_mutation = true

first_pattern_occurrence_min = 1
first_pattern_occurrence_max = 2
```

`population_size_A` and `population_size_B` are maximum population sizes. If
seed strategies are configured, generation 0 is exactly the expanded seed
population, capped by those values. If no seeds are configured for a side, that
side is randomly filled to its maximum.

With 200 strategies per side and 10000 trials, each generation evaluates about
400,000,000 A/B pair outcomes before duplicate-strategy optimization. Lower
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
- `observed_length`: `0` means unbounded; omit it to sample from `first_pattern_observed_length_min/max`.
- `fallback_index`: used when the pattern is absent; omit it to sample from `first_pattern_fallback_index_min/max`.
- `index_offset`: added to the requested occurrence index; omit it to sample from `first_pattern_index_offset_min/max`.
- `occurrence_number`: which pattern occurrence to use; `1` first, `2` second. Omit it for seeded first-pattern strategies to use `1`.
- `pattern`: list of bits to search for.

The current config has 60 seed blocks per player: 30 first-occurrence seeds and
30 matching second-occurrence seeds. Their `population` values sum to exactly
200 initial strategies per player, equal to the configured maximum. If that sum
is lower than the maximum, the initial seeded population stays lower; it is not
automatically filled with random strategies.

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

It reports each pair's checked score, a 95% confidence interval, and pairwise
score-difference intervals against the other checked pairs. It also assigns a
`certainty_group`: pairs in the same group are not statistically separated at
the selected interval width, so the honest result may be a partial ranking
rather than a forced 1/2/3 order.

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
