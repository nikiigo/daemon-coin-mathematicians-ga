import argparse
import csv
import html
import json
import logging
import random
import sys
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

PlayerType = str
SuccessCondition = Callable[[int, int], bool]
SUPPORTED_STRATEGY_TYPES = {"gene-table", "first-pattern", "block-lookup", "fsm"}
BLOCK_LOOKUP_FALLBACK_POLICIES = {
    "random-index",
    "first-index",
    "last-index",
    "random-block",
    "none",
}


@dataclass
class GAConfig:
    population_size_A: int = 100
    population_size_B: int = 100
    observed_length_min: int = 3
    observed_length_max: int = 12
    target_choice_range_min: int = 0
    target_choice_range_max: int = 0
    first_pattern_observed_length_min: int = 0
    first_pattern_observed_length_max: int = 100
    first_pattern_fallback_index_min: int = 0
    first_pattern_fallback_index_max: int = 10
    first_pattern_index_offset_min: int = -10
    first_pattern_index_offset_max: int = 10
    first_pattern_occurrence_min: int = 1
    first_pattern_occurrence_max: int = 2
    block_lookup_block_size_min: int = 3
    block_lookup_block_size_max: int = 3
    block_lookup_fallback_policy: str = "random-index"
    fsm_state_count_min: int = 2
    fsm_state_count_max: int = 6
    fsm_output_index_min: int = 0
    fsm_output_index_max: int = 10
    fsm_index_offset_min: int = -10
    fsm_index_offset_max: int = 10
    fsm_fallback_index_min: int = 0
    fsm_fallback_index_max: int = 10
    trials_per_pair: int = 500
    sequence_length: int = 1000
    generations: int = 20
    elite_count: int = 2
    selection_pressure: float = 2.0
    mutation_rate: float = 0.65
    crossover_rate: float = 0.25
    gene_join_rate: float = 0.10
    gene_mutation_probability: float = 0.05
    structure_mutation_probability: float = 0.15
    max_children_per_strategy: int = 4
    disable_observed_length_mutation: bool = True
    disable_fallback_index_mutation: bool = True
    epsilon: float = 1e-12
    success_condition: str = "same-bit"
    default_strategy_type: str = "first-pattern"
    output_dir: str = "ga_output"
    seed: int | None = None
    initial_population_A: list[dict] = field(default_factory=list)
    initial_population_B: list[dict] = field(default_factory=list)


CONFIG_SECTION = "ga"


@dataclass
class Strategy:
    id: str
    player_type: PlayerType
    observed_length: int
    target_choice_range: int
    genes: list[int]
    generation_created: int
    parent_ids: list[str]
    creation_method: str
    strategy_type: str = "gene-table"
    pattern: list[int] = field(default_factory=list)
    fallback_index: int = 1
    index_offset: int = 0
    occurrence_number: int = 1
    block_size: int = 0
    skip_blocks: list[int] = field(default_factory=list)
    lookup_offsets: list[int] = field(default_factory=list)
    fallback_policy: str = "random-index"
    fsm_state_count: int = 0
    fsm_start_state: int = 0
    fsm_outputs: list[int] = field(default_factory=list)
    fsm_transitions: list[list[int]] = field(default_factory=list)
    fsm_accepting_states: list[int] = field(default_factory=list)
    fsm_index_offset: int = 0
    fsm_fallback_index: int = 0
    survival_ratio: float = 0.0
    best_partner_id: str | None = None
    best_score: float = 0.0
    children_produced: int = 0


@dataclass
class BestPair:
    strategy_A_id: str
    strategy_B_id: str
    score: float


def same_bit_success(a_value: int, b_value: int) -> bool:
    return a_value == b_value


def both_one_success(a_value: int, b_value: int) -> bool:
    return a_value == 1 and b_value == 1


def success_condition_from_name(name: str) -> SuccessCondition:
    if name == "same-bit":
        return same_bit_success
    if name == "both-one":
        return both_one_success
    raise ValueError(f"Unsupported success condition: {name}")


def prefix_to_index(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def block_text(block_value: int, block_size: int) -> str:
    return format(block_value, f"0{block_size}b")


def normalize_fallback_policy(policy: str) -> str:
    normalized = policy.lower().replace("_", "-")
    if normalized not in BLOCK_LOOKUP_FALLBACK_POLICIES:
        raise ValueError(f"Unsupported block lookup fallback policy: {policy}")
    return normalized


def canonical_triple_lookup_offsets() -> list[int]:
    return [0, 0, 1, 0, 2, 2, 1, 0]


def canonical_triple_seed(player_type: PlayerType) -> dict:
    return {
        "id": f"{player_type}-canonical-triple",
        "strategy_type": "block-lookup",
        "population": 1,
        "observed_length": 0,
        "block_size": 3,
        "skip_blocks": [0, 7],
        "lookup_offsets": canonical_triple_lookup_offsets(),
        "fallback_policy": "random-index",
    }


def new_strategy_id(player_type: PlayerType, generation: int, counter: int) -> str:
    return f"{player_type}-g{generation:04d}-{counter:06d}"


def target_range_text(target_choice_range: int) -> str:
    if target_choice_range == 0:
        return "unbounded"
    return f"0 to {target_choice_range - 1}"


def observed_length_text(observed_length: int) -> str:
    if observed_length == 0:
        return "unbounded"
    return str(observed_length)


def random_first_pattern_observed_length(
    pattern_length: int,
    config: GAConfig,
    rng: random.Random,
) -> int:
    finite_min = max(config.first_pattern_observed_length_min, pattern_length)
    choices = list(range(finite_min, config.first_pattern_observed_length_max + 1))
    if (
        config.first_pattern_observed_length_min == 0
        and config.first_pattern_observed_length_max >= 0
    ):
        choices.insert(0, 0)

    if not choices:
        raise ValueError("No valid first-pattern observed length can be generated")
    return rng.choice(choices)


def random_first_pattern_fallback_index(config: GAConfig, rng: random.Random) -> int:
    return rng.randint(
        config.first_pattern_fallback_index_min,
        config.first_pattern_fallback_index_max,
    )


def random_first_pattern_index_offset(config: GAConfig, rng: random.Random) -> int:
    return rng.randint(
        config.first_pattern_index_offset_min,
        config.first_pattern_index_offset_max,
    )


def random_first_pattern_occurrence_number(config: GAConfig, rng: random.Random) -> int:
    return rng.randint(
        config.first_pattern_occurrence_min,
        config.first_pattern_occurrence_max,
    )


def random_block_lookup_block_size(config: GAConfig, rng: random.Random) -> int:
    return rng.randint(
        config.block_lookup_block_size_min,
        config.block_lookup_block_size_max,
    )


def random_block_lookup_skip_blocks(block_size: int, rng: random.Random) -> list[int]:
    block_count = 2**block_size
    skip_blocks = [
        block_value
        for block_value in range(block_count)
        if rng.random() < 0.25
    ]
    if len(skip_blocks) == block_count:
        skip_blocks.pop(rng.randrange(block_count))
    return skip_blocks


def random_block_lookup_offsets(block_size: int, rng: random.Random) -> list[int]:
    return [rng.randrange(block_size) for _block in range(2**block_size)]


def random_block_lookup_fallback_policy(config: GAConfig, rng: random.Random) -> str:
    return normalize_fallback_policy(config.block_lookup_fallback_policy)


def random_fsm_state_count(config: GAConfig, rng: random.Random) -> int:
    return rng.randint(config.fsm_state_count_min, config.fsm_state_count_max)


def random_fsm_outputs(
    state_count: int,
    config: GAConfig,
    rng: random.Random,
) -> list[int]:
    return [
        rng.randint(config.fsm_output_index_min, config.fsm_output_index_max)
        for _state in range(state_count)
    ]


def random_fsm_transitions(state_count: int, rng: random.Random) -> list[list[int]]:
    return [
        [rng.randrange(state_count), rng.randrange(state_count)]
        for _state in range(state_count)
    ]


def random_strategy(
    player_type: PlayerType,
    generation: int,
    counter: int,
    config: GAConfig,
    rng: random.Random,
) -> Strategy:
    genes = []
    pattern = []
    fallback_index = 0
    index_offset = 0
    occurrence_number = 1
    block_size = 0
    skip_blocks = []
    lookup_offsets = []
    fallback_policy = "random-index"
    fsm_state_count = 0
    fsm_start_state = 0
    fsm_outputs = []
    fsm_transitions = []
    fsm_accepting_states = []
    fsm_index_offset = 0
    fsm_fallback_index = 0

    if config.default_strategy_type == "gene-table":
        observed_length = rng.randint(
            config.observed_length_min,
            config.observed_length_max,
        )
        target_choice_range = rng.randint(
            config.target_choice_range_min,
            config.target_choice_range_max,
        )
        genes = [rng.randrange(target_choice_range) for _ in range(2**observed_length)]
    elif config.default_strategy_type == "first-pattern":
        target_choice_range = 0
        pattern_length = rng.randint(
            config.observed_length_min,
            config.observed_length_max,
        )
        pattern = [rng.randrange(2) for _ in range(pattern_length)]
        observed_length = random_first_pattern_observed_length(
            pattern_length,
            config,
            rng,
        )
        fallback_index = random_first_pattern_fallback_index(config, rng)
        index_offset = random_first_pattern_index_offset(config, rng)
        occurrence_number = random_first_pattern_occurrence_number(config, rng)
    elif config.default_strategy_type == "block-lookup":
        observed_length = 0
        target_choice_range = 0
        block_size = random_block_lookup_block_size(config, rng)
        skip_blocks = random_block_lookup_skip_blocks(block_size, rng)
        lookup_offsets = random_block_lookup_offsets(block_size, rng)
        fallback_policy = random_block_lookup_fallback_policy(config, rng)
    elif config.default_strategy_type == "fsm":
        observed_length = 0
        target_choice_range = 0
        fsm_state_count = random_fsm_state_count(config, rng)
        fsm_start_state = rng.randrange(fsm_state_count)
        fsm_outputs = random_fsm_outputs(fsm_state_count, config, rng)
        fsm_transitions = random_fsm_transitions(fsm_state_count, rng)
        fsm_accepting_states = [
            state
            for state in range(fsm_state_count)
            if rng.random() < 0.25
        ]
        fsm_index_offset = rng.randint(
            config.fsm_index_offset_min,
            config.fsm_index_offset_max,
        )
        fsm_fallback_index = rng.randint(
            config.fsm_fallback_index_min,
            config.fsm_fallback_index_max,
        )
    else:
        raise ValueError(f"Unsupported strategy type: {config.default_strategy_type}")

    return Strategy(
        id=new_strategy_id(player_type, generation, counter),
        player_type=player_type,
        observed_length=observed_length,
        target_choice_range=target_choice_range,
        genes=genes,
        generation_created=generation,
        parent_ids=[],
        creation_method="random",
        strategy_type=config.default_strategy_type,
        pattern=pattern,
        fallback_index=fallback_index,
        index_offset=index_offset,
        occurrence_number=occurrence_number,
        block_size=block_size,
        skip_blocks=skip_blocks,
        lookup_offsets=lookup_offsets,
        fallback_policy=fallback_policy,
        fsm_state_count=fsm_state_count,
        fsm_start_state=fsm_start_state,
        fsm_outputs=fsm_outputs,
        fsm_transitions=fsm_transitions,
        fsm_accepting_states=fsm_accepting_states,
        fsm_index_offset=fsm_index_offset,
        fsm_fallback_index=fsm_fallback_index,
    )


def random_population(
    player_type: PlayerType,
    size: int,
    config: GAConfig,
    rng: random.Random,
) -> list[Strategy]:
    return [random_strategy(player_type, 0, index, config, rng) for index in range(size)]


def strategy_from_seed(
    player_type: PlayerType,
    seed_config: dict,
    index: int,
    config: GAConfig,
    rng: random.Random,
    copy_index: int = 0,
    copy_count: int = 1,
) -> Strategy:
    base_strategy_id = seed_config.get("id", f"{player_type}-seed-{index:06d}")
    strategy_id = (
        base_strategy_id
        if copy_count == 1
        else f"{base_strategy_id}-copy-{copy_index:02d}"
    )
    strategy_type = seed_config.get("strategy_type")
    if strategy_type is None:
        if "fsm_outputs" in seed_config or "fsm_transitions" in seed_config:
            strategy_type = "fsm"
        elif "block_size" in seed_config or "lookup_offsets" in seed_config:
            strategy_type = "block-lookup"
        else:
            strategy_type = "first-pattern" if "pattern" in seed_config else "gene-table"
    pattern = list(seed_config.get("pattern", []))
    if strategy_type == "first-pattern":
        observed_length = seed_config.get("observed_length")
        if observed_length is None:
            observed_length = random_first_pattern_observed_length(
                len(pattern),
                config,
                rng,
            )
    elif strategy_type == "block-lookup":
        observed_length = seed_config.get("observed_length", 0)
    elif strategy_type == "fsm":
        observed_length = seed_config.get("observed_length", 0)
    else:
        observed_length = seed_config["observed_length"]

    block_size = seed_config.get("block_size", 0)
    fsm_state_count = seed_config.get("fsm_state_count", len(seed_config.get("fsm_outputs", [])))

    return Strategy(
        id=strategy_id,
        player_type=player_type,
        observed_length=observed_length,
        target_choice_range=(
            0
            if strategy_type in {"first-pattern", "block-lookup", "fsm"}
            else seed_config["target_choice_range"]
        ),
        genes=list(seed_config.get("genes", [])),
        generation_created=0,
        parent_ids=[],
        creation_method="initial-seed",
        strategy_type=strategy_type,
        pattern=pattern,
        fallback_index=(
            seed_config["fallback_index"]
            if "fallback_index" in seed_config
            else random_first_pattern_fallback_index(config, rng)
        )
        if strategy_type == "first-pattern"
        else 0,
        index_offset=(
            seed_config["index_offset"]
            if "index_offset" in seed_config
            else random_first_pattern_index_offset(config, rng)
        )
        if strategy_type == "first-pattern"
        else 0,
        occurrence_number=(
            seed_config.get("occurrence_number", 1)
            if strategy_type == "first-pattern"
            else 1
        ),
        block_size=block_size,
        skip_blocks=list(seed_config.get("skip_blocks", [])),
        lookup_offsets=list(seed_config.get("lookup_offsets", [])),
        fallback_policy=normalize_fallback_policy(
            seed_config.get("fallback_policy", config.block_lookup_fallback_policy)
        )
        if strategy_type == "block-lookup"
        else "random-index",
        fsm_state_count=fsm_state_count,
        fsm_start_state=seed_config.get("fsm_start_state", 0),
        fsm_outputs=list(seed_config.get("fsm_outputs", [])),
        fsm_transitions=[
            list(transition)
            for transition in seed_config.get("fsm_transitions", [])
        ],
        fsm_accepting_states=list(seed_config.get("fsm_accepting_states", [])),
        fsm_index_offset=seed_config.get("fsm_index_offset", 0),
        fsm_fallback_index=seed_config.get("fsm_fallback_index", 0),
    )


def initial_population(
    player_type: PlayerType,
    max_size: int,
    seed_configs: list[dict],
    config: GAConfig,
    rng: random.Random,
) -> list[Strategy]:
    population = []
    for index, seed_config in enumerate(seed_configs):
        copy_count = seed_config.get("population", 1)
        for copy_index in range(copy_count):
            population.append(
                strategy_from_seed(
                    player_type,
                    seed_config,
                    index,
                    config,
                    rng,
                    copy_index,
                    copy_count,
                )
            )

    if not seed_configs:
        while len(population) < max_size:
            population.append(
                random_strategy(player_type, 0, len(population), config, rng)
            )

    return population


def next_generation_size(current_size: int, max_size: int) -> int:
    return min(current_size, max_size)


def random_bits(length: int, rng: random.Random) -> list[int]:
    return [rng.randrange(2) for _ in range(length)]


def pattern_occurrence_index(
    bits: list[int],
    pattern: list[int],
    occurrence_number: int,
) -> int | None:
    matches_seen = 0
    last_start = len(bits) - len(pattern)
    for index in range(last_start + 1):
        if bits[index : index + len(pattern)] == pattern:
            matches_seen += 1
            if matches_seen == occurrence_number:
                return index
    return None


def first_pattern_index(bits: list[int], pattern: list[int]) -> int | None:
    return pattern_occurrence_index(bits, pattern, 1)


def strategy_signature(strategy: Strategy) -> tuple:
    return (
        strategy.strategy_type,
        strategy.observed_length,
        strategy.target_choice_range,
        tuple(strategy.genes),
        tuple(strategy.pattern),
        strategy.fallback_index,
        strategy.index_offset,
        strategy.occurrence_number,
        strategy.block_size,
        tuple(strategy.skip_blocks),
        tuple(strategy.lookup_offsets),
        strategy.fallback_policy,
        strategy.fsm_state_count,
        strategy.fsm_start_state,
        tuple(strategy.fsm_outputs),
        tuple(tuple(transition) for transition in strategy.fsm_transitions),
        tuple(strategy.fsm_accepting_states),
        strategy.fsm_index_offset,
        strategy.fsm_fallback_index,
    )


def unique_strategy_groups(population: list[Strategy]) -> tuple[list[Strategy], list[int]]:
    unique_strategies = []
    signature_to_index = {}
    population_to_unique = []

    for strategy in population:
        signature = strategy_signature(strategy)
        unique_index = signature_to_index.get(signature)
        if unique_index is None:
            unique_index = len(unique_strategies)
            signature_to_index[signature] = unique_index
            unique_strategies.append(strategy)
        population_to_unique.append(unique_index)

    return unique_strategies, population_to_unique


def choose_block_lookup_fallback(
    strategy: Strategy,
    observed_bits: list[int],
    rng: random.Random | None,
) -> int | None:
    if not observed_bits:
        return None

    policy = normalize_fallback_policy(strategy.fallback_policy)
    if policy == "first-index":
        return 0
    if policy == "last-index":
        return len(observed_bits) - 1
    if policy == "none":
        return None

    fallback_rng = rng or random.Random(prefix_to_index(observed_bits[:16]))
    if policy == "random-block":
        complete_blocks = len(observed_bits) // strategy.block_size
        if complete_blocks > 0:
            block_start = fallback_rng.randrange(complete_blocks) * strategy.block_size
            return block_start + fallback_rng.randrange(strategy.block_size)
    return fallback_rng.randrange(len(observed_bits))


def choose_target_index(
    strategy: Strategy,
    observed_bits: list[int],
    rng: random.Random | None = None,
) -> int | None:
    if strategy.strategy_type == "gene-table":
        return strategy.genes[prefix_to_index(observed_bits)]

    if strategy.strategy_type == "first-pattern":
        match_index = pattern_occurrence_index(
            observed_bits,
            strategy.pattern,
            strategy.occurrence_number,
        )
        if match_index is None:
            return strategy.fallback_index
        return clamp(match_index + strategy.index_offset, 0, len(observed_bits) - 1)

    if strategy.strategy_type == "block-lookup":
        skip_blocks = set(strategy.skip_blocks)
        block_size = strategy.block_size
        for block_start in range(0, len(observed_bits) - block_size + 1, block_size):
            block_bits = observed_bits[block_start : block_start + block_size]
            block_value = prefix_to_index(block_bits)
            if block_value in skip_blocks:
                continue
            local_offset = strategy.lookup_offsets[block_value] % block_size
            return block_start + local_offset
        return choose_block_lookup_fallback(strategy, observed_bits, rng)

    if strategy.strategy_type == "fsm":
        state = strategy.fsm_start_state
        accepting_states = set(strategy.fsm_accepting_states)
        for bit_index, bit in enumerate(observed_bits):
            state = strategy.fsm_transitions[state][bit]
            if state in accepting_states:
                return clamp(
                    bit_index + strategy.fsm_outputs[state] + strategy.fsm_index_offset,
                    0,
                    len(observed_bits) - 1,
                )
        if accepting_states:
            return strategy.fsm_fallback_index
        return strategy.fsm_outputs[state]

    raise ValueError(f"Unsupported strategy type: {strategy.strategy_type}")


def evaluate_pair(
    strategy_a: Strategy,
    strategy_b: Strategy,
    trials: int,
    sequence_length: int,
    success_condition: SuccessCondition,
    rng: random.Random,
) -> float:
    sequence_length = max(
        sequence_length,
        strategy_a.observed_length,
        strategy_b.observed_length,
        strategy_a.target_choice_range,
        strategy_b.target_choice_range,
        strategy_a.fallback_index + 1,
        strategy_b.fallback_index + 1,
        strategy_a.block_size,
        strategy_b.block_size,
        max(strategy_a.fsm_outputs, default=0) + 1,
        max(strategy_b.fsm_outputs, default=0) + 1,
        strategy_a.fsm_fallback_index + 1,
        strategy_b.fsm_fallback_index + 1,
    )
    wins = 0

    for _ in range(trials):
        sequence_a = random_bits(sequence_length, rng)
        sequence_b = random_bits(sequence_length, rng)

        observed_a = (
            sequence_a
            if strategy_a.observed_length == 0
            else sequence_a[: strategy_a.observed_length]
        )
        observed_b = (
            sequence_b
            if strategy_b.observed_length == 0
            else sequence_b[: strategy_b.observed_length]
        )
        target_in_b = choose_target_index(strategy_a, observed_a, rng)
        target_in_a = choose_target_index(strategy_b, observed_b, rng)

        if (
            target_in_a is not None
            and target_in_b is not None
            and success_condition(sequence_a[target_in_a], sequence_b[target_in_b])
        ):
            wins += 1

    return wins / trials


def evaluate_populations(
    population_a: list[Strategy],
    population_b: list[Strategy],
    config: GAConfig,
    success_condition: SuccessCondition,
    rng: random.Random,
) -> list[list[float]]:
    unique_a, population_a_to_unique = unique_strategy_groups(population_a)
    unique_b, population_b_to_unique = unique_strategy_groups(population_b)
    sequence_length = max(
        config.sequence_length,
        *(strategy.fallback_index + 1 for strategy in unique_a),
        *(strategy.fallback_index + 1 for strategy in unique_b),
        *(strategy.observed_length for strategy in unique_a),
        *(strategy.observed_length for strategy in unique_b),
        *(strategy.target_choice_range for strategy in unique_a),
        *(strategy.target_choice_range for strategy in unique_b),
        *(strategy.block_size for strategy in unique_a),
        *(strategy.block_size for strategy in unique_b),
        *(max(strategy.fsm_outputs, default=0) + 1 for strategy in unique_a),
        *(max(strategy.fsm_outputs, default=0) + 1 for strategy in unique_b),
        *(strategy.fsm_fallback_index + 1 for strategy in unique_a),
        *(strategy.fsm_fallback_index + 1 for strategy in unique_b),
    )
    wins = [
        [0 for _strategy_b in unique_b]
        for _strategy_a in unique_a
    ]

    for _trial in range(config.trials_per_pair):
        sequence_a = random_bits(sequence_length, rng)
        sequence_b = random_bits(sequence_length, rng)
        targets_in_b = [
            choose_target_index(
                strategy,
                sequence_a
                if strategy.observed_length == 0
                else sequence_a[: strategy.observed_length],
                rng,
            )
            for strategy in unique_a
        ]
        targets_in_a = [
            choose_target_index(
                strategy,
                sequence_b
                if strategy.observed_length == 0
                else sequence_b[: strategy.observed_length],
                rng,
            )
            for strategy in unique_b
        ]

        for row_index, target_in_b in enumerate(targets_in_b):
            if target_in_b is None:
                continue
            b_value = sequence_b[target_in_b]
            for column_index, target_in_a in enumerate(targets_in_a):
                if target_in_a is not None and success_condition(
                    sequence_a[target_in_a],
                    b_value,
                ):
                    wins[row_index][column_index] += 1

    unique_score_matrix = [
        [wins_count / config.trials_per_pair for wins_count in row]
        for row in wins
    ]
    score_matrix = [
        [
            unique_score_matrix[unique_a_index][unique_b_index]
            for unique_b_index in population_b_to_unique
        ]
        for unique_a_index in population_a_to_unique
    ]

    assign_survival(population_a, population_b, score_matrix)
    return score_matrix
def assign_survival(
    population_a: list[Strategy],
    population_b: list[Strategy],
    score_matrix: list[list[float]],
) -> None:
    for row_index, strategy_a in enumerate(population_a):
        best_score = max(score_matrix[row_index])
        best_partner_index = score_matrix[row_index].index(best_score)
        strategy_a.survival_ratio = best_score
        strategy_a.best_score = best_score
        strategy_a.best_partner_id = population_b[best_partner_index].id

    for column_index, strategy_b in enumerate(population_b):
        column_scores = [row[column_index] for row in score_matrix]
        best_score = max(column_scores)
        best_partner_index = column_scores.index(best_score)
        strategy_b.survival_ratio = best_score
        strategy_b.best_score = best_score
        strategy_b.best_partner_id = population_a[best_partner_index].id


def elite_strategies(population: list[Strategy], elite_count: int) -> list[Strategy]:
    return sorted(population, key=lambda strategy: strategy.survival_ratio, reverse=True)[
        :elite_count
    ]


def kill_weak_strategies(
    population: list[Strategy],
    elite_count: int,
    selection_pressure: float,
    epsilon: float,
    rng: random.Random,
) -> list[Strategy]:
    elites = set(strategy.id for strategy in elite_strategies(population, elite_count))
    min_survival = min(strategy.survival_ratio for strategy in population)
    max_survival = max(strategy.survival_ratio for strategy in population)
    survivors = []

    for strategy in population:
        if strategy.id in elites:
            survivors.append(strategy)
            continue

        normalized = (strategy.survival_ratio - min_survival) / (
            max_survival - min_survival + epsilon
        )
        kill_probability = (1 - normalized) ** selection_pressure
        if rng.random() >= kill_probability:
            survivors.append(strategy)

    return survivors


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def resize_genes(
    genes: list[int],
    old_observed_length: int,
    new_observed_length: int,
    target_choice_range: int,
    rng: random.Random,
) -> list[int]:
    new_gene_count = 2**new_observed_length
    if new_observed_length == old_observed_length:
        resized = list(genes)
    elif new_observed_length < old_observed_length:
        resized = genes[:new_gene_count]
    else:
        resized = list(genes)
        resized.extend(
            rng.randrange(target_choice_range)
            for _ in range(new_gene_count - len(genes))
        )
    return [gene % target_choice_range for gene in resized]


def resize_lookup_offsets(
    lookup_offsets: list[int],
    old_block_size: int,
    new_block_size: int,
    rng: random.Random,
) -> list[int]:
    new_count = 2**new_block_size
    if new_block_size == old_block_size:
        resized = list(lookup_offsets)
    elif new_block_size < old_block_size:
        resized = lookup_offsets[:new_count]
    else:
        resized = list(lookup_offsets)
        resized.extend(
            rng.randrange(new_block_size)
            for _index in range(new_count - len(lookup_offsets))
        )
    return [offset % new_block_size for offset in resized[:new_count]]


def normalize_skip_blocks(skip_blocks: list[int], block_size: int) -> list[int]:
    block_count = 2**block_size
    normalized = sorted({block for block in skip_blocks if 0 <= block < block_count})
    if len(normalized) == block_count:
        normalized.pop()
    return normalized


def resize_fsm(
    outputs: list[int],
    transitions: list[list[int]],
    old_state_count: int,
    new_state_count: int,
    config: GAConfig,
    rng: random.Random,
) -> tuple[list[int], list[list[int]]]:
    resized_outputs = list(outputs[:new_state_count])
    while len(resized_outputs) < new_state_count:
        resized_outputs.append(
            rng.randint(config.fsm_output_index_min, config.fsm_output_index_max)
        )

    resized_transitions = [list(transition[:2]) for transition in transitions[:new_state_count]]
    while len(resized_transitions) < new_state_count:
        resized_transitions.append(
            [rng.randrange(new_state_count), rng.randrange(new_state_count)]
        )
    resized_transitions = [
        [
            transition[0] % new_state_count,
            transition[1] % new_state_count,
        ]
        for transition in resized_transitions
    ]
    return resized_outputs, resized_transitions


def mutate_strategy(
    parent: Strategy,
    generation: int,
    counter: int,
    config: GAConfig,
    rng: random.Random,
) -> Strategy:
    observed_length = parent.observed_length
    target_choice_range = parent.target_choice_range

    if (
        parent.strategy_type == "gene-table"
        and not config.disable_observed_length_mutation
        and rng.random() < config.structure_mutation_probability
    ):
        observed_length = clamp(
            observed_length + rng.choice([-1, 1]),
            config.observed_length_min,
            config.observed_length_max,
        )

    if (
        parent.strategy_type == "gene-table"
        and rng.random() < config.structure_mutation_probability
    ):
        target_choice_range = clamp(
            target_choice_range + rng.choice([-1, 1]),
            config.target_choice_range_min,
            config.target_choice_range_max,
        )

    genes = []
    pattern = list(parent.pattern)
    fallback_index = 0
    index_offset = 0
    occurrence_number = parent.occurrence_number
    block_size = parent.block_size
    skip_blocks = list(parent.skip_blocks)
    lookup_offsets = list(parent.lookup_offsets)
    fallback_policy = parent.fallback_policy
    fsm_state_count = parent.fsm_state_count
    fsm_start_state = parent.fsm_start_state
    fsm_outputs = list(parent.fsm_outputs)
    fsm_transitions = [list(transition) for transition in parent.fsm_transitions]
    fsm_accepting_states = list(parent.fsm_accepting_states)
    fsm_index_offset = parent.fsm_index_offset
    fsm_fallback_index = parent.fsm_fallback_index

    if parent.strategy_type == "gene-table":
        occurrence_number = 1
        block_size = 0
        skip_blocks = []
        lookup_offsets = []
        fallback_policy = "random-index"
        fsm_state_count = 0
        fsm_start_state = 0
        fsm_outputs = []
        fsm_transitions = []
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
        genes = resize_genes(
            parent.genes,
            parent.observed_length,
            observed_length,
            target_choice_range,
            rng,
        )
        for index in range(len(genes)):
            if rng.random() < config.gene_mutation_probability:
                genes[index] = rng.randrange(target_choice_range)
    elif parent.strategy_type == "first-pattern":
        target_choice_range = 0
        block_size = 0
        skip_blocks = []
        lookup_offsets = []
        fallback_policy = "random-index"
        fsm_state_count = 0
        fsm_start_state = 0
        fsm_outputs = []
        fsm_transitions = []
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
        if (
            not config.disable_observed_length_mutation
            and rng.random() < config.structure_mutation_probability
        ):
            if observed_length == 0:
                observed_length = rng.randint(
                    max(len(pattern), config.observed_length_min),
                    config.observed_length_max,
                )
            elif rng.random() < 0.2:
                observed_length = 0
            else:
                observed_length = clamp(
                    observed_length + rng.choice([-1, 1]),
                    max(len(pattern), config.observed_length_min),
                    config.observed_length_max,
                )

        if (
            not config.disable_fallback_index_mutation
            and rng.random() < config.structure_mutation_probability
        ):
            fallback_index = clamp(
                parent.fallback_index + rng.choice([-1, 1]),
                0,
                config.sequence_length - 1,
            )
        else:
            fallback_index = parent.fallback_index

        if rng.random() < config.structure_mutation_probability:
            index_offset = clamp(
                parent.index_offset + rng.choice([-1, 1]),
                config.first_pattern_index_offset_min,
                config.first_pattern_index_offset_max,
            )
        else:
            index_offset = parent.index_offset

        if rng.random() < config.structure_mutation_probability:
            occurrence_number = clamp(
                parent.occurrence_number + rng.choice([-1, 1]),
                config.first_pattern_occurrence_min,
                config.first_pattern_occurrence_max,
            )

        if rng.random() < config.structure_mutation_probability:
            if len(pattern) == config.observed_length_min:
                direction = 1
            elif len(pattern) == config.observed_length_max:
                direction = -1
            else:
                direction = rng.choice([-1, 1])

            if direction < 0:
                pattern = pattern[:-1]
            else:
                pattern.append(rng.randrange(2))

        while not pattern:
            pattern.append(rng.randrange(2))
        if observed_length != 0 and len(pattern) > observed_length:
            pattern = pattern[:observed_length]
        for index in range(len(pattern)):
            if rng.random() < config.gene_mutation_probability:
                pattern[index] = 1 - pattern[index]
    elif parent.strategy_type == "block-lookup":
        observed_length = 0
        target_choice_range = 0
        fallback_index = 0
        index_offset = 0
        occurrence_number = 1
        genes = []
        pattern = []

        if (
            config.block_lookup_block_size_min < config.block_lookup_block_size_max
            and rng.random() < config.structure_mutation_probability
        ):
            block_size = clamp(
                block_size + rng.choice([-1, 1]),
                config.block_lookup_block_size_min,
                config.block_lookup_block_size_max,
            )
        lookup_offsets = resize_lookup_offsets(
            parent.lookup_offsets,
            parent.block_size,
            block_size,
            rng,
        )
        skip_blocks = normalize_skip_blocks(parent.skip_blocks, block_size)

        if rng.random() < config.structure_mutation_probability:
            block_to_toggle = rng.randrange(2**block_size)
            if block_to_toggle in skip_blocks:
                skip_blocks.remove(block_to_toggle)
            else:
                skip_blocks.append(block_to_toggle)
            skip_blocks = normalize_skip_blocks(skip_blocks, block_size)

        for index in range(len(lookup_offsets)):
            if rng.random() < config.gene_mutation_probability:
                lookup_offsets[index] = rng.randrange(block_size)

        if rng.random() < config.structure_mutation_probability:
            fallback_policy = rng.choice(sorted(BLOCK_LOOKUP_FALLBACK_POLICIES))
        fsm_state_count = 0
        fsm_start_state = 0
        fsm_outputs = []
        fsm_transitions = []
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
    elif parent.strategy_type == "fsm":
        observed_length = 0
        target_choice_range = 0
        fallback_index = 0
        index_offset = 0
        occurrence_number = 1
        genes = []
        pattern = []
        block_size = 0
        skip_blocks = []
        lookup_offsets = []
        fallback_policy = "random-index"

        if (
            config.fsm_state_count_min < config.fsm_state_count_max
            and rng.random() < config.structure_mutation_probability
        ):
            fsm_state_count = clamp(
                fsm_state_count + rng.choice([-1, 1]),
                config.fsm_state_count_min,
                config.fsm_state_count_max,
            )
        fsm_outputs, fsm_transitions = resize_fsm(
            parent.fsm_outputs,
            parent.fsm_transitions,
            parent.fsm_state_count,
            fsm_state_count,
            config,
            rng,
        )
        fsm_start_state %= fsm_state_count
        fsm_accepting_states = [
            state for state in fsm_accepting_states if state < fsm_state_count
        ]

        if rng.random() < config.structure_mutation_probability:
            fsm_start_state = rng.randrange(fsm_state_count)
        if rng.random() < config.structure_mutation_probability:
            state = rng.randrange(fsm_state_count)
            if state in fsm_accepting_states:
                fsm_accepting_states.remove(state)
            else:
                fsm_accepting_states.append(state)
            fsm_accepting_states = sorted(set(fsm_accepting_states))
        if rng.random() < config.structure_mutation_probability:
            fsm_index_offset = clamp(
                fsm_index_offset + rng.choice([-1, 1]),
                config.fsm_index_offset_min,
                config.fsm_index_offset_max,
            )
        if rng.random() < config.structure_mutation_probability:
            fsm_fallback_index = clamp(
                fsm_fallback_index + rng.choice([-1, 1]),
                config.fsm_fallback_index_min,
                config.fsm_fallback_index_max,
            )

        for index in range(len(fsm_outputs)):
            if rng.random() < config.gene_mutation_probability:
                fsm_outputs[index] = rng.randint(
                    config.fsm_output_index_min,
                    config.fsm_output_index_max,
                )

        for state in range(fsm_state_count):
            for bit in range(2):
                if rng.random() < config.gene_mutation_probability:
                    fsm_transitions[state][bit] = rng.randrange(fsm_state_count)
    else:
        raise ValueError(f"Unsupported strategy type: {parent.strategy_type}")

    return Strategy(
        id=new_strategy_id(parent.player_type, generation, counter),
        player_type=parent.player_type,
        observed_length=observed_length,
        target_choice_range=target_choice_range,
        genes=genes,
        generation_created=generation,
        parent_ids=[parent.id],
        creation_method="mutation",
        strategy_type=parent.strategy_type,
        pattern=pattern,
        fallback_index=fallback_index,
        index_offset=index_offset,
        occurrence_number=occurrence_number,
        block_size=block_size,
        skip_blocks=skip_blocks,
        lookup_offsets=lookup_offsets,
        fallback_policy=fallback_policy,
        fsm_state_count=fsm_state_count,
        fsm_start_state=fsm_start_state,
        fsm_outputs=fsm_outputs,
        fsm_transitions=fsm_transitions,
        fsm_accepting_states=fsm_accepting_states,
        fsm_index_offset=fsm_index_offset,
        fsm_fallback_index=fsm_fallback_index,
    )


def compatible(parent_1: Strategy, parent_2: Strategy) -> bool:
    return (
        parent_1.player_type == parent_2.player_type
        and parent_1.strategy_type == parent_2.strategy_type
        and parent_1.observed_length == parent_2.observed_length
        and parent_1.target_choice_range == parent_2.target_choice_range
        and len(parent_1.genes) == len(parent_2.genes)
        and len(parent_1.pattern) == len(parent_2.pattern)
        and parent_1.block_size == parent_2.block_size
        and len(parent_1.lookup_offsets) == len(parent_2.lookup_offsets)
        and parent_1.fsm_state_count == parent_2.fsm_state_count
        and len(parent_1.fsm_outputs) == len(parent_2.fsm_outputs)
        and len(parent_1.fsm_transitions) == len(parent_2.fsm_transitions)
    )


def crossover_strategy(
    parent_1: Strategy,
    parent_2: Strategy,
    generation: int,
    counter: int,
    rng: random.Random,
) -> Strategy:
    if parent_1.strategy_type == "gene-table":
        genes = [
            gene_1 if rng.random() < 0.5 else gene_2
            for gene_1, gene_2 in zip(parent_1.genes, parent_2.genes)
        ]
        pattern = []
        fallback_index = 0
        index_offset = 0
        occurrence_number = 1
        block_size = 0
        skip_blocks = []
        lookup_offsets = []
        fallback_policy = "random-index"
        fsm_state_count = 0
        fsm_start_state = 0
        fsm_outputs = []
        fsm_transitions = []
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
    elif parent_1.strategy_type == "first-pattern":
        genes = []
        pattern = [
            bit_1 if rng.random() < 0.5 else bit_2
            for bit_1, bit_2 in zip(parent_1.pattern, parent_2.pattern)
        ]
        fallback_index = rng.choice([parent_1.fallback_index, parent_2.fallback_index])
        index_offset = rng.choice([parent_1.index_offset, parent_2.index_offset])
        occurrence_number = rng.choice(
            [parent_1.occurrence_number, parent_2.occurrence_number]
        )
        block_size = 0
        skip_blocks = []
        lookup_offsets = []
        fallback_policy = "random-index"
        fsm_state_count = 0
        fsm_start_state = 0
        fsm_outputs = []
        fsm_transitions = []
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
    elif parent_1.strategy_type == "block-lookup":
        genes = []
        pattern = []
        fallback_index = 0
        index_offset = 0
        occurrence_number = 1
        block_size = parent_1.block_size
        parent_1_skip = set(parent_1.skip_blocks)
        parent_2_skip = set(parent_2.skip_blocks)
        skip_blocks = [
            block
            for block in range(2**block_size)
            if (
                block in parent_1_skip
                if rng.random() < 0.5
                else block in parent_2_skip
            )
        ]
        skip_blocks = normalize_skip_blocks(skip_blocks, block_size)
        lookup_offsets = [
            offset_1 if rng.random() < 0.5 else offset_2
            for offset_1, offset_2 in zip(
                parent_1.lookup_offsets,
                parent_2.lookup_offsets,
            )
        ]
        fallback_policy = rng.choice([parent_1.fallback_policy, parent_2.fallback_policy])
        fsm_state_count = 0
        fsm_start_state = 0
        fsm_outputs = []
        fsm_transitions = []
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
    elif parent_1.strategy_type == "fsm":
        genes = []
        pattern = []
        fallback_index = 0
        index_offset = 0
        occurrence_number = 1
        block_size = 0
        skip_blocks = []
        lookup_offsets = []
        fallback_policy = "random-index"
        fsm_state_count = parent_1.fsm_state_count
        fsm_start_state = rng.choice([parent_1.fsm_start_state, parent_2.fsm_start_state])
        fsm_outputs = [
            output_1 if rng.random() < 0.5 else output_2
            for output_1, output_2 in zip(parent_1.fsm_outputs, parent_2.fsm_outputs)
        ]
        fsm_transitions = [
            [
                transition_1[bit] if rng.random() < 0.5 else transition_2[bit]
                for bit in range(2)
            ]
            for transition_1, transition_2 in zip(
                parent_1.fsm_transitions,
                parent_2.fsm_transitions,
            )
        ]
        fsm_accepting_states = [
            state
            for state in range(fsm_state_count)
            if (
                state in parent_1.fsm_accepting_states
                if rng.random() < 0.5
                else state in parent_2.fsm_accepting_states
            )
        ]
        fsm_index_offset = rng.choice([parent_1.fsm_index_offset, parent_2.fsm_index_offset])
        fsm_fallback_index = rng.choice([parent_1.fsm_fallback_index, parent_2.fsm_fallback_index])
    else:
        raise ValueError(f"Unsupported strategy type: {parent_1.strategy_type}")

    return Strategy(
        id=new_strategy_id(parent_1.player_type, generation, counter),
        player_type=parent_1.player_type,
        observed_length=parent_1.observed_length,
        target_choice_range=parent_1.target_choice_range,
        genes=genes,
        generation_created=generation,
        parent_ids=[parent_1.id, parent_2.id],
        creation_method="crossover",
        strategy_type=parent_1.strategy_type,
        pattern=pattern,
        fallback_index=fallback_index,
        index_offset=index_offset,
        occurrence_number=occurrence_number,
        block_size=block_size,
        skip_blocks=skip_blocks,
        lookup_offsets=lookup_offsets,
        fallback_policy=fallback_policy,
        fsm_state_count=fsm_state_count,
        fsm_start_state=fsm_start_state,
        fsm_outputs=fsm_outputs,
        fsm_transitions=fsm_transitions,
        fsm_accepting_states=fsm_accepting_states,
        fsm_index_offset=fsm_index_offset,
        fsm_fallback_index=fsm_fallback_index,
    )


def gene_join_strategy(
    parents: list[Strategy],
    generation: int,
    counter: int,
    rng: random.Random,
) -> Strategy:
    first = parents[0]
    if first.strategy_type == "gene-table":
        genes = [rng.choice(parents).genes[index] for index in range(len(first.genes))]
        pattern = []
        fallback_index = 0
        index_offset = 0
        occurrence_number = 1
        block_size = 0
        skip_blocks = []
        lookup_offsets = []
        fallback_policy = "random-index"
        fsm_state_count = 0
        fsm_start_state = 0
        fsm_outputs = []
        fsm_transitions = []
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
    elif first.strategy_type == "first-pattern":
        genes = []
        pattern = [
            rng.choice(parents).pattern[index] for index in range(len(first.pattern))
        ]
        fallback_index = rng.choice(parents).fallback_index
        index_offset = rng.choice(parents).index_offset
        occurrence_number = rng.choice(parents).occurrence_number
        block_size = 0
        skip_blocks = []
        lookup_offsets = []
        fallback_policy = "random-index"
        fsm_state_count = 0
        fsm_start_state = 0
        fsm_outputs = []
        fsm_transitions = []
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
    elif first.strategy_type == "block-lookup":
        genes = []
        pattern = []
        fallback_index = 0
        index_offset = 0
        occurrence_number = 1
        block_size = first.block_size
        skip_blocks = [
            block
            for block in range(2**block_size)
            if block in set(rng.choice(parents).skip_blocks)
        ]
        skip_blocks = normalize_skip_blocks(skip_blocks, block_size)
        lookup_offsets = [
            rng.choice(parents).lookup_offsets[index]
            for index in range(len(first.lookup_offsets))
        ]
        fallback_policy = rng.choice(parents).fallback_policy
        fsm_state_count = 0
        fsm_start_state = 0
        fsm_outputs = []
        fsm_transitions = []
        fsm_accepting_states = []
        fsm_index_offset = 0
        fsm_fallback_index = 0
    elif first.strategy_type == "fsm":
        genes = []
        pattern = []
        fallback_index = 0
        index_offset = 0
        occurrence_number = 1
        block_size = 0
        skip_blocks = []
        lookup_offsets = []
        fallback_policy = "random-index"
        fsm_state_count = first.fsm_state_count
        fsm_start_state = rng.choice(parents).fsm_start_state
        fsm_outputs = [
            rng.choice(parents).fsm_outputs[index]
            for index in range(len(first.fsm_outputs))
        ]
        fsm_transitions = [
            [
                rng.choice(parents).fsm_transitions[state][bit]
                for bit in range(2)
            ]
            for state in range(first.fsm_state_count)
        ]
        fsm_accepting_states = [
            state
            for state in range(first.fsm_state_count)
            if state in rng.choice(parents).fsm_accepting_states
        ]
        fsm_index_offset = rng.choice(parents).fsm_index_offset
        fsm_fallback_index = rng.choice(parents).fsm_fallback_index
    else:
        raise ValueError(f"Unsupported strategy type: {first.strategy_type}")

    return Strategy(
        id=new_strategy_id(first.player_type, generation, counter),
        player_type=first.player_type,
        observed_length=first.observed_length,
        target_choice_range=first.target_choice_range,
        genes=genes,
        generation_created=generation,
        parent_ids=[parent.id for parent in parents],
        creation_method="gene-join",
        strategy_type=first.strategy_type,
        pattern=pattern,
        fallback_index=fallback_index,
        index_offset=index_offset,
        occurrence_number=occurrence_number,
        block_size=block_size,
        skip_blocks=skip_blocks,
        lookup_offsets=lookup_offsets,
        fallback_policy=fallback_policy,
        fsm_state_count=fsm_state_count,
        fsm_start_state=fsm_start_state,
        fsm_outputs=fsm_outputs,
        fsm_transitions=fsm_transitions,
        fsm_accepting_states=fsm_accepting_states,
        fsm_index_offset=fsm_index_offset,
        fsm_fallback_index=fsm_fallback_index,
    )


def strategy_selection_weight(strategy: Strategy) -> float:
    return max(strategy.survival_ratio, 0.0001)


def weighted_strategy(population: list[Strategy], rng: random.Random) -> Strategy:
    weights = [strategy_selection_weight(strategy) for strategy in population]
    return rng.choices(population, weights=weights, k=1)[0]


def weighted_strategy_sample(
    population: list[Strategy],
    sample_size: int,
    rng: random.Random,
) -> list[Strategy]:
    remaining = list(population)
    selected = []
    for _ in range(min(sample_size, len(remaining))):
        choice = weighted_strategy(remaining, rng)
        selected.append(choice)
        remaining.remove(choice)
    return selected


def weighted_parent(population: list[Strategy], rng: random.Random) -> Strategy:
    return weighted_strategy(population, rng)


def weighted_partner(partners: list[Strategy], rng: random.Random) -> Strategy:
    return weighted_strategy(partners, rng)


def weighted_partners(
    partners: list[Strategy],
    sample_size: int,
    rng: random.Random,
) -> list[Strategy]:
    return weighted_strategy_sample(partners, sample_size, rng)


def compatible_partners(parent: Strategy, population: list[Strategy]) -> list[Strategy]:
    return [
        candidate
        for candidate in population
        if candidate.id != parent.id and compatible(parent, candidate)
    ]


def make_child(
    population: list[Strategy],
    player_type: PlayerType,
    generation: int,
    counter: int,
    config: GAConfig,
    rng: random.Random,
) -> Strategy:
    if not population:
        return random_strategy(player_type, generation, counter, config, rng)

    operator_total = config.mutation_rate + config.crossover_rate + config.gene_join_rate
    method_roll = rng.random() * operator_total if operator_total > 0 else 0
    parent = weighted_parent(population, rng)
    partners = compatible_partners(parent, population)

    if method_roll < config.mutation_rate:
        return mutate_strategy(parent, generation, counter, config, rng)

    if method_roll < config.mutation_rate + config.crossover_rate and partners:
        return crossover_strategy(
            parent,
            weighted_partner(partners, rng),
            generation,
            counter,
            rng,
        )

    if method_roll < operator_total and partners:
        compatible_group = [parent] + weighted_partners(
            partners,
            min(len(partners), rng.randint(1, 3)),
            rng,
        )
        return gene_join_strategy(compatible_group, generation, counter, rng)

    return mutate_strategy(parent, generation, counter, config, rng)


def remove_parents_after_child_production(
    population: list[Strategy],
    parent_ids: list[str],
    max_children_per_strategy: int,
) -> list[Strategy]:
    if not parent_ids:
        return population

    parent_id_set = set(parent_ids)
    for strategy in population:
        if strategy.id in parent_id_set:
            strategy.children_produced += 1

    return active_strategies(population, max_children_per_strategy)


def active_strategies(
    population: list[Strategy],
    max_children_per_strategy: int,
) -> list[Strategy]:
    return [
        strategy
        for strategy in population
        if strategy.children_produced < max_children_per_strategy
    ]


def next_generation(
    population: list[Strategy],
    player_type: PlayerType,
    target_size: int,
    generation: int,
    config: GAConfig,
    rng: random.Random,
) -> list[Strategy]:
    survivors = kill_weak_strategies(
        population,
        config.elite_count,
        config.selection_pressure,
        config.epsilon,
        rng,
    )
    survivors = sorted(survivors, key=lambda strategy: strategy.survival_ratio, reverse=True)
    survivors = survivors[:target_size]
    breeding_pool = list(survivors)

    counter = 0
    while len(survivors) < target_size:
        if breeding_pool:
            child = make_child(breeding_pool, player_type, generation, counter, config, rng)
            breeding_pool = remove_parents_after_child_production(
                breeding_pool,
                child.parent_ids,
                config.max_children_per_strategy,
            )
            survivors = active_strategies(survivors, config.max_children_per_strategy)
        else:
            child = random_strategy(player_type, generation, counter, config, rng)
        survivors.append(child)
        counter += 1

    return survivors


def write_yaml(path: Path, config: GAConfig) -> None:
    with path.open("w", encoding="utf-8") as output:
        for key, value in asdict(config).items():
            output.write(f"{key}: {json.dumps(value)}\n")


def write_population(path: Path, population: list[Strategy]) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump([asdict(strategy) for strategy in population], output, indent=2)


def write_strategy(path: Path, strategy: Strategy) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(asdict(strategy), output, indent=2)


def write_best_pair(path: Path, best_pair: BestPair) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(asdict(best_pair), output, indent=2)


def pattern_text(pattern: list[int]) -> str:
    return "".join(str(bit) for bit in pattern)


def block_list_text(blocks: list[int], block_size: int) -> str:
    if not blocks:
        return "none"
    return ", ".join(block_text(block, block_size) for block in blocks)


def lookup_offsets_text(lookup_offsets: list[int], block_size: int) -> str:
    return ", ".join(
        f"{block_text(block, block_size)} -> {offset}"
        for block, offset in enumerate(lookup_offsets)
    )


def fsm_transitions_text(transitions: list[list[int]]) -> str:
    return ", ".join(
        f"{state}:0->{transition[0]},1->{transition[1]}"
        for state, transition in enumerate(transitions)
    )


def strategy_plain_description(strategy: Strategy) -> str:
    if strategy.strategy_type == "first-pattern":
        occurrence_text = (
            "first"
            if strategy.occurrence_number == 1
            else f"#{strategy.occurrence_number}"
        )
        return (
            "Look through the generated own coin sequence. "
            f"Find the {occurrence_text} occurrence of pattern "
            f"{pattern_text(strategy.pattern)}. "
            f"Add index offset {strategy.index_offset} to that occurrence position. "
            "Choose the resulting position in the other player's sequence. "
            f"If that occurrence is absent, choose index {strategy.fallback_index}."
        )

    if strategy.strategy_type == "gene-table":
        return (
            f"Look at the first {strategy.observed_length} own coin flips. "
            "Use that observed bit pattern as a lookup key in the gene table. "
            "The selected gene gives the target index in the other player's "
            "sequence."
        )

    if strategy.strategy_type == "block-lookup":
        return (
            "Split the generated own coin sequence into consecutive "
            f"{strategy.block_size}-bit blocks. "
            f"Skip blocks {block_list_text(strategy.skip_blocks, strategy.block_size)}. "
            "For the first non-skipped block, use its lookup-table offset and "
            "choose block start index plus that offset in the other player's "
            f"sequence. If every complete block is skipped, use fallback policy "
            f"{strategy.fallback_policy}."
        )

    if strategy.strategy_type == "fsm":
        return (
            "Run a finite-state machine over the generated own coin sequence. "
            f"Start in state {strategy.fsm_start_state}. For each bit, follow "
            "that state's transition for 0 or 1. If the machine reaches an "
            "accepting state, choose acceptance index plus that state's output "
            f"plus index offset {strategy.fsm_index_offset}. If accepting "
            f"states are configured but none is reached, choose fallback index "
            f"{strategy.fsm_fallback_index}. If no accepting state is "
            "configured, choose the target index stored as the final state's "
            "output."
        )

    return f"Unsupported strategy type: {strategy.strategy_type}"


def strategy_table(strategy: Strategy) -> str:
    rows = [
        ("ID", strategy.id),
        ("Player", strategy.player_type),
        ("Type", strategy.strategy_type),
        ("Observed length", observed_length_text(strategy.observed_length)),
        ("Target choice range", target_range_text(strategy.target_choice_range)),
        ("Created in generation", str(strategy.generation_created)),
        ("Creation method", strategy.creation_method),
        ("Best score", f"{strategy.best_score:.2%}"),
        ("Children produced", str(strategy.children_produced)),
    ]
    if strategy.strategy_type == "first-pattern":
        rows.extend(
            [
                ("Pattern", pattern_text(strategy.pattern)),
                ("Occurrence number", str(strategy.occurrence_number)),
                ("Index offset", str(strategy.index_offset)),
                ("Fallback index", str(strategy.fallback_index)),
            ]
        )
    elif strategy.strategy_type == "block-lookup":
        rows.extend(
            [
                ("Block size", str(strategy.block_size)),
                ("Skipped blocks", block_list_text(strategy.skip_blocks, strategy.block_size)),
                (
                    "Lookup offsets",
                    lookup_offsets_text(strategy.lookup_offsets, strategy.block_size),
                ),
                ("Fallback policy", strategy.fallback_policy),
            ]
        )
    elif strategy.strategy_type == "fsm":
        rows.extend(
            [
                ("State count", str(strategy.fsm_state_count)),
                ("Start state", str(strategy.fsm_start_state)),
                ("State outputs", ", ".join(str(output) for output in strategy.fsm_outputs)),
                ("Transitions", fsm_transitions_text(strategy.fsm_transitions)),
                ("Accepting states", ", ".join(str(state) for state in strategy.fsm_accepting_states) or "none"),
                ("FSM index offset", str(strategy.fsm_index_offset)),
                ("FSM fallback index", str(strategy.fsm_fallback_index)),
            ]
        )
    else:
        rows.append(("Gene count", str(len(strategy.genes))))

    table_rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(label)}</th>"
        f"<td>{html.escape(value)}</td>"
        "</tr>"
        for label, value in rows
    )
    description = html.escape(strategy_plain_description(strategy))
    return f"<table>{table_rows}</table><p>{description}</p>"


def write_score_matrix(
    path: Path,
    population_a: list[Strategy],
    population_b: list[Strategy],
    score_matrix: list[list[float]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["strategy_A"] + [strategy.id for strategy in population_b])
        for strategy, row in zip(population_a, score_matrix):
            writer.writerow([strategy.id] + [f"{score:.6f}" for score in row])


def find_best_pair(
    population_a: list[Strategy],
    population_b: list[Strategy],
    score_matrix: list[list[float]],
) -> BestPair:
    best_score = -1.0
    best_a = population_a[0]
    best_b = population_b[0]

    for row_index, row in enumerate(score_matrix):
        for column_index, score in enumerate(row):
            if score > best_score:
                best_score = score
                best_a = population_a[row_index]
                best_b = population_b[column_index]

    return BestPair(best_a.id, best_b.id, best_score)


def find_best_pairs(
    population_a: list[Strategy],
    population_b: list[Strategy],
    score_matrix: list[list[float]],
    limit: int,
) -> list[BestPair]:
    best_by_signature_pair: dict[tuple[tuple, tuple], BestPair] = {}
    for row_index, row in enumerate(score_matrix):
        strategy_a = population_a[row_index]
        signature_a = strategy_signature(strategy_a)
        for column_index, score in enumerate(row):
            strategy_b = population_b[column_index]
            signature_pair = (signature_a, strategy_signature(strategy_b))
            current_best = best_by_signature_pair.get(signature_pair)
            if current_best is None or score > current_best.score:
                best_by_signature_pair[signature_pair] = BestPair(
                    strategy_a.id,
                    strategy_b.id,
                    score,
                )

    return sorted(
        best_by_signature_pair.values(),
        key=lambda pair: pair.score,
        reverse=True,
    )[:limit]


def strategy_by_id(population: list[Strategy], strategy_id: str) -> Strategy:
    for strategy in population:
        if strategy.id == strategy_id:
            return strategy
    raise ValueError(f"Strategy not found: {strategy_id}")


def best_pair_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as history_file:
        return list(csv.DictReader(history_file))


def load_strategy_from_snapshot(
    output_dir: Path,
    player_type: PlayerType,
    generation: int,
    strategy_id: str,
) -> Strategy:
    snapshot_path = (
        output_dir
        / "population_snapshots"
        / f"population_{player_type}_generation_{generation:04d}.json"
    )
    with snapshot_path.open(encoding="utf-8") as snapshot_file:
        strategies = json.load(snapshot_file)

    for strategy_data in strategies:
        if strategy_data["id"] == strategy_id:
            return Strategy(**strategy_data)
    raise ValueError(f"Strategy not found in snapshot: {strategy_id}")


def best_history_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row["score"]))


def write_html_report(
    output_dir: Path,
    generation: int,
    config: GAConfig,
    population_a: list[Strategy],
    population_b: list[Strategy],
    score_matrix: list[list[float]],
) -> None:
    current_pair = find_best_pair(population_a, population_b, score_matrix)
    current_a = strategy_by_id(population_a, current_pair.strategy_A_id)
    current_b = strategy_by_id(population_b, current_pair.strategy_B_id)
    top_pairs = find_best_pairs(population_a, population_b, score_matrix, 3)
    top_pair_sections = "\n".join(
        f"""
        <article>
          <h3>#{rank}: {pair.score:.2%}</h3>
          <p>
            <code>{html.escape(pair.strategy_A_id)}</code> with
            <code>{html.escape(pair.strategy_B_id)}</code>
          </p>
          <div class="grid">
            <div>
              <h4>Player A Strategy</h4>
              {strategy_table(strategy_by_id(population_a, pair.strategy_A_id))}
            </div>
            <div>
              <h4>Player B Strategy</h4>
              {strategy_table(strategy_by_id(population_b, pair.strategy_B_id))}
            </div>
          </div>
        </article>
        """
        for rank, pair in enumerate(top_pairs, start=1)
    )
    rows = best_pair_history(output_dir / "best_pairs.csv")
    overall_row = best_history_row(rows)

    overall_section = ""
    if overall_row is not None:
        overall_generation = int(overall_row["generation"])
        overall_a = load_strategy_from_snapshot(
            output_dir,
            "A",
            overall_generation,
            overall_row["strategy_A"],
        )
        overall_b = load_strategy_from_snapshot(
            output_dir,
            "B",
            overall_generation,
            overall_row["strategy_B"],
        )
        overall_section = f"""
        <section>
          <h2>Best Pair Seen During The Run</h2>
          <p>
            Generation {overall_generation} reached
            <strong>{float(overall_row["score"]):.2%}</strong> with
            <code>{html.escape(overall_row["strategy_A"])}</code> and
            <code>{html.escape(overall_row["strategy_B"])}</code>.
          </p>
          <div class="grid">
            <article>
              <h3>Player A Strategy</h3>
              {strategy_table(overall_a)}
            </article>
            <article>
              <h3>Player B Strategy</h3>
              {strategy_table(overall_b)}
            </article>
          </div>
        </section>
        """

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Daemon Coin Strategy Report</title>
  <style>
    body {{
      color: #202124;
      font-family: Arial, sans-serif;
      line-height: 1.5;
      margin: 32px auto;
      max-width: 1100px;
      padding: 0 20px;
    }}
    h1, h2, h3 {{ line-height: 1.2; }}
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
    .grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    article {{
      border: 1px solid #dadce0;
      border-radius: 8px;
      padding: 16px;
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
    th {{ width: 42%; }}
    h4 {{ margin-bottom: 8px; }}
  </style>
</head>
<body>
  <h1>Daemon Coin Strategy Report</h1>
  <p>
    This report explains the strongest strategy pairs in plain language.
    A strategy observes its own coin flips and returns a target index in the
    other player's hidden sequence.
  </p>

  <section>
    <h2>Final Generation Best Pair</h2>
    <p>
      Generation {generation} ended with
      <strong>{current_pair.score:.2%}</strong> for
      <code>{html.escape(current_pair.strategy_A_id)}</code> and
      <code>{html.escape(current_pair.strategy_B_id)}</code>.
    </p>
    <div class="grid">
      <article>
        <h3>Player A Strategy</h3>
        {strategy_table(current_a)}
      </article>
      <article>
        <h3>Player B Strategy</h3>
        {strategy_table(current_b)}
      </article>
    </div>
  </section>

  <section>
    <h2>Top 3 Distinct Final Generation Pairs</h2>
    <p>
      These are the three highest-scoring distinct A/B strategy definitions in
      generation {generation}. Copies and equivalent descendants are grouped.
    </p>
    <div class="pair-list">
      {top_pair_sections}
    </div>
  </section>

  {overall_section}

  <section>
    <h2>Configuration Summary</h2>
    <table>
      <tr><th>Generations</th><td>{config.generations}</td></tr>
      <tr><th>Trials per pair</th><td>{config.trials_per_pair}</td></tr>
      <tr><th>Simulation sequence length</th><td>{config.sequence_length}</td></tr>
      <tr><th>Population A</th><td>{config.population_size_A}</td></tr>
      <tr><th>Population B</th><td>{config.population_size_B}</td></tr>
      <tr><th>Max children per strategy</th><td>{config.max_children_per_strategy}</td></tr>
      <tr><th>Success condition</th><td>{html.escape(config.success_condition)}</td></tr>
      <tr><th>Default strategy type</th><td>{html.escape(config.default_strategy_type)}</td></tr>
    </table>
  </section>
</body>
</html>
"""
    (output_dir / "experiment_report.html").write_text(report, encoding="utf-8")


def append_generation_stats(
    path: Path,
    generation: int,
    population_a: list[Strategy],
    population_b: list[Strategy],
) -> None:
    write_header = generation == 0 or not path.exists()
    mode = "w" if generation == 0 else "a"
    with path.open(mode, newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        if write_header:
            writer.writerow(
                [
                    "generation",
                    "best_A",
                    "avg_A",
                    "best_B",
                    "avg_B",
                    "best_A_id",
                    "best_B_id",
                ]
            )
        best_a = max(population_a, key=lambda strategy: strategy.survival_ratio)
        best_b = max(population_b, key=lambda strategy: strategy.survival_ratio)
        writer.writerow(
            [
                generation,
                f"{best_a.survival_ratio:.6f}",
                f"{mean(strategy.survival_ratio for strategy in population_a):.6f}",
                f"{best_b.survival_ratio:.6f}",
                f"{mean(strategy.survival_ratio for strategy in population_b):.6f}",
                best_a.id,
                best_b.id,
            ]
        )


def append_best_pairs(
    path: Path,
    generation: int,
    population_a: list[Strategy],
    population_b: list[Strategy],
    score_matrix: list[list[float]],
) -> None:
    write_header = generation == 0 or not path.exists()
    mode = "w" if generation == 0 else "a"
    with path.open(mode, newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        if write_header:
            writer.writerow(["generation", "strategy_A", "strategy_B", "score"])

        best_pair = find_best_pair(population_a, population_b, score_matrix)
        writer.writerow(
            [
                generation,
                best_pair.strategy_A_id,
                best_pair.strategy_B_id,
                f"{best_pair.score:.6f}",
            ]
        )


def save_artifacts(
    output_dir: Path,
    generation: int,
    config: GAConfig,
    population_a: list[Strategy],
    population_b: list[Strategy],
    score_matrix: list[list[float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "population_snapshots"
    matrices_dir = output_dir / "score_matrices"
    snapshots_dir.mkdir(exist_ok=True)
    matrices_dir.mkdir(exist_ok=True)

    if generation == 0:
        write_yaml(output_dir / "experiment_config.yaml", config)

    append_generation_stats(
        output_dir / "generation_stats.csv",
        generation,
        population_a,
        population_b,
    )
    append_best_pairs(
        output_dir / "best_pairs.csv",
        generation,
        population_a,
        population_b,
        score_matrix,
    )
    write_population(
        snapshots_dir / f"population_A_generation_{generation:04d}.json",
        population_a,
    )
    write_population(
        snapshots_dir / f"population_B_generation_{generation:04d}.json",
        population_b,
    )
    write_score_matrix(
        matrices_dir / f"score_matrix_generation_{generation:04d}.csv",
        population_a,
        population_b,
        score_matrix,
    )
    write_strategy(
        output_dir / "best_A_strategy.json",
        max(population_a, key=lambda strategy: strategy.survival_ratio),
    )
    write_strategy(
        output_dir / "best_B_strategy.json",
        max(population_b, key=lambda strategy: strategy.survival_ratio),
    )
    write_best_pair(
        output_dir / "best_pair.json",
        find_best_pair(population_a, population_b, score_matrix),
    )
    write_html_report(
        output_dir,
        generation,
        config,
        population_a,
        population_b,
        score_matrix,
    )


def run_genetic_algorithm(
    config: GAConfig,
    custom_success_condition: SuccessCondition | None = None,
    include_best_pair: bool = False,
) -> tuple[Strategy, Strategy] | tuple[Strategy, Strategy, BestPair]:
    rng = random.Random(config.seed)
    success_condition = custom_success_condition or success_condition_from_name(
        config.success_condition
    )
    population_a = initial_population(
        "A",
        config.population_size_A,
        config.initial_population_A,
        config,
        rng,
    )
    population_b = initial_population(
        "B",
        config.population_size_B,
        config.initial_population_B,
        config,
        rng,
    )
    output_dir = Path(config.output_dir)

    for generation in range(config.generations + 1):
        score_matrix = evaluate_populations(
            population_a,
            population_b,
            config,
            success_condition,
            rng,
        )
        save_artifacts(output_dir, generation, config, population_a, population_b, score_matrix)

        best_a = max(population_a, key=lambda strategy: strategy.survival_ratio)
        best_b = max(population_b, key=lambda strategy: strategy.survival_ratio)
        best_pair = find_best_pair(population_a, population_b, score_matrix)
        logging.info(
            (
                "generation=%s best_pair=%.4f pair=[%s %s] "
                "best_A=%.4f best_B=%.4f"
            ),
            generation,
            best_pair.score,
            best_pair.strategy_A_id,
            best_pair.strategy_B_id,
            best_a.survival_ratio,
            best_b.survival_ratio,
        )

        if generation == config.generations:
            if include_best_pair:
                return best_a, best_b, best_pair
            return best_a, best_b

        population_a = next_generation(
            population_a,
            "A",
            next_generation_size(len(population_a), config.population_size_A),
            generation + 1,
            config,
            rng,
        )
        population_b = next_generation(
            population_b,
            "B",
            next_generation_size(len(population_b), config.population_size_B),
            generation + 1,
            config,
            rng,
        )

    raise RuntimeError("GA loop ended unexpectedly")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def load_config_file(path: Path) -> GAConfig:
    with path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    config_values = raw_config.get(CONFIG_SECTION, raw_config)
    if not isinstance(config_values, dict):
        raise ValueError(f"{path} must contain TOML key/value pairs")

    allowed_fields = {field.name for field in fields(GAConfig)}
    unknown_fields = sorted(set(config_values) - allowed_fields)
    if unknown_fields:
        unknown = ", ".join(unknown_fields)
        raise ValueError(f"Unknown config field(s) in {path}: {unknown}")

    config = GAConfig(**config_values)
    validate_config(config)
    return config


def apply_recommended_preset(config: GAConfig) -> None:
    config.population_size_A = 300
    config.population_size_B = 300
    config.observed_length_min = 3
    config.observed_length_max = 12
    config.target_choice_range_min = 3
    config.target_choice_range_max = 12
    config.trials_per_pair = 1000
    config.sequence_length = 1000
    config.generations = 500


def expanded_seed_count(seed_configs: list[dict]) -> int:
    return sum(
        seed.get("population", 1)
        if isinstance(seed.get("population", 1), int)
        else 1
        for seed in seed_configs
    )


def add_known_triple_seed(config: GAConfig) -> None:
    if not any(
        seed.get("id") == "A-canonical-triple"
        for seed in config.initial_population_A
    ):
        config.initial_population_A.append(canonical_triple_seed("A"))
        config.population_size_A = max(
            config.population_size_A,
            expanded_seed_count(config.initial_population_A),
        )
    if not any(
        seed.get("id") == "B-canonical-triple"
        for seed in config.initial_population_B
    ):
        config.initial_population_B.append(canonical_triple_seed("B"))
        config.population_size_B = max(
            config.population_size_B,
            expanded_seed_count(config.initial_population_B),
        )


def apply_cli_overrides(config: GAConfig, args: argparse.Namespace) -> None:
    for field in fields(GAConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            setattr(config, field.name, value)


def validate_initial_strategy(
    player_type: PlayerType,
    strategy_config: dict,
    index: int,
    config: GAConfig,
) -> None:
    if not isinstance(strategy_config, dict):
        raise ValueError(f"initial_population_{player_type}[{index}] must be a table")

    allowed_fields = {
        "id",
        "population",
        "strategy_type",
        "observed_length",
        "target_choice_range",
        "fallback_index",
        "index_offset",
        "occurrence_number",
        "genes",
        "pattern",
        "block_size",
        "skip_blocks",
        "lookup_offsets",
        "fallback_policy",
        "fsm_state_count",
        "fsm_start_state",
        "fsm_outputs",
        "fsm_transitions",
        "fsm_accepting_states",
        "fsm_index_offset",
        "fsm_fallback_index",
    }
    unknown_fields = sorted(set(strategy_config) - allowed_fields)
    if unknown_fields:
        unknown = ", ".join(unknown_fields)
        raise ValueError(
            f"Unknown field(s) in initial_population_{player_type}[{index}]: {unknown}"
        )

    strategy_type = strategy_config.get("strategy_type")
    if strategy_type is None:
        if "fsm_outputs" in strategy_config or "fsm_transitions" in strategy_config:
            strategy_type = "fsm"
        elif "block_size" in strategy_config or "lookup_offsets" in strategy_config:
            strategy_type = "block-lookup"
        else:
            strategy_type = "first-pattern" if "pattern" in strategy_config else "gene-table"
    if strategy_type not in SUPPORTED_STRATEGY_TYPES:
        raise ValueError(
            f"initial_population_{player_type}[{index}].strategy_type is unsupported"
        )

    required_fields = set()
    if strategy_type == "gene-table":
        required_fields.add("observed_length")
        required_fields.add("target_choice_range")
        required_fields.add("genes")
    elif strategy_type == "first-pattern":
        required_fields.add("pattern")
    elif strategy_type == "block-lookup":
        required_fields.add("block_size")
        required_fields.add("lookup_offsets")
    else:
        required_fields.add("fsm_outputs")
        required_fields.add("fsm_transitions")
    missing_fields = sorted(required_fields - set(strategy_config))
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(
            f"Missing field(s) in initial_population_{player_type}[{index}]: {missing}"
        )

    observed_length = strategy_config.get("observed_length", 0)
    population_count = strategy_config.get("population", 1)
    target_choice_range = strategy_config.get("target_choice_range", 0)
    fallback_index = strategy_config.get(
        "fallback_index",
        config.first_pattern_fallback_index_min,
    )
    index_offset = strategy_config.get(
        "index_offset",
        config.first_pattern_index_offset_min,
    )
    occurrence_number = strategy_config.get("occurrence_number", 1)
    genes = strategy_config.get("genes", [])
    pattern = strategy_config.get("pattern", [])
    block_size = strategy_config.get("block_size", 0)
    skip_blocks = strategy_config.get("skip_blocks", [])
    lookup_offsets = strategy_config.get("lookup_offsets", [])
    fallback_policy = strategy_config.get(
        "fallback_policy",
        config.block_lookup_fallback_policy,
    )
    fsm_state_count = strategy_config.get("fsm_state_count", len(strategy_config.get("fsm_outputs", [])))
    fsm_start_state = strategy_config.get("fsm_start_state", 0)
    fsm_outputs = strategy_config.get("fsm_outputs", [])
    fsm_transitions = strategy_config.get("fsm_transitions", [])
    fsm_accepting_states = strategy_config.get("fsm_accepting_states", [])
    fsm_index_offset = strategy_config.get("fsm_index_offset", 0)
    fsm_fallback_index = strategy_config.get("fsm_fallback_index", 0)

    if not isinstance(observed_length, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].observed_length must be int"
        )
    if not isinstance(population_count, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].population must be int"
        )
    if population_count <= 0:
        raise ValueError(
            f"initial_population_{player_type}[{index}].population must be positive"
        )
    if not isinstance(target_choice_range, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].target_choice_range must be int"
        )
    if not isinstance(fallback_index, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].fallback_index must be int"
        )
    if not isinstance(index_offset, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].index_offset must be int"
        )
    if not isinstance(occurrence_number, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].occurrence_number must be int"
        )
    if not isinstance(genes, list):
        raise ValueError(f"initial_population_{player_type}[{index}].genes must be a list")
    if not isinstance(pattern, list):
        raise ValueError(
            f"initial_population_{player_type}[{index}].pattern must be a list"
        )
    if not isinstance(block_size, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].block_size must be int"
        )
    if not isinstance(skip_blocks, list):
        raise ValueError(
            f"initial_population_{player_type}[{index}].skip_blocks must be a list"
        )
    if not isinstance(lookup_offsets, list):
        raise ValueError(
            f"initial_population_{player_type}[{index}].lookup_offsets must be a list"
        )
    if not isinstance(fallback_policy, str):
        raise ValueError(
            f"initial_population_{player_type}[{index}].fallback_policy must be str"
        )
    if not isinstance(fsm_state_count, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].fsm_state_count must be int"
        )
    if not isinstance(fsm_start_state, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].fsm_start_state must be int"
        )
    if not isinstance(fsm_outputs, list):
        raise ValueError(
            f"initial_population_{player_type}[{index}].fsm_outputs must be a list"
        )
    if not isinstance(fsm_transitions, list):
        raise ValueError(
            f"initial_population_{player_type}[{index}].fsm_transitions must be a list"
        )
    if not isinstance(fsm_accepting_states, list):
        raise ValueError(
            f"initial_population_{player_type}[{index}].fsm_accepting_states must be a list"
        )
    if not isinstance(fsm_index_offset, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].fsm_index_offset must be int"
        )
    if not isinstance(fsm_fallback_index, int):
        raise ValueError(
            f"initial_population_{player_type}[{index}].fsm_fallback_index must be int"
        )

    if (
        strategy_type == "gene-table"
        and not config.observed_length_min
        <= observed_length
        <= config.observed_length_max
    ):
        raise ValueError(
            f"initial_population_{player_type}[{index}].observed_length is outside "
            "configured bounds"
        )
    if (
        strategy_type == "first-pattern"
        and observed_length != 0
        and not config.observed_length_min
        <= observed_length
        <= config.observed_length_max
    ):
        raise ValueError(
            f"initial_population_{player_type}[{index}].observed_length is outside "
            "configured bounds"
        )
    if (
        strategy_type == "gene-table"
        and not config.target_choice_range_min
        <= target_choice_range
        <= config.target_choice_range_max
    ):
        raise ValueError(
            f"initial_population_{player_type}[{index}].target_choice_range is outside "
            "configured bounds"
        )
    if strategy_type == "gene-table" and target_choice_range <= 0:
        raise ValueError(
            f"initial_population_{player_type}[{index}].target_choice_range must be "
            "positive for gene-table"
        )
    if strategy_type == "gene-table" and len(genes) != 2**observed_length:
        raise ValueError(
            f"initial_population_{player_type}[{index}].genes must contain "
            f"{2**observed_length} values"
        )
    if strategy_type == "gene-table":
        if not all(isinstance(gene, int) for gene in genes):
            raise ValueError(f"initial_population_{player_type}[{index}].genes must be ints")
        if not all(0 <= gene < target_choice_range for gene in genes):
            raise ValueError(
                f"initial_population_{player_type}[{index}].genes must be within "
                "target_choice_range"
            )

    if strategy_type == "first-pattern":
        if not 0 <= fallback_index < config.sequence_length:
            raise ValueError(
                f"initial_population_{player_type}[{index}].fallback_index must be "
                "within sequence_length"
            )
        if not (
            config.first_pattern_index_offset_min
            <= index_offset
            <= config.first_pattern_index_offset_max
        ):
            raise ValueError(
                f"initial_population_{player_type}[{index}].index_offset is outside "
                "configured bounds"
            )
        if not (
            config.first_pattern_occurrence_min
            <= occurrence_number
            <= config.first_pattern_occurrence_max
        ):
            raise ValueError(
                f"initial_population_{player_type}[{index}].occurrence_number is "
                "outside configured bounds"
            )
        if not pattern:
            raise ValueError(
                f"initial_population_{player_type}[{index}].pattern cannot be empty"
            )
        if len(pattern) > config.sequence_length:
            raise ValueError(
                f"initial_population_{player_type}[{index}].pattern cannot be "
                "longer than sequence_length"
            )
        if observed_length != 0 and len(pattern) > observed_length:
            raise ValueError(
                f"initial_population_{player_type}[{index}].pattern cannot be "
                "longer than observed_length"
            )
        if not all(bit in {0, 1} for bit in pattern):
            raise ValueError(
                f"initial_population_{player_type}[{index}].pattern must contain bits"
            )
    if strategy_type == "block-lookup":
        if observed_length != 0 and not config.observed_length_min <= observed_length <= config.observed_length_max:
            raise ValueError(
                f"initial_population_{player_type}[{index}].observed_length is outside "
                "configured bounds"
            )
        if not (
            config.block_lookup_block_size_min
            <= block_size
            <= config.block_lookup_block_size_max
        ):
            raise ValueError(
                f"initial_population_{player_type}[{index}].block_size is outside "
                "configured bounds"
            )
        block_count = 2**block_size
        if len(lookup_offsets) != block_count:
            raise ValueError(
                f"initial_population_{player_type}[{index}].lookup_offsets must contain "
                f"{block_count} values"
            )
        if not all(isinstance(offset, int) for offset in lookup_offsets):
            raise ValueError(
                f"initial_population_{player_type}[{index}].lookup_offsets must be ints"
            )
        if not all(0 <= offset < block_size for offset in lookup_offsets):
            raise ValueError(
                f"initial_population_{player_type}[{index}].lookup_offsets must be "
                "local block offsets"
            )
        if not all(isinstance(block, int) for block in skip_blocks):
            raise ValueError(
                f"initial_population_{player_type}[{index}].skip_blocks must be ints"
            )
        if not all(0 <= block < block_count for block in skip_blocks):
            raise ValueError(
                f"initial_population_{player_type}[{index}].skip_blocks must be valid "
                "block values"
            )
        if len(set(skip_blocks)) != len(skip_blocks):
            raise ValueError(
                f"initial_population_{player_type}[{index}].skip_blocks cannot contain "
                "duplicates"
            )
        if len(skip_blocks) == block_count:
            raise ValueError(
                f"initial_population_{player_type}[{index}].skip_blocks cannot skip "
                "every block"
            )
        normalize_fallback_policy(fallback_policy)
    if strategy_type == "fsm":
        if observed_length != 0 and not config.observed_length_min <= observed_length <= config.observed_length_max:
            raise ValueError(
                f"initial_population_{player_type}[{index}].observed_length is outside "
                "configured bounds"
            )
        if not config.fsm_state_count_min <= fsm_state_count <= config.fsm_state_count_max:
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_state_count is outside "
                "configured bounds"
            )
        if not 0 <= fsm_start_state < fsm_state_count:
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_start_state must be "
                "a valid state"
            )
        if len(fsm_outputs) != fsm_state_count:
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_outputs must contain "
                f"{fsm_state_count} values"
            )
        if len(fsm_transitions) != fsm_state_count:
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_transitions must contain "
                f"{fsm_state_count} rows"
            )
        if not all(isinstance(state, int) for state in fsm_accepting_states):
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_accepting_states must be ints"
            )
        if not all(0 <= state < fsm_state_count for state in fsm_accepting_states):
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_accepting_states must "
                "reference valid states"
            )
        if len(set(fsm_accepting_states)) != len(fsm_accepting_states):
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_accepting_states "
                "cannot contain duplicates"
            )
        if not config.fsm_index_offset_min <= fsm_index_offset <= config.fsm_index_offset_max:
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_index_offset is outside "
                "configured bounds"
            )
        if not config.fsm_fallback_index_min <= fsm_fallback_index <= config.fsm_fallback_index_max:
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_fallback_index is outside "
                "configured bounds"
            )
        if not all(isinstance(output, int) for output in fsm_outputs):
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_outputs must be ints"
            )
        if not all(
            config.fsm_output_index_min <= output <= config.fsm_output_index_max
            for output in fsm_outputs
        ):
            raise ValueError(
                f"initial_population_{player_type}[{index}].fsm_outputs are outside "
                "configured bounds"
            )
        for row_index, transition in enumerate(fsm_transitions):
            if (
                not isinstance(transition, list)
                or len(transition) != 2
                or not all(isinstance(state, int) for state in transition)
            ):
                raise ValueError(
                    f"initial_population_{player_type}[{index}].fsm_transitions"
                    f"[{row_index}] must contain two state ints"
                )
            if not all(0 <= state < fsm_state_count for state in transition):
                raise ValueError(
                    f"initial_population_{player_type}[{index}].fsm_transitions"
                    f"[{row_index}] must reference valid states"
                )


def validate_initial_population(
    player_type: PlayerType,
    population: list[dict],
    population_size: int,
    config: GAConfig,
) -> None:
    if not isinstance(population, list):
        raise ValueError(f"initial_population_{player_type} must be a list")
    expanded_size = 0
    for strategy_config in population:
        if isinstance(strategy_config, dict):
            population_count = strategy_config.get("population", 1)
            if isinstance(population_count, int):
                expanded_size += population_count
            else:
                expanded_size += 1
        else:
            expanded_size += 1
    if expanded_size > population_size:
        raise ValueError(
            f"initial_population_{player_type} cannot exceed population size"
        )

    seen_ids = set()
    for index, strategy_config in enumerate(population):
        validate_initial_strategy(player_type, strategy_config, index, config)
        strategy_id = strategy_config.get("id")
        if strategy_id is None:
            continue
        if not isinstance(strategy_id, str):
            raise ValueError(f"initial_population_{player_type}[{index}].id must be str")
        if strategy_id in seen_ids:
            raise ValueError(f"Duplicate initial_population_{player_type} id: {strategy_id}")
        seen_ids.add(strategy_id)


def parse_args(argv: Iterable[str]) -> GAConfig:
    parser = argparse.ArgumentParser(
        description="Genetic algorithm for asymmetric variable-length daemon coin strategies."
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="TOML configuration file. Defaults to config.toml when present.",
    )
    parser.add_argument(
        "--population-size-a",
        type=positive_int,
        dest="population_size_A",
    )
    parser.add_argument(
        "--population-size-b",
        type=positive_int,
        dest="population_size_B",
    )
    parser.add_argument("--generations", type=int)
    parser.add_argument("--trials-per-pair", type=positive_int, dest="trials_per_pair")
    parser.add_argument("--sequence-length", type=positive_int, dest="sequence_length")
    parser.add_argument(
        "--observed-length-min",
        type=positive_int,
        dest="observed_length_min",
    )
    parser.add_argument(
        "--observed-length-max",
        type=positive_int,
        dest="observed_length_max",
    )
    parser.add_argument(
        "--target-choice-range-min",
        type=non_negative_int,
        dest="target_choice_range_min",
    )
    parser.add_argument(
        "--target-choice-range-max",
        type=non_negative_int,
        dest="target_choice_range_max",
    )
    parser.add_argument(
        "--first-pattern-observed-length-min",
        type=non_negative_int,
        dest="first_pattern_observed_length_min",
    )
    parser.add_argument(
        "--first-pattern-observed-length-max",
        type=non_negative_int,
        dest="first_pattern_observed_length_max",
    )
    parser.add_argument(
        "--first-pattern-fallback-index-min",
        type=non_negative_int,
        dest="first_pattern_fallback_index_min",
    )
    parser.add_argument(
        "--first-pattern-fallback-index-max",
        type=non_negative_int,
        dest="first_pattern_fallback_index_max",
    )
    parser.add_argument(
        "--first-pattern-index-offset-min",
        type=int,
        dest="first_pattern_index_offset_min",
    )
    parser.add_argument(
        "--first-pattern-index-offset-max",
        type=int,
        dest="first_pattern_index_offset_max",
    )
    parser.add_argument(
        "--first-pattern-occurrence-min",
        type=positive_int,
        dest="first_pattern_occurrence_min",
    )
    parser.add_argument(
        "--first-pattern-occurrence-max",
        type=positive_int,
        dest="first_pattern_occurrence_max",
    )
    parser.add_argument(
        "--block-lookup-block-size-min",
        type=positive_int,
        dest="block_lookup_block_size_min",
    )
    parser.add_argument(
        "--block-lookup-block-size-max",
        type=positive_int,
        dest="block_lookup_block_size_max",
    )
    parser.add_argument(
        "--block-lookup-fallback-policy",
        choices=sorted(BLOCK_LOOKUP_FALLBACK_POLICIES),
        dest="block_lookup_fallback_policy",
    )
    parser.add_argument(
        "--block-size",
        type=positive_int,
        help="Shortcut for setting both block lookup block-size bounds.",
    )
    parser.add_argument("--fsm-state-count-min", type=positive_int, dest="fsm_state_count_min")
    parser.add_argument("--fsm-state-count-max", type=positive_int, dest="fsm_state_count_max")
    parser.add_argument("--fsm-output-index-min", type=non_negative_int, dest="fsm_output_index_min")
    parser.add_argument("--fsm-output-index-max", type=non_negative_int, dest="fsm_output_index_max")
    parser.add_argument("--fsm-index-offset-min", type=int, dest="fsm_index_offset_min")
    parser.add_argument("--fsm-index-offset-max", type=int, dest="fsm_index_offset_max")
    parser.add_argument("--fsm-fallback-index-min", type=non_negative_int, dest="fsm_fallback_index_min")
    parser.add_argument("--fsm-fallback-index-max", type=non_negative_int, dest="fsm_fallback_index_max")
    parser.add_argument("--elite-count", type=int, dest="elite_count")
    parser.add_argument("--selection-pressure", type=float, dest="selection_pressure")
    parser.add_argument("--mutation-rate", type=float, dest="mutation_rate")
    parser.add_argument("--crossover-rate", type=float, dest="crossover_rate")
    parser.add_argument("--gene-join-rate", type=float, dest="gene_join_rate")
    parser.add_argument(
        "--gene-mutation-probability",
        type=float,
        dest="gene_mutation_probability",
    )
    parser.add_argument(
        "--structure-mutation-probability",
        type=float,
        dest="structure_mutation_probability",
    )
    parser.add_argument(
        "--max-children-per-strategy",
        type=positive_int,
        dest="max_children_per_strategy",
    )
    parser.add_argument(
        "--disable-observed-length-mutation",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="disable_observed_length_mutation",
    )
    parser.add_argument(
        "--disable-fallback-index-mutation",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="disable_fallback_index_mutation",
    )
    parser.add_argument(
        "--success-condition",
        choices=["same-bit", "both-one"],
        dest="success_condition",
    )
    parser.add_argument(
        "--default-strategy-type",
        choices=sorted(SUPPORTED_STRATEGY_TYPES),
        dest="default_strategy_type",
    )
    parser.add_argument(
        "--strategy-family",
        choices=sorted(SUPPORTED_STRATEGY_TYPES),
        dest="default_strategy_type",
        help="Alias for --default-strategy-type.",
    )
    parser.add_argument(
        "--seed-known-triple-strategy",
        action="store_true",
        help="Append the canonical 3-bit block lookup strategy to both populations.",
    )
    parser.add_argument(
        "--allow-asymmetric",
        action="store_true",
        help="Accepted for spec-compatible commands; A/B populations are already asymmetric.",
    )
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--recommended-preset",
        action="store_true",
        help="Use the large configuration recommended by the PDF specification.",
    )
    args = parser.parse_args(list(argv))

    config_path = Path(args.config)
    if config_path.exists():
        config = load_config_file(config_path)
    else:
        config = GAConfig()

    if args.recommended_preset:
        apply_recommended_preset(config)

    apply_cli_overrides(config, args)
    if args.block_size is not None:
        config.block_lookup_block_size_min = args.block_size
        config.block_lookup_block_size_max = args.block_size
    if args.seed_known_triple_strategy:
        add_known_triple_seed(config)
    validate_config(config)
    return config


def validate_config(config: GAConfig) -> None:
    if config.observed_length_min > config.observed_length_max:
        raise ValueError("observed_length_min cannot exceed observed_length_max")
    if config.target_choice_range_min > config.target_choice_range_max:
        raise ValueError("target_choice_range_min cannot exceed target_choice_range_max")
    if config.elite_count < 0:
        raise ValueError("elite_count cannot be negative")
    if config.elite_count >= min(config.population_size_A, config.population_size_B):
        raise ValueError("elite_count must be smaller than both populations")
    if config.generations < 0:
        raise ValueError("generations cannot be negative")
    if config.max_children_per_strategy <= 0:
        raise ValueError("max_children_per_strategy must be positive")
    if config.first_pattern_fallback_index_min < 0:
        raise ValueError("first_pattern_fallback_index_min cannot be negative")
    if config.first_pattern_fallback_index_max < config.first_pattern_fallback_index_min:
        raise ValueError(
            "first_pattern_fallback_index_max cannot be smaller than min"
        )
    if config.first_pattern_fallback_index_max >= config.sequence_length:
        raise ValueError(
            "first_pattern_fallback_index_max must be within sequence_length"
        )
    if config.first_pattern_index_offset_min > config.first_pattern_index_offset_max:
        raise ValueError(
            "first_pattern_index_offset_min cannot exceed max"
        )
    if config.first_pattern_occurrence_min <= 0:
        raise ValueError("first_pattern_occurrence_min must be positive")
    if config.first_pattern_occurrence_max < config.first_pattern_occurrence_min:
        raise ValueError(
            "first_pattern_occurrence_max cannot be smaller than min"
        )
    if config.first_pattern_observed_length_min < 0:
        raise ValueError("first_pattern_observed_length_min cannot be negative")
    if config.first_pattern_observed_length_max < config.first_pattern_observed_length_min:
        raise ValueError(
            "first_pattern_observed_length_max cannot be smaller than min"
        )
    if config.first_pattern_observed_length_max > config.sequence_length:
        raise ValueError("first_pattern_observed_length_max cannot exceed sequence_length")
    if config.observed_length_max > config.sequence_length:
        raise ValueError("observed_length_max cannot exceed sequence_length")
    if config.block_lookup_block_size_min <= 0:
        raise ValueError("block_lookup_block_size_min must be positive")
    if config.block_lookup_block_size_max < config.block_lookup_block_size_min:
        raise ValueError("block_lookup_block_size_max cannot be smaller than min")
    if config.block_lookup_block_size_max > config.sequence_length:
        raise ValueError("block_lookup_block_size_max cannot exceed sequence_length")
    normalize_fallback_policy(config.block_lookup_fallback_policy)
    if config.fsm_state_count_min <= 0:
        raise ValueError("fsm_state_count_min must be positive")
    if config.fsm_state_count_max < config.fsm_state_count_min:
        raise ValueError("fsm_state_count_max cannot be smaller than min")
    if config.fsm_output_index_min < 0:
        raise ValueError("fsm_output_index_min cannot be negative")
    if config.fsm_output_index_max < config.fsm_output_index_min:
        raise ValueError("fsm_output_index_max cannot be smaller than min")
    if config.fsm_output_index_max >= config.sequence_length:
        raise ValueError("fsm_output_index_max must be within sequence_length")
    if config.fsm_index_offset_min > config.fsm_index_offset_max:
        raise ValueError("fsm_index_offset_min cannot exceed max")
    if config.fsm_fallback_index_min < 0:
        raise ValueError("fsm_fallback_index_min cannot be negative")
    if config.fsm_fallback_index_max < config.fsm_fallback_index_min:
        raise ValueError("fsm_fallback_index_max cannot be smaller than min")
    if config.fsm_fallback_index_max >= config.sequence_length:
        raise ValueError("fsm_fallback_index_max must be within sequence_length")
    if config.default_strategy_type not in SUPPORTED_STRATEGY_TYPES:
        raise ValueError("default_strategy_type is unsupported")
    if config.default_strategy_type == "gene-table" and config.target_choice_range_min <= 0:
        raise ValueError(
            "target_choice_range_min must be positive for gene-table strategies"
        )
    if not isinstance(config.disable_observed_length_mutation, bool):
        raise ValueError("disable_observed_length_mutation must be bool")
    if not isinstance(config.disable_fallback_index_mutation, bool):
        raise ValueError("disable_fallback_index_mutation must be bool")
    for field_name in [
        "mutation_rate",
        "crossover_rate",
        "gene_join_rate",
        "gene_mutation_probability",
        "structure_mutation_probability",
    ]:
        value = getattr(config, field_name)
        if not 0 <= value <= 1:
            raise ValueError(f"{field_name} must be between 0 and 1")

    validate_initial_population(
        "A",
        config.initial_population_A,
        config.population_size_A,
        config,
    )
    validate_initial_population(
        "B",
        config.initial_population_B,
        config.population_size_B,
        config,
    )


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    config = parse_args(argv)
    best_a, best_b, best_pair = run_genetic_algorithm(config, include_best_pair=True)
    print(
        "Best pair:"
        f" A={best_pair.strategy_A_id}"
        f" B={best_pair.strategy_B_id}"
        f" score={best_pair.score:.2%}"
    )
    print(
        "Best A strategy:"
        f" id={best_a.id} score={best_a.survival_ratio:.2%}"
        f" partner={best_a.best_partner_id}"
    )
    print(
        "Best B strategy:"
        f" id={best_b.id} score={best_b.survival_ratio:.2%}"
        f" partner={best_b.best_partner_id}"
    )
    print(f"Artifacts written to: {Path(config.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
