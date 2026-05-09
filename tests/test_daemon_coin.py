import csv
import json
import random
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from daemon_coin import (
    GAConfig,
    Strategy,
    assign_survival,
    both_one_success,
    choose_target_index,
    compatible,
    crossover_strategy,
    evaluate_pair,
    evaluate_populations,
    find_best_pair,
    find_best_pairs,
    first_pattern_index,
    gene_join_strategy,
    initial_population,
    load_config_file,
    mutate_strategy,
    next_generation,
    next_generation_size,
    parse_args,
    pattern_occurrence_index,
    prefix_to_index,
    random_strategy,
    remove_parents_after_child_production,
    resize_genes,
    run_genetic_algorithm,
    same_bit_success,
    success_condition_from_name,
    validate_config,
    weighted_partner,
    weighted_partners,
)
from compare_strategy_pairs import (
    annotate_checked_ranks,
    annotate_certainty_groups,
    comparison_rows,
    current_best_is_separated,
    certainty_ranking_is_complete,
    evaluate_pairs_on_shared_trials,
    interval_label,
    latest_generation,
    load_population,
    load_score_matrix,
    render_html_report,
    sequential_comparison_rows,
)


def make_strategy(
    strategy_id,
    player_type,
    observed_length=2,
    target_choice_range=3,
    genes=None,
    survival_ratio=0.0,
):
    if genes is None:
        genes = [0] * (2**observed_length)

    return Strategy(
        id=strategy_id,
        player_type=player_type,
        observed_length=observed_length,
        target_choice_range=target_choice_range,
        genes=genes,
        generation_created=0,
        parent_ids=[],
        creation_method="test",
        survival_ratio=survival_ratio,
    )


class DaemonCoinGATests(unittest.TestCase):
    def test_prefix_to_index_interprets_bits_as_binary_number(self):
        self.assertEqual(prefix_to_index([0, 0, 0]), 0)
        self.assertEqual(prefix_to_index([1, 0, 1]), 5)
        self.assertEqual(prefix_to_index([1, 1, 1]), 7)

    def test_random_strategy_respects_configured_dimensions(self):
        config = GAConfig(
            observed_length_min=3,
            observed_length_max=3,
            target_choice_range_min=4,
            target_choice_range_max=4,
            first_pattern_observed_length_min=3,
            first_pattern_observed_length_max=3,
            first_pattern_fallback_index_min=2,
            first_pattern_fallback_index_max=2,
            first_pattern_occurrence_min=2,
            first_pattern_occurrence_max=2,
        )

        strategy = random_strategy("A", 1, 2, config, random.Random(1))

        self.assertEqual(strategy.id, "A-g0001-000002")
        self.assertEqual(strategy.observed_length, 3)
        self.assertEqual(strategy.target_choice_range, 0)
        self.assertEqual(strategy.strategy_type, "first-pattern")
        self.assertEqual(strategy.genes, [])
        self.assertTrue(1 <= len(strategy.pattern) <= 3)
        self.assertTrue(all(bit in {0, 1} for bit in strategy.pattern))
        self.assertEqual(strategy.fallback_index, 2)
        self.assertEqual(strategy.occurrence_number, 2)

    def test_random_strategy_can_still_generate_gene_table_strategy(self):
        config = GAConfig(
            observed_length_min=3,
            observed_length_max=3,
            target_choice_range_min=4,
            target_choice_range_max=4,
            default_strategy_type="gene-table",
        )

        strategy = random_strategy("A", 1, 2, config, random.Random(1))

        self.assertEqual(strategy.strategy_type, "gene-table")
        self.assertEqual(len(strategy.genes), 8)
        self.assertTrue(all(0 <= gene < 4 for gene in strategy.genes))
        self.assertEqual(strategy.pattern, [])

    def test_success_condition_lookup(self):
        self.assertIs(success_condition_from_name("same-bit"), same_bit_success)
        self.assertIs(success_condition_from_name("both-one"), both_one_success)

        with self.assertRaises(ValueError):
            success_condition_from_name("unknown")

    def test_evaluate_pair_uses_supplied_success_condition(self):
        strategy_a = make_strategy("A1", "A")
        strategy_b = make_strategy("B1", "B")

        always_true_score = evaluate_pair(
            strategy_a,
            strategy_b,
            trials=20,
            sequence_length=20,
            success_condition=lambda _a_value, _b_value: True,
            rng=random.Random(1),
        )
        always_false_score = evaluate_pair(
            strategy_a,
            strategy_b,
            trials=20,
            sequence_length=20,
            success_condition=lambda _a_value, _b_value: False,
            rng=random.Random(1),
        )

        self.assertEqual(always_true_score, 1.0)
        self.assertEqual(always_false_score, 0.0)

    def test_first_pattern_strategy_chooses_first_match_index(self):
        self.assertEqual(first_pattern_index([1, 0, 1, 1], [1, 1]), 2)
        self.assertIsNone(first_pattern_index([1, 0, 1, 0], [1, 1]))
        self.assertEqual(pattern_occurrence_index([1, 1, 1], [1, 1], 2), 1)
        self.assertIsNone(pattern_occurrence_index([1, 1, 1], [1, 1], 3))

        strategy_a = make_strategy(
            "A-pattern",
            "A",
            observed_length=3,
            target_choice_range=3,
            genes=[],
        )
        strategy_a.strategy_type = "first-pattern"
        strategy_a.pattern = [1, 1]

        self.assertEqual(choose_target_index(strategy_a, [0, 0, 0]), 1)
        strategy_a.index_offset = 1
        self.assertEqual(choose_target_index(strategy_a, [1, 0, 1, 1]), 3)
        strategy_a.occurrence_number = 2
        self.assertEqual(choose_target_index(strategy_a, [1, 1, 0, 1, 1]), 4)
        self.assertEqual(choose_target_index(strategy_a, [1, 0, 1, 1]), 1)
        strategy_a.index_offset = -5
        strategy_a.occurrence_number = 1
        self.assertEqual(choose_target_index(strategy_a, [1, 0, 1, 1]), 0)

        strategy_b = make_strategy(
            "B-pattern",
            "B",
            observed_length=3,
            target_choice_range=3,
            genes=[],
        )
        strategy_b.strategy_type = "first-pattern"
        strategy_b.pattern = [1, 1]

        score = evaluate_pair(
            strategy_a,
            strategy_b,
            trials=20,
            sequence_length=20,
            success_condition=lambda _a_value, _b_value: True,
            rng=random.Random(1),
        )

        self.assertEqual(score, 1.0)

    def test_block_lookup_strategy_uses_first_non_skipped_block(self):
        strategy = make_strategy("A-block", "A", genes=[])
        strategy.strategy_type = "block-lookup"
        strategy.observed_length = 0
        strategy.target_choice_range = 0
        strategy.block_size = 3
        strategy.skip_blocks = [0, 7]
        strategy.lookup_offsets = [0, 0, 1, 0, 2, 2, 1, 0]
        strategy.fallback_policy = "first-index"

        self.assertEqual(choose_target_index(strategy, [0, 0, 0, 1, 0, 1]), 5)
        self.assertEqual(choose_target_index(strategy, [1, 1, 1, 0, 1, 0]), 4)
        self.assertEqual(choose_target_index(strategy, [0, 0, 1]), 0)

    def test_block_lookup_strategy_fallback_policies(self):
        strategy = make_strategy("A-block", "A", genes=[])
        strategy.strategy_type = "block-lookup"
        strategy.observed_length = 0
        strategy.target_choice_range = 0
        strategy.block_size = 3
        strategy.skip_blocks = [0]
        strategy.lookup_offsets = [0, 0, 1, 0, 2, 2, 1, 0]

        strategy.fallback_policy = "last-index"
        self.assertEqual(choose_target_index(strategy, [0, 0, 0]), 2)

        strategy.fallback_policy = "none"
        self.assertIsNone(choose_target_index(strategy, [0, 0, 0]))

        strategy.fallback_policy = "random-block"
        self.assertIn(
            choose_target_index(strategy, [0, 0, 0], random.Random(1)),
            {0, 1, 2},
        )

    def test_random_block_lookup_strategy_respects_config(self):
        config = GAConfig(
            default_strategy_type="block-lookup",
            block_lookup_block_size_min=3,
            block_lookup_block_size_max=3,
            block_lookup_fallback_policy="random-index",
        )

        strategy = random_strategy("A", 1, 2, config, random.Random(1))

        self.assertEqual(strategy.strategy_type, "block-lookup")
        self.assertEqual(strategy.observed_length, 0)
        self.assertEqual(strategy.target_choice_range, 0)
        self.assertEqual(strategy.block_size, 3)
        self.assertEqual(len(strategy.lookup_offsets), 8)
        self.assertTrue(all(0 <= offset < 3 for offset in strategy.lookup_offsets))
        self.assertTrue(all(0 <= block < 8 for block in strategy.skip_blocks))
        self.assertEqual(strategy.fallback_policy, "random-index")

    def test_block_lookup_seed_strategy_is_supported(self):
        config = GAConfig(
            population_size_A=2,
            population_size_B=2,
            elite_count=0,
            default_strategy_type="block-lookup",
            initial_population_A=[
                {
                    "id": "A-canonical-triple",
                    "strategy_type": "block-lookup",
                    "observed_length": 0,
                    "block_size": 3,
                    "skip_blocks": [0, 7],
                    "lookup_offsets": [0, 0, 1, 0, 2, 2, 1, 0],
                    "fallback_policy": "random-index",
                }
            ],
        )
        validate_config(config)

        population = initial_population(
            "A",
            2,
            config.initial_population_A,
            config,
            random.Random(1),
        )

        self.assertEqual(population[0].strategy_type, "block-lookup")
        self.assertEqual(population[0].skip_blocks, [0, 7])
        self.assertEqual(population[0].lookup_offsets, [0, 0, 1, 0, 2, 2, 1, 0])

    def test_fsm_strategy_runs_transitions_and_chooses_state_output(self):
        strategy = make_strategy("A-fsm", "A", genes=[])
        strategy.strategy_type = "fsm"
        strategy.observed_length = 0
        strategy.target_choice_range = 0
        strategy.fsm_state_count = 2
        strategy.fsm_start_state = 0
        strategy.fsm_outputs = [0, 1]
        strategy.fsm_transitions = [[0, 1], [1, 0]]

        self.assertEqual(choose_target_index(strategy, [1]), 1)
        self.assertEqual(choose_target_index(strategy, [1, 1]), 0)
        self.assertEqual(choose_target_index(strategy, [1, 0, 1]), 0)

    def test_fsm_accepting_state_chooses_acceptance_index_with_offset(self):
        strategy = make_strategy("A-fsm-accepting", "A", genes=[])
        strategy.strategy_type = "fsm"
        strategy.observed_length = 0
        strategy.target_choice_range = 0
        strategy.fsm_state_count = 3
        strategy.fsm_start_state = 0
        strategy.fsm_outputs = [0, 0, 2]
        strategy.fsm_transitions = [[1, 0], [1, 2], [2, 2]]
        strategy.fsm_accepting_states = [2]
        strategy.fsm_index_offset = 1
        strategy.fsm_fallback_index = 0

        self.assertEqual(choose_target_index(strategy, [0, 1, 1, 0, 0]), 4)
        self.assertEqual(choose_target_index(strategy, [1, 1, 1]), 0)

    def test_random_fsm_strategy_respects_config(self):
        config = GAConfig(
            default_strategy_type="fsm",
            fsm_state_count_min=3,
            fsm_state_count_max=3,
            fsm_output_index_min=1,
            fsm_output_index_max=2,
        )

        strategy = random_strategy("A", 1, 2, config, random.Random(1))

        self.assertEqual(strategy.strategy_type, "fsm")
        self.assertEqual(strategy.fsm_state_count, 3)
        self.assertTrue(0 <= strategy.fsm_start_state < 3)
        self.assertEqual(len(strategy.fsm_outputs), 3)
        self.assertTrue(all(1 <= output <= 2 for output in strategy.fsm_outputs))
        self.assertEqual(len(strategy.fsm_transitions), 3)
        self.assertTrue(
            all(0 <= state < 3 for row in strategy.fsm_transitions for state in row)
        )
        self.assertTrue(
            all(0 <= state < 3 for state in strategy.fsm_accepting_states)
        )

    def test_fsm_seed_strategy_is_supported(self):
        config = GAConfig(
            population_size_A=2,
            population_size_B=2,
            elite_count=0,
            default_strategy_type="fsm",
            initial_population_A=[
                {
                    "id": "A-fsm-parity",
                    "strategy_type": "fsm",
                    "observed_length": 0,
                    "fsm_state_count": 2,
                    "fsm_start_state": 0,
                    "fsm_outputs": [0, 1],
                    "fsm_transitions": [[0, 1], [1, 0]],
                    "fsm_accepting_states": [1],
                    "fsm_index_offset": 0,
                    "fsm_fallback_index": 0,
                }
            ],
        )
        validate_config(config)

        population = initial_population(
            "A",
            2,
            config.initial_population_A,
            config,
            random.Random(1),
        )

        self.assertEqual(population[0].strategy_type, "fsm")
        self.assertEqual(population[0].fsm_outputs, [0, 1])
        self.assertEqual(population[0].fsm_transitions, [[0, 1], [1, 0]])
        self.assertEqual(population[0].fsm_accepting_states, [1])

    def test_assign_survival_uses_best_partner_per_strategy(self):
        population_a = [
            make_strategy("A1", "A"),
            make_strategy("A2", "A"),
        ]
        population_b = [
            make_strategy("B1", "B"),
            make_strategy("B2", "B"),
        ]
        score_matrix = [
            [0.25, 0.75],
            [0.60, 0.40],
        ]

        assign_survival(population_a, population_b, score_matrix)

        self.assertEqual(population_a[0].survival_ratio, 0.75)
        self.assertEqual(population_a[0].best_partner_id, "B2")
        self.assertEqual(population_a[1].survival_ratio, 0.60)
        self.assertEqual(population_a[1].best_partner_id, "B1")
        self.assertEqual(population_b[0].survival_ratio, 0.60)
        self.assertEqual(population_b[0].best_partner_id, "A2")
        self.assertEqual(population_b[1].survival_ratio, 0.75)
        self.assertEqual(population_b[1].best_partner_id, "A1")

    def test_find_best_pair_returns_highest_scoring_pair(self):
        population_a = [
            make_strategy("A1", "A"),
            make_strategy("A2", "A"),
        ]
        population_b = [
            make_strategy("B1", "B"),
            make_strategy("B2", "B"),
        ]

        best_pair = find_best_pair(
            population_a,
            population_b,
            [
                [0.30, 0.50],
                [0.90, 0.40],
            ],
        )

        self.assertEqual(best_pair.strategy_A_id, "A2")
        self.assertEqual(best_pair.strategy_B_id, "B1")
        self.assertEqual(best_pair.score, 0.90)

    def test_find_best_pairs_returns_top_pairs_in_score_order(self):
        population_a = [
            make_strategy("A1", "A"),
            make_strategy("A2", "A", genes=[1, 1, 1, 1]),
        ]
        population_b = [
            make_strategy("B1", "B"),
            make_strategy("B2", "B", genes=[1, 1, 1, 1]),
        ]

        best_pairs = find_best_pairs(
            population_a,
            population_b,
            [
                [0.30, 0.50],
                [0.90, 0.40],
            ],
            3,
        )

        self.assertEqual(
            [(pair.strategy_A_id, pair.strategy_B_id, pair.score) for pair in best_pairs],
            [
                ("A2", "B1", 0.90),
                ("A1", "B2", 0.50),
                ("A2", "B2", 0.40),
            ],
        )

    def test_find_best_pairs_groups_equivalent_strategy_pairs(self):
        population_a = [
            make_strategy("A1", "A", genes=[0, 0, 0, 0]),
            make_strategy("A1-copy", "A", genes=[0, 0, 0, 0]),
            make_strategy("A2", "A", genes=[1, 1, 1, 1]),
        ]
        population_b = [
            make_strategy("B1", "B", genes=[0, 0, 0, 0]),
            make_strategy("B1-copy", "B", genes=[0, 0, 0, 0]),
            make_strategy("B2", "B", genes=[1, 1, 1, 1]),
        ]

        best_pairs = find_best_pairs(
            population_a,
            population_b,
            [
                [0.91, 0.90, 0.70],
                [0.89, 0.88, 0.69],
                [0.80, 0.79, 0.60],
            ],
            3,
        )

        self.assertEqual(len(best_pairs), 3)
        self.assertEqual(
            [(pair.strategy_A_id, pair.strategy_B_id, pair.score) for pair in best_pairs],
            [
                ("A1", "B1", 0.91),
                ("A2", "B1", 0.80),
                ("A1", "B2", 0.70),
            ],
        )

    def test_resize_genes_preserves_valid_gene_count_and_range(self):
        rng = random.Random(1)

        shrunk = resize_genes([0, 1, 2, 3], 2, 1, 3, rng)
        grown = resize_genes([0, 1], 1, 3, 2, rng)
        remapped = resize_genes([0, 5, 8, 2], 2, 2, 3, rng)

        self.assertEqual(shrunk, [0, 1])
        self.assertEqual(len(grown), 8)
        self.assertEqual(grown[:2], [0, 1])
        self.assertTrue(all(0 <= gene < 2 for gene in grown))
        self.assertEqual(remapped, [0, 2, 2, 2])

    def test_mutate_strategy_can_change_structure_and_keeps_genes_valid(self):
        config = GAConfig(
            observed_length_min=1,
            observed_length_max=3,
            target_choice_range_min=2,
            target_choice_range_max=4,
            structure_mutation_probability=1.0,
            gene_mutation_probability=1.0,
        )
        parent = make_strategy(
            "A-parent",
            "A",
            observed_length=2,
            target_choice_range=3,
            genes=[0, 1, 2, 0],
        )

        child = mutate_strategy(parent, 2, 7, config, random.Random(4))

        self.assertEqual(child.id, "A-g0002-000007")
        self.assertEqual(child.creation_method, "mutation")
        self.assertEqual(child.parent_ids, ["A-parent"])
        self.assertEqual(len(child.genes), 2**child.observed_length)
        self.assertTrue(all(0 <= gene < child.target_choice_range for gene in child.genes))

    def test_mutate_first_pattern_can_evolve_observed_length_and_fallback(self):
        config = GAConfig(
            observed_length_min=2,
            observed_length_max=4,
            sequence_length=5,
            first_pattern_index_offset_min=-2,
            first_pattern_index_offset_max=2,
            disable_observed_length_mutation=False,
            disable_fallback_index_mutation=False,
            structure_mutation_probability=1.0,
            gene_mutation_probability=0.0,
        )
        parent = make_strategy(
            "A-pattern-parent",
            "A",
            observed_length=0,
            target_choice_range=0,
            genes=[],
        )
        parent.strategy_type = "first-pattern"
        parent.pattern = [1, 0]
        parent.fallback_index = 1

        child = mutate_strategy(parent, 2, 8, config, random.Random(1))

        self.assertEqual(child.strategy_type, "first-pattern")
        self.assertIn(child.observed_length, [0, 2, 3, 4])
        self.assertTrue(0 <= child.fallback_index < config.sequence_length)
        self.assertTrue(
            config.first_pattern_index_offset_min
            <= child.index_offset
            <= config.first_pattern_index_offset_max
        )
        if child.observed_length != 0:
            self.assertLessEqual(len(child.pattern), child.observed_length)

    def test_mutate_block_lookup_keeps_lookup_valid(self):
        config = GAConfig(
            default_strategy_type="block-lookup",
            block_lookup_block_size_min=3,
            block_lookup_block_size_max=4,
            structure_mutation_probability=1.0,
            gene_mutation_probability=1.0,
        )
        parent = make_strategy("A-block-parent", "A", genes=[])
        parent.strategy_type = "block-lookup"
        parent.observed_length = 0
        parent.target_choice_range = 0
        parent.block_size = 3
        parent.skip_blocks = [0, 7]
        parent.lookup_offsets = [0, 0, 1, 0, 2, 2, 1, 0]
        parent.fallback_policy = "random-index"

        child = mutate_strategy(parent, 2, 9, config, random.Random(2))

        self.assertEqual(child.strategy_type, "block-lookup")
        self.assertEqual(child.observed_length, 0)
        self.assertEqual(child.target_choice_range, 0)
        self.assertIn(child.block_size, [3, 4])
        self.assertEqual(len(child.lookup_offsets), 2**child.block_size)
        self.assertTrue(all(0 <= offset < child.block_size for offset in child.lookup_offsets))
        self.assertTrue(all(0 <= block < 2**child.block_size for block in child.skip_blocks))
        self.assertLess(len(child.skip_blocks), 2**child.block_size)
        self.assertIn(child.fallback_policy, {
            "random-index",
            "first-index",
            "last-index",
            "random-block",
            "none",
        })

    def test_mutate_fsm_keeps_machine_valid(self):
        config = GAConfig(
            default_strategy_type="fsm",
            fsm_state_count_min=2,
            fsm_state_count_max=4,
            fsm_output_index_min=0,
            fsm_output_index_max=3,
            structure_mutation_probability=1.0,
            gene_mutation_probability=1.0,
        )
        parent = make_strategy("A-fsm-parent", "A", genes=[])
        parent.strategy_type = "fsm"
        parent.observed_length = 0
        parent.target_choice_range = 0
        parent.fsm_state_count = 2
        parent.fsm_start_state = 0
        parent.fsm_outputs = [0, 1]
        parent.fsm_transitions = [[0, 1], [1, 0]]

        child = mutate_strategy(parent, 2, 10, config, random.Random(2))

        self.assertEqual(child.strategy_type, "fsm")
        self.assertIn(child.fsm_state_count, [2, 3])
        self.assertTrue(0 <= child.fsm_start_state < child.fsm_state_count)
        self.assertEqual(len(child.fsm_outputs), child.fsm_state_count)
        self.assertEqual(len(child.fsm_transitions), child.fsm_state_count)
        self.assertTrue(all(0 <= output <= 3 for output in child.fsm_outputs))
        self.assertTrue(
            all(
                0 <= state < child.fsm_state_count
                for row in child.fsm_transitions
                for state in row
            )
        )
        self.assertTrue(
            all(0 <= state < child.fsm_state_count for state in child.fsm_accepting_states)
        )

    def test_crossover_and_gene_join_require_compatible_structures(self):
        parent_1 = make_strategy("A1", "A", genes=[0, 0, 1, 1])
        parent_2 = make_strategy("A2", "A", genes=[1, 1, 2, 2])
        incompatible_parent = make_strategy(
            "B1",
            "B",
            observed_length=2,
            target_choice_range=3,
            genes=[0, 1, 2, 0],
        )

        self.assertTrue(compatible(parent_1, parent_2))
        self.assertFalse(compatible(parent_1, incompatible_parent))

        crossed = crossover_strategy(parent_1, parent_2, 3, 1, random.Random(2))
        joined = gene_join_strategy([parent_1, parent_2], 3, 2, random.Random(3))

        self.assertEqual(crossed.creation_method, "crossover")
        self.assertEqual(crossed.parent_ids, ["A1", "A2"])
        self.assertEqual(len(crossed.genes), 4)
        self.assertEqual(joined.creation_method, "gene-join")
        self.assertEqual(joined.parent_ids, ["A1", "A2"])
        self.assertEqual(len(joined.genes), 4)

    def test_block_lookup_crossover_and_gene_join_combine_tables(self):
        parent_1 = make_strategy("A-block-1", "A", genes=[])
        parent_1.strategy_type = "block-lookup"
        parent_1.observed_length = 0
        parent_1.target_choice_range = 0
        parent_1.block_size = 3
        parent_1.skip_blocks = [0, 7]
        parent_1.lookup_offsets = [0, 0, 1, 0, 2, 2, 1, 0]
        parent_1.fallback_policy = "first-index"

        parent_2 = make_strategy("A-block-2", "A", genes=[])
        parent_2.strategy_type = "block-lookup"
        parent_2.observed_length = 0
        parent_2.target_choice_range = 0
        parent_2.block_size = 3
        parent_2.skip_blocks = [1, 6]
        parent_2.lookup_offsets = [2, 2, 0, 2, 0, 0, 2, 2]
        parent_2.fallback_policy = "last-index"

        self.assertTrue(compatible(parent_1, parent_2))
        crossed = crossover_strategy(parent_1, parent_2, 3, 3, random.Random(1))
        joined = gene_join_strategy([parent_1, parent_2], 3, 4, random.Random(2))

        self.assertEqual(crossed.strategy_type, "block-lookup")
        self.assertEqual(joined.strategy_type, "block-lookup")
        self.assertEqual(len(crossed.lookup_offsets), 8)
        self.assertEqual(len(joined.lookup_offsets), 8)
        self.assertTrue(all(0 <= offset < 3 for offset in crossed.lookup_offsets))
        self.assertTrue(all(0 <= offset < 3 for offset in joined.lookup_offsets))

    def test_fsm_crossover_and_gene_join_combine_machines(self):
        parent_1 = make_strategy("A-fsm-1", "A", genes=[])
        parent_1.strategy_type = "fsm"
        parent_1.observed_length = 0
        parent_1.target_choice_range = 0
        parent_1.fsm_state_count = 2
        parent_1.fsm_start_state = 0
        parent_1.fsm_outputs = [0, 1]
        parent_1.fsm_transitions = [[0, 1], [1, 0]]
        parent_1.fsm_accepting_states = [1]

        parent_2 = make_strategy("A-fsm-2", "A", genes=[])
        parent_2.strategy_type = "fsm"
        parent_2.observed_length = 0
        parent_2.target_choice_range = 0
        parent_2.fsm_state_count = 2
        parent_2.fsm_start_state = 1
        parent_2.fsm_outputs = [2, 3]
        parent_2.fsm_transitions = [[1, 0], [0, 1]]
        parent_2.fsm_accepting_states = [0]

        self.assertTrue(compatible(parent_1, parent_2))
        crossed = crossover_strategy(parent_1, parent_2, 3, 5, random.Random(1))
        joined = gene_join_strategy([parent_1, parent_2], 3, 6, random.Random(2))

        self.assertEqual(crossed.strategy_type, "fsm")
        self.assertEqual(joined.strategy_type, "fsm")
        self.assertEqual(crossed.fsm_state_count, 2)
        self.assertEqual(joined.fsm_state_count, 2)
        self.assertEqual(len(crossed.fsm_outputs), 2)
        self.assertEqual(len(joined.fsm_outputs), 2)
        self.assertEqual(len(crossed.fsm_transitions), 2)
        self.assertEqual(len(joined.fsm_transitions), 2)
        self.assertTrue(all(0 <= state < 2 for state in crossed.fsm_accepting_states))
        self.assertTrue(all(0 <= state < 2 for state in joined.fsm_accepting_states))

    def test_reproduction_partners_are_weighted_by_survival_ratio(self):
        class RecordingRandom:
            def __init__(self):
                self.recorded_weights = []

            def choices(self, population, weights, k):
                self.recorded_weights.append(list(weights))
                best_index = max(range(len(weights)), key=weights.__getitem__)
                return [population[best_index]]

        weak = make_strategy("weak", "A", survival_ratio=0.05)
        strong = make_strategy("strong", "A", survival_ratio=0.90)
        middle = make_strategy("middle", "A", survival_ratio=0.40)
        rng = RecordingRandom()

        selected_partner = weighted_partner([weak, strong, middle], rng)
        selected_partners = weighted_partners([weak, strong, middle], 2, rng)

        self.assertEqual(selected_partner.id, "strong")
        self.assertEqual([partner.id for partner in selected_partners], ["strong", "middle"])
        self.assertEqual(rng.recorded_weights[0], [0.05, 0.90, 0.40])

    def test_parent_dies_after_fourth_child(self):
        parent = make_strategy("A-parent", "A")
        parent.children_produced = 3
        other = make_strategy("A-other", "A")

        remaining = remove_parents_after_child_production(
            [parent, other],
            ["A-parent"],
            4,
        )

        self.assertEqual(parent.children_produced, 4)
        self.assertEqual([strategy.id for strategy in remaining], ["A-other"])

    def test_all_child_parents_increment_for_crossover_or_gene_join(self):
        parent_1 = make_strategy("A1", "A")
        parent_2 = make_strategy("A2", "A")

        remaining = remove_parents_after_child_production(
            [parent_1, parent_2],
            ["A1", "A2"],
            4,
        )

        self.assertEqual(parent_1.children_produced, 1)
        self.assertEqual(parent_2.children_produced, 1)
        self.assertEqual([strategy.id for strategy in remaining], ["A1", "A2"])

    def test_next_generation_removes_exhausted_parents_and_keeps_target_size(self):
        config = GAConfig(
            population_size_A=4,
            population_size_B=4,
            observed_length_min=2,
            observed_length_max=2,
            target_choice_range_min=2,
            target_choice_range_max=2,
            elite_count=1,
            mutation_rate=1.0,
            crossover_rate=0.0,
            gene_join_rate=0.0,
            max_children_per_strategy=1,
            default_strategy_type="gene-table",
        )
        parent = make_strategy("A-parent", "A", survival_ratio=1.0)

        next_population = next_generation(
            [parent],
            "A",
            4,
            1,
            config,
            random.Random(1),
        )

        self.assertEqual(len(next_population), 4)
        self.assertNotIn("A-parent", [strategy.id for strategy in next_population])
        self.assertEqual(parent.children_produced, 1)

    def test_validate_config_rejects_invalid_bounds_and_rates(self):
        with self.assertRaises(ValueError):
            validate_config(GAConfig(observed_length_min=4, observed_length_max=3))

        with self.assertRaises(ValueError):
            validate_config(GAConfig(mutation_rate=1.5))

        with self.assertRaises(ValueError):
            validate_config(GAConfig(default_strategy_type="unknown"))

        with self.assertRaises(ValueError):
            validate_config(GAConfig(population_size_A=2, population_size_B=2, elite_count=2))

        with self.assertRaises(ValueError):
            validate_config(GAConfig(max_children_per_strategy=0))

        with self.assertRaises(ValueError):
            validate_config(
                GAConfig(
                    first_pattern_occurrence_min=3,
                    first_pattern_occurrence_max=2,
                )
            )

    def test_load_config_file_reads_toml_ga_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[ga]",
                        "population_size_A = 5",
                        "population_size_B = 6",
                        "observed_length_min = 1",
                        "observed_length_max = 3",
                        "target_choice_range_min = 2",
                        "target_choice_range_max = 3",
                        "first_pattern_observed_length_min = 2",
                        "first_pattern_observed_length_max = 3",
                        "first_pattern_fallback_index_min = 2",
                        "first_pattern_fallback_index_max = 4",
                        "first_pattern_index_offset_min = -3",
                        "first_pattern_index_offset_max = 3",
                        "first_pattern_occurrence_min = 1",
                        "first_pattern_occurrence_max = 2",
                        "disable_observed_length_mutation = false",
                        "disable_fallback_index_mutation = false",
                        "trials_per_pair = 7",
                        "generations = 8",
                        "elite_count = 1",
                        "max_children_per_strategy = 4",
                        'success_condition = "both-one"',
                        'output_dir = "custom_output"',
                        "seed = 123",
                        "",
                        "[[ga.initial_population_A]]",
                        'id = "A-seed"',
                        "population = 2",
                        "observed_length = 1",
                        "target_choice_range = 2",
                        "genes = [0, 1]",
                        "",
                        "[[ga.initial_population_B]]",
                        'id = "B-pattern"',
                        'strategy_type = "first-pattern"',
                        "observed_length = 3",
                        "fallback_index = 2",
                        "index_offset = -1",
                        "occurrence_number = 2",
                        "pattern = [1, 1]",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config_file(config_path)

            self.assertEqual(config.population_size_A, 5)
            self.assertEqual(config.population_size_B, 6)
            self.assertEqual(config.trials_per_pair, 7)
            self.assertEqual(config.generations, 8)
            self.assertEqual(config.max_children_per_strategy, 4)
            self.assertEqual(config.success_condition, "both-one")
            self.assertEqual(config.output_dir, "custom_output")
            self.assertEqual(config.seed, 123)
            self.assertEqual(config.first_pattern_observed_length_min, 2)
            self.assertEqual(config.first_pattern_observed_length_max, 3)
            self.assertEqual(config.first_pattern_fallback_index_min, 2)
            self.assertEqual(config.first_pattern_fallback_index_max, 4)
            self.assertEqual(config.first_pattern_index_offset_min, -3)
            self.assertEqual(config.first_pattern_index_offset_max, 3)
            self.assertEqual(config.first_pattern_occurrence_min, 1)
            self.assertEqual(config.first_pattern_occurrence_max, 2)
            self.assertFalse(config.disable_observed_length_mutation)
            self.assertFalse(config.disable_fallback_index_mutation)
            self.assertEqual(config.initial_population_A[0]["id"], "A-seed")
            self.assertEqual(config.initial_population_A[0]["population"], 2)
            self.assertEqual(config.initial_population_A[0]["genes"], [0, 1])
            self.assertEqual(
                config.initial_population_B[0]["strategy_type"],
                "first-pattern",
            )
            self.assertEqual(config.initial_population_B[0]["pattern"], [1, 1])
            self.assertEqual(config.initial_population_B[0]["fallback_index"], 2)
            self.assertEqual(config.initial_population_B[0]["index_offset"], -1)
            self.assertEqual(config.initial_population_B[0]["occurrence_number"], 2)

    def test_initial_population_uses_only_seeded_strategies_when_present(self):
        config = GAConfig(
            population_size_A=3,
            observed_length_min=1,
            observed_length_max=1,
            target_choice_range_min=2,
            target_choice_range_max=2,
            initial_population_A=[
                {
                    "id": "A-fixed",
                    "observed_length": 1,
                    "target_choice_range": 2,
                    "genes": [0, 1],
                }
            ],
        )

        population = initial_population(
            "A",
            config.population_size_A,
            config.initial_population_A,
            config,
            random.Random(1),
        )

        self.assertEqual(len(population), 1)
        self.assertEqual(population[0].id, "A-fixed")
        self.assertEqual(population[0].creation_method, "initial-seed")
        self.assertEqual(population[0].genes, [0, 1])

    def test_initial_population_random_fills_to_cap_when_no_seeds_exist(self):
        config = GAConfig(
            population_size_A=3,
            observed_length_min=1,
            observed_length_max=1,
            target_choice_range_min=2,
            target_choice_range_max=2,
            first_pattern_observed_length_min=1,
            first_pattern_observed_length_max=1,
        )

        population = initial_population(
            "A",
            config.population_size_A,
            [],
            config,
            random.Random(1),
        )

        self.assertEqual(len(population), 3)
        self.assertTrue(
            all(strategy.creation_method == "random" for strategy in population)
        )

    def test_initial_population_expands_seed_population_count(self):
        config = GAConfig(
            population_size_A=3,
            observed_length_min=1,
            observed_length_max=1,
            target_choice_range_min=2,
            target_choice_range_max=2,
            initial_population_A=[
                {
                    "id": "A-fixed",
                    "population": 3,
                    "observed_length": 1,
                    "target_choice_range": 2,
                    "genes": [0, 1],
                }
            ],
        )

        population = initial_population(
            "A",
            config.population_size_A,
            config.initial_population_A,
            config,
            random.Random(1),
        )

        self.assertEqual([strategy.id for strategy in population], [
            "A-fixed-copy-00",
            "A-fixed-copy-01",
            "A-fixed-copy-02",
        ])
        self.assertTrue(
            all(strategy.creation_method == "initial-seed" for strategy in population)
        )

    def test_disabled_mutation_flags_freeze_observed_length_and_fallback(self):
        config = GAConfig(
            observed_length_min=2,
            observed_length_max=4,
            sequence_length=5,
            disable_observed_length_mutation=True,
            disable_fallback_index_mutation=True,
            structure_mutation_probability=1.0,
            gene_mutation_probability=0.0,
        )
        parent = make_strategy(
            "A-pattern-parent",
            "A",
            observed_length=0,
            target_choice_range=0,
            genes=[],
        )
        parent.strategy_type = "first-pattern"
        parent.pattern = [1, 0]
        parent.fallback_index = 1

        child = mutate_strategy(parent, 2, 8, config, random.Random(1))

        self.assertEqual(child.observed_length, 0)
        self.assertEqual(child.fallback_index, 1)

    def test_next_generation_size_preserves_current_size_under_cap(self):
        self.assertEqual(next_generation_size(12, 100), 12)
        self.assertEqual(next_generation_size(120, 100), 100)

    def test_first_pattern_seed_without_observed_length_gets_random_length(self):
        config = GAConfig(
            population_size_A=1,
            observed_length_min=3,
            observed_length_max=3,
            first_pattern_observed_length_min=100,
            first_pattern_observed_length_max=100,
            first_pattern_fallback_index_min=7,
            first_pattern_fallback_index_max=7,
            first_pattern_index_offset_min=3,
            first_pattern_index_offset_max=3,
            initial_population_A=[
                {
                    "id": "A-pattern",
                    "strategy_type": "first-pattern",
                    "pattern": [0, 1],
                }
            ],
        )

        population = initial_population(
            "A",
            config.population_size_A,
            config.initial_population_A,
            config,
            random.Random(1),
        )

        self.assertEqual(population[0].observed_length, 100)
        self.assertEqual(population[0].fallback_index, 7)
        self.assertEqual(population[0].index_offset, 3)
        self.assertEqual(population[0].occurrence_number, 1)

    def test_load_config_file_rejects_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text("[ga]\nnot_a_config_key = 1\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_config_file(config_path)

    def test_current_config_seed_totals_match_population_caps(self):
        config = load_config_file(Path("config.toml"))

        self.assertEqual(config.population_size_A, 244)
        self.assertEqual(config.population_size_B, 244)
        self.assertEqual(len(config.initial_population_A), 71)
        self.assertEqual(len(config.initial_population_B), 71)
        self.assertEqual(
            sum(seed.get("population", 1) for seed in config.initial_population_A),
            config.population_size_A,
        )
        self.assertEqual(
            sum(seed.get("population", 1) for seed in config.initial_population_B),
            config.population_size_B,
        )
        self.assertEqual(
            Counter(seed["strategy_type"] for seed in config.initial_population_A),
            Counter({"first-pattern": 60, "fsm": 7, "block-lookup": 4}),
        )
        self.assertEqual(
            Counter(seed["strategy_type"] for seed in config.initial_population_B),
            Counter({"first-pattern": 60, "fsm": 7, "block-lookup": 4}),
        )

    def test_validate_config_rejects_invalid_initial_population(self):
        with self.assertRaises(ValueError):
            validate_config(
                GAConfig(
                    population_size_A=1,
                    observed_length_min=2,
                    observed_length_max=2,
                    target_choice_range_min=2,
                    target_choice_range_max=2,
                    initial_population_A=[
                        {
                            "id": "bad",
                            "observed_length": 2,
                            "target_choice_range": 2,
                            "genes": [0, 1],
                        }
                    ],
                )
            )

    def test_parse_args_uses_toml_then_cli_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[ga]",
                        "population_size_A = 5",
                        "population_size_B = 5",
                        "observed_length_min = 1",
                        "observed_length_max = 2",
                        "target_choice_range_min = 2",
                        "target_choice_range_max = 2",
                        "trials_per_pair = 10",
                        "generations = 3",
                        "elite_count = 1",
                        'output_dir = "from_file"',
                    ]
                ),
                encoding="utf-8",
            )

            config = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--generations",
                    "4",
                    "--output-dir",
                    "from_cli",
                    "--first-pattern-observed-length-min",
                    "0",
                    "--first-pattern-observed-length-max",
                    "5",
                    "--first-pattern-fallback-index-min",
                    "1",
                    "--first-pattern-fallback-index-max",
                    "3",
                    "--first-pattern-index-offset-min",
                    "-2",
                    "--first-pattern-index-offset-max",
                    "2",
                    "--first-pattern-occurrence-min",
                    "1",
                    "--first-pattern-occurrence-max",
                    "2",
                    "--max-children-per-strategy",
                    "4",
                    "--no-disable-observed-length-mutation",
                    "--no-disable-fallback-index-mutation",
                ]
            )

            self.assertEqual(config.population_size_A, 5)
            self.assertEqual(config.population_size_B, 5)
            self.assertEqual(config.generations, 4)
            self.assertEqual(config.output_dir, "from_cli")
            self.assertEqual(config.first_pattern_observed_length_min, 0)
            self.assertEqual(config.first_pattern_observed_length_max, 5)
            self.assertEqual(config.first_pattern_fallback_index_min, 1)
            self.assertEqual(config.first_pattern_fallback_index_max, 3)
            self.assertEqual(config.first_pattern_index_offset_min, -2)
            self.assertEqual(config.first_pattern_index_offset_max, 2)
            self.assertEqual(config.first_pattern_occurrence_min, 1)
            self.assertEqual(config.first_pattern_occurrence_max, 2)
            self.assertEqual(config.max_children_per_strategy, 4)
            self.assertFalse(config.disable_observed_length_mutation)
            self.assertFalse(config.disable_fallback_index_mutation)

    def test_evaluate_populations_scores_duplicate_strategies_consistently(self):
        config = GAConfig(
            population_size_A=2,
            population_size_B=2,
            trials_per_pair=10,
            elite_count=1,
        )
        strategy_a_1 = make_strategy("A1", "A", genes=[])
        strategy_a_1.strategy_type = "first-pattern"
        strategy_a_1.pattern = [1]
        strategy_a_1.observed_length = 0
        strategy_a_1.target_choice_range = 0
        strategy_a_1.fallback_index = 0
        strategy_a_2 = Strategy(**{**strategy_a_1.__dict__, "id": "A2"})

        strategy_b_1 = make_strategy("B1", "B", genes=[])
        strategy_b_1.strategy_type = "first-pattern"
        strategy_b_1.pattern = [0]
        strategy_b_1.observed_length = 0
        strategy_b_1.target_choice_range = 0
        strategy_b_1.fallback_index = 0
        strategy_b_2 = Strategy(**{**strategy_b_1.__dict__, "id": "B2"})

        score_matrix = evaluate_populations(
            [strategy_a_1, strategy_a_2],
            [strategy_b_1, strategy_b_2],
            config,
            lambda _a_value, _b_value: True,
            random.Random(1),
        )

        self.assertEqual(score_matrix, [[1.0, 1.0], [1.0, 1.0]])

    def test_pair_comparison_uses_shared_trials_and_pairwise_differences(self):
        strategy_a_1 = make_strategy("A1", "A", genes=[])
        strategy_a_1.strategy_type = "first-pattern"
        strategy_a_1.pattern = [0]
        strategy_a_1.observed_length = 0
        strategy_a_1.target_choice_range = 0
        strategy_a_1.fallback_index = 0

        strategy_b_1 = make_strategy("B1", "B", genes=[])
        strategy_b_1.strategy_type = "first-pattern"
        strategy_b_1.pattern = [0]
        strategy_b_1.observed_length = 0
        strategy_b_1.target_choice_range = 0
        strategy_b_1.fallback_index = 0

        strategy_a_2 = Strategy(**{**strategy_a_1.__dict__, "id": "A2"})
        strategy_b_2 = Strategy(**{**strategy_b_1.__dict__, "id": "B2"})
        strategy_b_2.index_offset = 1

        wins, difference_sums, difference_square_sums = evaluate_pairs_on_shared_trials(
            [(strategy_a_1, strategy_b_1), (strategy_a_2, strategy_b_2)],
            trials=20,
            sequence_length=20,
            success_condition_name="same-bit",
            seed=1,
        )

        rows = comparison_rows(
            [
                find_best_pair([strategy_a_1], [strategy_b_1], [[0.0]]),
                find_best_pair([strategy_a_2], [strategy_b_2], [[0.0]]),
            ],
            [(strategy_a_1, strategy_b_1), (strategy_a_2, strategy_b_2)],
            wins,
            difference_sums,
            difference_square_sums,
            trials=20,
            sigma=5.0,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["sigma"], 5.0)
        self.assertEqual(interval_label(5.0), "5-sigma interval")
        self.assertEqual(rows[0]["trials"], 20)
        self.assertEqual(len(rows[0]["pairwise_comparisons"]), 2)
        self.assertEqual(
            rows[0]["pairwise_comparisons"][0]["score_difference"],
            0.0,
        )
        annotate_checked_ranks(rows)
        report = render_html_report(rows)
        self.assertIn("Top Strategy Pair Statistical Check", report)
        self.assertIn("Pairwise Checks", report)

    def test_current_best_is_separated_uses_pairwise_intervals(self):
        rows = [
            {
                "strategy_A": "A1",
                "strategy_B": "B1",
                "checked_score": 0.6,
                "pairwise_comparisons": [
                    {
                        "against_strategy_A": "A1",
                        "against_strategy_B": "B1",
                        "statistically_better": False,
                    },
                    {
                        "against_strategy_A": "A2",
                        "against_strategy_B": "B2",
                        "statistically_better": True,
                    },
                ],
            },
            {
                "strategy_A": "A2",
                "strategy_B": "B2",
                "checked_score": 0.5,
                "pairwise_comparisons": [],
            },
        ]

        self.assertTrue(current_best_is_separated(rows))
        rows[0]["pairwise_comparisons"][1]["statistically_better"] = False
        self.assertFalse(current_best_is_separated(rows))

    def test_certainty_groups_represent_partial_ranking(self):
        rows = [
            {
                "strategy_A": "A1",
                "strategy_B": "B1",
                "checked_score": 0.70,
                "pairwise_comparisons": [
                    {
                        "against_strategy_A": "A1",
                        "against_strategy_B": "B1",
                        "statistically_better": False,
                    },
                    {
                        "against_strategy_A": "A2",
                        "against_strategy_B": "B2",
                        "statistically_better": False,
                    },
                    {
                        "against_strategy_A": "A3",
                        "against_strategy_B": "B3",
                        "statistically_better": True,
                    },
                ],
            },
            {
                "strategy_A": "A2",
                "strategy_B": "B2",
                "checked_score": 0.69,
                "pairwise_comparisons": [
                    {
                        "against_strategy_A": "A1",
                        "against_strategy_B": "B1",
                        "statistically_better": False,
                    },
                    {
                        "against_strategy_A": "A2",
                        "against_strategy_B": "B2",
                        "statistically_better": False,
                    },
                    {
                        "against_strategy_A": "A3",
                        "against_strategy_B": "B3",
                        "statistically_better": True,
                    },
                ],
            },
            {
                "strategy_A": "A3",
                "strategy_B": "B3",
                "checked_score": 0.60,
                "pairwise_comparisons": [],
            },
        ]

        annotate_certainty_groups(rows)

        self.assertEqual([row["certainty_group"] for row in rows], [1, 1, 2])
        self.assertFalse(certainty_ranking_is_complete(rows))
        rows[0]["pairwise_comparisons"][1]["statistically_better"] = True
        annotate_certainty_groups(rows)
        self.assertEqual([row["certainty_group"] for row in rows], [1, 2, 3])
        self.assertTrue(certainty_ranking_is_complete(rows))

    def test_sequential_comparison_returns_stopping_metadata(self):
        strategy_a_1 = make_strategy("A1", "A", genes=[0, 0, 0, 0])
        strategy_b_1 = make_strategy("B1", "B", genes=[0, 0, 0, 0])
        strategy_a_2 = make_strategy("A2", "A", genes=[1, 1, 1, 1])
        strategy_b_2 = make_strategy("B2", "B", genes=[1, 1, 1, 1])

        rows = sequential_comparison_rows(
            [
                find_best_pair([strategy_a_1], [strategy_b_1], [[0.6]]),
                find_best_pair([strategy_a_2], [strategy_b_2], [[0.5]]),
            ],
            [(strategy_a_1, strategy_b_1), (strategy_a_2, strategy_b_2)],
            max_trials=20,
            batch_size=5,
            sequence_length=20,
            success_condition_name="same-bit",
            seed=1,
        )

        self.assertEqual(rows[0]["trials"] % 5, 0)
        self.assertTrue(rows[0]["sequential"])
        self.assertIn(
            rows[0]["stopping_reason"],
            {"max_trials_reached", "full_ranking_statistically_separated"},
        )
        self.assertGreaterEqual(rows[0]["batches_completed"], 1)
        self.assertIn("certainty_group", rows[0])

    def test_pair_comparison_loads_generation_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            snapshots_dir = output_dir / "population_snapshots"
            matrices_dir = output_dir / "score_matrices"
            snapshots_dir.mkdir()
            matrices_dir.mkdir()

            strategy_a = make_strategy("A1", "A")
            strategy_b = make_strategy("B1", "B")
            (snapshots_dir / "population_A_generation_0002.json").write_text(
                json.dumps([asdict(strategy_a)]),
                encoding="utf-8",
            )
            (snapshots_dir / "population_B_generation_0002.json").write_text(
                json.dumps([asdict(strategy_b)]),
                encoding="utf-8",
            )
            (matrices_dir / "score_matrix_generation_0002.csv").write_text(
                "strategy_A,B1\nA1,0.750000\n",
                encoding="utf-8",
            )

            self.assertEqual(latest_generation(output_dir), 2)
            self.assertEqual(load_population(output_dir, "A", 2)[0].id, "A1")
            self.assertEqual(load_score_matrix(output_dir, 2), [[0.75]])

    def test_run_genetic_algorithm_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = GAConfig(
                population_size_A=4,
                population_size_B=4,
                observed_length_min=1,
                observed_length_max=2,
                target_choice_range_min=2,
                target_choice_range_max=2,
                trials_per_pair=4,
                generations=1,
                elite_count=1,
                output_dir=temp_dir,
                seed=3,
                initial_population_A=[
                    {
                        "id": "A-seeded",
                        "observed_length": 1,
                        "target_choice_range": 2,
                        "genes": [0, 0],
                    }
                ],
                initial_population_B=[
                    {
                        "id": "B-seeded",
                        "observed_length": 1,
                        "target_choice_range": 2,
                        "genes": [0, 0],
                    }
                ],
            )

            best_a, best_b = run_genetic_algorithm(
                config,
                custom_success_condition=lambda _a_value, _b_value: True,
            )

            output_dir = Path(temp_dir)
            self.assertEqual(best_a.survival_ratio, 1.0)
            self.assertEqual(best_b.survival_ratio, 1.0)
            self.assertTrue((output_dir / "experiment_config.yaml").exists())
            self.assertTrue((output_dir / "experiment_manifest.json").exists())
            self.assertTrue((output_dir / "generation_stats.csv").exists())
            self.assertTrue((output_dir / "best_pairs.csv").exists())
            self.assertTrue((output_dir / "best_pair.json").exists())
            self.assertTrue((output_dir / "experiment_report.html").exists())
            self.assertTrue((output_dir / "best_A_strategy.json").exists())
            self.assertTrue((output_dir / "best_B_strategy.json").exists())
            self.assertTrue(
                (
                    output_dir
                    / "score_matrices"
                    / "score_matrix_generation_0001.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "population_snapshots"
                    / "population_A_generation_0001.json"
                ).exists()
            )

            with (output_dir / "generation_stats.csv").open(
                newline="",
                encoding="utf-8",
            ) as stats_file:
                rows = list(csv.reader(stats_file))
            self.assertEqual(len(rows), 3)

            with (output_dir / "best_A_strategy.json").open(encoding="utf-8") as best_file:
                saved_best_a = json.load(best_file)
            self.assertEqual(saved_best_a["player_type"], "A")

            with (output_dir / "best_pair.json").open(encoding="utf-8") as pair_file:
                saved_best_pair = json.load(pair_file)
            self.assertEqual(saved_best_pair["score"], 1.0)

            with (output_dir / "experiment_manifest.json").open(
                encoding="utf-8",
            ) as manifest_file:
                manifest = json.load(manifest_file)
            self.assertEqual(manifest["completed_generation"], 1)
            self.assertEqual(manifest["configured_generations"], 1)
            self.assertEqual(manifest["trials_per_pair"], 4)
            self.assertEqual(manifest["population_size_A"], 4)
            self.assertIn("config_hash_sha256", manifest)
            self.assertIn("elapsed_seconds", manifest)

            report = (output_dir / "experiment_report.html").read_text(
                encoding="utf-8",
            )
            self.assertIn("Daemon Coin Strategy Report", report)
            self.assertIn("Final Generation Best Pair", report)
            self.assertIn("Top 3 Distinct Final Generation Pairs", report)
            self.assertIn("Use that observed bit pattern as a lookup key", report)

            with (
                output_dir
                / "population_snapshots"
                / "population_A_generation_0000.json"
            ).open(encoding="utf-8") as snapshot_file:
                initial_a = json.load(snapshot_file)
            self.assertEqual(initial_a[0]["id"], "A-seeded")
            self.assertEqual(initial_a[0]["creation_method"], "initial-seed")


if __name__ == "__main__":
    unittest.main()
