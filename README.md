# daemon-coin-mathematicians

Monte Carlo simulation for the “daemon coin” probability game.  
Inspired by: https://youtu.be/oAj4xPXKzwg

---

## Overview

This project simulates a coordination problem under incomplete information.

Two players observe different halves of the same conceptual infinite coin-toss process and must independently choose hidden positions.  
The objective is to design a shared strategy that gives them a probability of success greater than 50%.

The implementation uses **Monte Carlo simulation** and **pattern-based stopping rules**.

---

## Problem Statement

An Evil Spirit captures two mathematicians and isolates them in separate rooms.

He flips a fair coin infinitely many times:

- Player A observes the outcomes of all **even-indexed** tosses
- Player B observes the outcomes of all **odd-indexed** tosses

Each player must then choose the index of a toss whose value is **unknown** to them:

- Player A must choose an **odd** index
- Player B must choose an **even** index

### Winning Condition

- If the values at the two chosen positions are equal → both players survive
- Otherwise → both players lose

Before separation, they are allowed to agree on a deterministic strategy.

---

## What This Program Simulates

The code approximates the infinite process using two finite random binary sequences.

For each trial:

1. Two random binary sequences are generated
2. Each player searches their own sequence for the **first occurrence** of a configured pattern
3. The index found by Player A is applied to Player B’s sequence
4. The index found by Player B is applied to Player A’s sequence
5. If the two selected values are equal, the trial counts as success

In code terms, success is evaluated by comparing:

```python
mathematist1[index2] == mathematist2[index1]
```

---

## Strategy

The implemented strategy is based on **pattern matching**.

Each player:

1. Observes their own binary sequence
2. Searches for the **first occurrence** of a predefined pattern
3. Uses the position of that occurrence as their effective choice

Patterns are configured like this:

```python
m1_pattern = r"0"
m2_pattern = r"0"
```

The implementation uses Python regular expressions, so patterns are interpreted as regex expressions.

---

## Random Pattern Mode

The code also supports a mode where both players use the same randomly generated pattern.

Example configuration:

```python
random_pattern = True
random_pattern_length = 3
```

In this mode:

- one random binary pattern is generated
- both players use that same pattern
- the simulation estimates the corresponding survival probability

---

## Default Parameters

```python
random_pattern = False
random_pattern_length = 3
m1_pattern = r"0"
m2_pattern = r"0"
trials = 10000
serie_length = 1000
```

---

## Example Results

Some example outcomes produced by this strategy family:

| Patterns        | Estimated Survival Probability |
|----------------|--------------------------------|
| `[0, 0]`       | ~66.78%                        |
| `[01, 01]`     | ~55.10%                        |
| `[01, 00]`     | ~62.55%                        |
| Random length 3| ~51.78%                        |

Because this is a Monte Carlo simulation, exact values may vary slightly between runs.

---

## Key Insight

Even though each player sees only half of the underlying random process, carefully coordinated stopping rules can create statistical dependence between their choices.

That dependence allows them to outperform naive random guessing and achieve:

```text
P(success) > 0.5
```

without communication after separation.

---

## Notes and Limitations

- The simulation uses **finite** sequences, not truly infinite ones
- Pattern search is based on the **first match**
- Sequence length must be large enough so the configured pattern is likely to appear
- Since regular expressions are used, pattern syntax follows Python `re` semantics
- Results are empirical estimates rather than formal proofs

---

## How to Run

```bash
git clone https://github.com/nikiigo/daemon-coin-mathematicians.git
cd daemon-coin-mathematicians
python daemon_coin.py
```

---

## Output Example

Example output for fixed patterns:

```text
Patterns: [0 0]. The survival ratio is approximately 66.78%.
```

Example output for random-pattern mode:

```text
Random pattern with length 3. The survival ratio is approximately 51.78%.
```

---

## File Structure

```text
daemon_coin.py   # main simulation script
```

---

## Possible Extensions

- Analytical derivation of success probabilities
- Search for optimal patterns
- Genetic algorithm optimization of strategies
- Visualization of convergence over trials
- Comparison between empirical and theoretical results

---

## Author

Igor Nikitin
# daemon-coin-mathematicians-ga
