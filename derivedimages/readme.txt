# CS656 Reinforcement Learning – Programming Assignment
## Part I, Problem 8: Cliff Walking (Sutton & Barto, Example 6.6)
### Comparison of Sarsa (on-policy) and Q-learning (off-policy)

---

## 1. Problem description

The agent moves on a 4 × 12 gridworld from the start cell **S** (bottom-left) to the goal cell **G** (bottom-right). The cells between S and G in the bottom row form **the cliff**.

| Item | Definition |
|---|---|
| States | Grid cells (row, column) |
| Actions | Up, down, left, right |
| Reward | −1 for every step |
| Cliff | Stepping on it gives −100 and sends the agent back to S |
| Episode ends | When the agent reaches G |
| Discount | γ = 1 (episodic task) |
| Learning rate | α = 0.5 |
| Exploration | ε = 0.1 |

---

## 2. ε-greedy policy

At every step the agent chooses:

- with probability **1 − ε**, the greedy action (highest Q-value),
- with probability **ε**, a random action chosen uniformly from all actions.

$$
\pi(a \mid s) =
\begin{cases}
1-\varepsilon+\dfrac{\varepsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s,a') \\[2mm]
\dfrac{\varepsilon}{|A|} & \text{otherwise}
\end{cases}
$$

With |A| = 4 and ε = 0.1:

- Greedy action: 0.9 + 0.025 = **0.925**
- Each of the other 3 actions: **0.025**
- Check: 0.925 + 3(0.025) = 1

---

## 3. The two update rules

**(i) Sarsa (on-policy)**

$$
Q(S,A) \leftarrow Q(S,A) + \alpha\big[R + \gamma\,Q(S',A') - Q(S,A)\big]
$$

Here A′ is the action actually chosen in S′ by the ε-greedy policy.

**(ii) Q-learning (off-policy)**

$$
Q(S,A) \leftarrow Q(S,A) + \alpha\big[R + \gamma\,\max_{a} Q(S',a) - Q(S,A)\big]
$$

The only difference is **Q(S′, A′)** versus **max_a Q(S′, a)**.

---

## 4. Worked illustration

Suppose the agent is in a cell just above the cliff and moves right with R = −1. The current estimates are:

- Q(S, right) = −10
- At S′: Q(S′, right) = −8 (the best action) and Q(S′, down) = −40 (the cliff direction)

**(i) Q-learning** always uses the max (−8):

$$
\begin{aligned}
Q(S,A) &= -10 + 0.5\,[-1 + 1(-8) - (-10)] \\
       &= -10 + 0.5\,[-9 + 10] \\
       &= -10 + 0.5(1) \\
       &= \mathbf{-9.5}
\end{aligned}
$$

**(ii) Sarsa**, when the ε-greedy policy happens to pick the random action "down" at S′ (probability 0.025):

$$
\begin{aligned}
Q(S,A) &= -10 + 0.5\,[-1 + 1(-40) - (-10)] \\
       &= -10 + 0.5\,[-1 - 40 + 10] \\
       &= -10 + 0.5(-31) \\
       &= -10 + (-15.5) \\
       &= \mathbf{-25.5}
\end{aligned}
$$

**Interpretation:** Sarsa's estimate for the edge cell is pulled down every time exploration takes the bad action, so it learns that the edge is dangerous. Q-learning ignores these bad outcomes in its target.

**Path lengths (no falls):**

- Optimal path = 1 up + 11 right + 1 down = **13 steps** (return −13)
- Safer path = 3 up + 11 right + 3 down = **17 steps** (return −17), via the top row

**Risk along the edge:** on the optimal path the agent passes 10 cells directly above the cliff. At each cell the chance of a random "down" move is ε/|A| = 0.025.

$$
P(\text{no fall}) = (1 - 0.025)^{10} \approx 0.776
$$

So about 22% of attempts fall at least once, and each fall costs roughly −100. This is why Q-learning's online reward is far below the −13 it would earn with no exploration.

---

## 5. Results

![Book-style figure](images/cliff_walking_book_figure.png)

*Left: gridworld with the optimal path (red) and the safer path (blue). Right: online performance, averaged over 100 independent runs of 500 episodes.*

| Method | Avg reward (last 100 episodes) | Learned greedy path |
|---|---|---|
| Sarsa | about −27 | Safe, upper route (about 17 steps) |
| Q-learning | about −50 | Optimal, along the cliff edge (13 steps) |

### Reward curves
![Reward curves](images/cliff_walking_result.png)

### Learned paths
![Learned paths](images/cliff_paths.png)

### Learned state values
![State values](images/cliff_values.png)

---

## 6. Discussion

- **Q-learning** learns the values of the optimal policy and finds the shortest path along the cliff edge. However, it behaves ε-greedily while learning, so random exploratory moves sometimes push it off the cliff. Its **online performance is worse**.
- **Sarsa** learns the values of the ε-greedy policy it actually follows. It takes the risk of exploration into account, so it learns the longer but safer path through the upper rows. Its **online performance is better**.
- If ε were gradually reduced towards 0, both methods would converge to the optimal policy.

---

## 7. Exercises from the book

**Exercise 6.11: Why is Q-learning considered an off-policy control method?**
The behaviour policy that generates the actions is ε-greedy, but the update target uses max_a Q(S′, a), which is the **greedy** target policy. Because the policy being learned differs from the policy generating the data, Q-learning is off-policy. In Sarsa the target uses A′ drawn from the same ε-greedy policy, so it is on-policy.

**Exercise 6.12: If action selection is greedy, is Q-learning exactly the same as Sarsa? Will they make exactly the same action selections and weight updates?**
With greedy selection (ε = 0), Sarsa's A′ = argmax_a Q(S′, a), so Q(S′, A′) = max_a Q(S′, a) and the two update targets are the same. They are nearly the same algorithm but **not always identical**: after Q-learning updates Q(S, A), it chooses its next action from the updated table, whereas Sarsa had already chosen A′ before the update. If S′ = S (for example, bumping into a wall), the update changes the values at S′, so the next action can differ. Tie-breaking can also make them differ.

---

## 8. Repository contents

```
CS656-RL-Assignment/
├── README.md
├── cliff_walking.ipynb      # Colab notebook (code + outputs)
├── cliff_walking.py         # Python script version
└── images/                  # Result figures used in this README
```

## 9. How to run

1. Open `cliff_walking.ipynb` in Google Colab (or run `python cliff_walking.py` locally).
2. Run all cells. Only `numpy` and `matplotlib` are required.
3. Training takes about a minute with 100 runs. Increase `RUNS` to 500 for smoother curves.

## 10. Reference

Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*, Example 6.6 (Cliff Walking), Exercises 6.11 and 6.12.
