# Stage 0.2F: Paired Multi-Seed Stability Evaluation

## 1. Why separate the seed roles

A single `seed` previously controlled several unrelated sources of randomness:

- synthetic market generation;
- model initialization;
- dropout;
- incorrect graph permutation.

Changing that seed changed the problem and the model simultaneously. The result
could not reveal whether performance variation came from the data or training.

Stage 0.2F uses three explicit roles.

### Data seed

\[
s_{\text{data}}
\]

It generates one fixed synthetic market. All model runs see the same features,
targets, graph, and chronological split.

### Model seed

\[
s_{\text{model}}
\]

It controls neural initialization and training randomness. Stage 0.2F varies
this seed across:

```text
42, 43, 44, 45, 46
```

### Graph permutation seed

\[
s_{\text{perm}}
\]

It creates one fixed incorrect graph:

\[
A_{\text{perm}}=P A P^T
\]

The control graph remains unchanged across model seeds.

## 2. Why this is a paired experiment

For each model seed, all compared models use the same:

- synthetic market;
- train/validation/test split;
- correct graph;
- incorrect graph;
- optimizer settings;
- maximum epochs;
- early-stopping rule.

Therefore WAPE differences can be computed within each seed:

\[
\Delta_s
=
\operatorname{WAPE}_{\text{static},s}
-
\operatorname{WAPE}_{\text{reference},s}
\]

A negative value means Safe Static Graph GRU is better.

## 3. Aggregate statistics

For \(n\) model seeds, the mean WAPE is:

\[
\bar{x}
=
\frac{1}{n}\sum_{i=1}^{n}x_i
\]

The sample standard deviation is:

\[
s
=
\sqrt{
\frac{
\sum_{i=1}^{n}(x_i-\bar{x})^2
}{
n-1
}
}
\]

The standard error is:

\[
SE=\frac{s}{\sqrt{n}}
\]

For five runs, the 95% confidence interval uses the Student-\(t\) critical value
with four degrees of freedom:

\[
\bar{x}\pm 2.776SE
\]

With only five seeds, the interval can be wide. It should be interpreted as a
stability diagnostic, not a publication-grade statistical conclusion.

## 4. Decision criteria

The graph hypothesis receives stronger support when:

1. Safe Static Graph GRU has lower mean WAPE than Residual Temporal GRU.
2. It wins against the temporal model in most or all paired seeds.
3. It beats the permuted graph in most or all paired seeds.
4. Its paired mean delta is negative.
5. Its advantage is not produced by one extreme seed.

If the confidence interval of the paired delta includes zero, the evidence is
still uncertain even when the mean is favorable.

## 5. Output files

### `per_seed_metrics.csv`

One row per model and model seed.

### `aggregate_metrics.csv`

Mean, standard deviation, standard error, and 95% interval by model.

### `paired_comparisons.csv`

Safe Static Graph GRU win counts and WAPE deltas against each control.

### `experiment_summary.md`

A human-readable audit of the final Stage 0.2F result.
