# Code Reading Guide

Read the project in this order.

## Pass 1: understand interfaces

1. `src/autogfm/config.py`
2. `src/autogfm/data/synthetic_market.py`
3. `src/autogfm/graph/ops.py`
4. `src/autogfm/models/simple_graph_temporal.py`
5. `scripts/train_synthetic.py`

At this stage, focus only on inputs, outputs, and tensor shapes.

## Pass 2: trace one sample

Use a batch size of 2 and write down every shape:

```text
x
node_sequences
final_hidden
temporal_state
graph_state
gate
fused_state
forecast
```

## Pass 3: explain every transformation

You should be able to answer:

1. Why is `permute` needed before the GRU?
2. Why is the vehicle dimension merged into the batch dimension?
3. Why are self-loops added to the graph?
4. What does the Einstein summation string mean?
5. Why is the gate vector-valued rather than scalar?
6. Why must the train-validation split preserve chronology?

## Pass 4: controlled modifications

Make only one change at a time:

1. Remove graph aggregation and compare validation loss.
2. Replace the gate with simple addition.
3. Change the history from 12 to 6 months.
4. Change graph sparsity from four neighbors to two.
5. Replace MSE with MAE.

Predict the effect of each change before running it.
