"""One training loop for every NHMM trainer (original, weighted, structured, data-scaling).

Budgets: `epochs` (permutation per epoch from the global numpy stream, early stopping on the
first criterion; this reproduces the original trainers) or `steps` (a fixed number of optimizer
updates drawn from seeded epoch-wise permutations, evaluated every `eval_every` updates).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch


@dataclass
class Budget:
    mode: str = "epochs"      # epochs | steps
    epochs: int = 10
    patience: int = 3
    steps: int = 1200
    eval_every: int = 40
    batch_size: int = 4


@dataclass
class FitResult:
    history: list
    best: dict                 # criterion -> (value, step)
    steps: int
    schedule: list = field(default_factory=list)


def fit(model, train, batch_loss: Callable, evaluate: Callable, budget: Budget, *,
        criteria=(("objective", "min"),), on_best: Optional[Callable] = None,
        train_summary: Callable = lambda sums: sums, lr=8e-4, weight_decay=1e-5, clip=2.0,
        rng=None, on_epoch_start: Optional[Callable] = None) -> FitResult:
    """batch_loss(model, batch) -> (loss, {stat: float}) with stats summed between evaluations;
    evaluate(model) -> {name: value} including every criterion; on_best(criterion, step) saves."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = {k: (float("inf") if mode == "min" else -float("inf"), 0) for k, mode in criteria}
    history, schedule, steps, sums = [], [], 0, {}

    def step(batch):
        nonlocal steps
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, stats = batch_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {steps}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        steps += 1
        for k, v in stats.items():
            sums[k] = sums.get(k, 0.0) + v

    def checkpoint(epoch):
        row = {"epoch": epoch, "step": steps, **train_summary(dict(sums)), **evaluate(model)}
        sums.clear()
        improved = False
        for k, mode in criteria:
            v = row[k]
            if (v < best[k][0] - 1e-5) if mode == "min" else (v > best[k][0] + 1e-9):
                best[k] = (v, steps)
                improved |= k == criteria[0][0]
                if on_best:
                    on_best(k, steps)
        history.append(row)
        return improved

    if budget.mode == "epochs":
        wait = 0
        for epoch in range(1, budget.epochs + 1):
            if on_epoch_start:
                on_epoch_start(epoch)
            order = np.random.permutation(len(train))
            schedule.append(order.tolist())
            for i in range(0, len(order), budget.batch_size):
                step([train[j] for j in order[i:i + budget.batch_size]])
            wait = 0 if checkpoint(epoch) else wait + 1
            if wait >= budget.patience:
                break
    else:
        rng = rng or np.random.default_rng(0)
        queue = []
        while steps < budget.steps:
            while len(queue) < budget.batch_size:
                order = rng.permutation(len(train)).tolist()
                schedule.append(order)
                queue += order
            batch, queue = queue[:budget.batch_size], queue[budget.batch_size:]
            step([train[j] for j in batch])
            if steps % budget.eval_every == 0 or steps == budget.steps:
                checkpoint(steps * budget.batch_size / len(train))
    return FitResult(history, best, steps, schedule)
