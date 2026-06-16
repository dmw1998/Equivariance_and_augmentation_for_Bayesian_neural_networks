"""Shared training / evaluation loops for the theorem-verification baselines.

Both ``train_theorem2.py`` and ``train_theorem4.py`` use the same plain mean-field ELBO training loop (AdamW + cosine schedule, beta = 1/N) and the same MC-averaged test-accuracy routine, so they live here to avoid duplication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _unpack(batch):
    if len(batch) == 3:
        inputs, targets, _ = batch
    else:
        inputs, targets = batch
    return inputs, targets


def train_baseline_bnn(model, train_loader, device, num_epochs, lr, train_samples, weight_decay, use_wandb):
    """Train a Bayesian model by maximising the ELBO with beta = 1/N and a cosine LR schedule."""
    import wandb

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    kl_weight = 1.0 / len(train_loader.dataset)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs, targets = _unpack(batch)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            total_nll = 0.0
            total_kl = 0.0
            outputs = None
            for _ in range(train_samples):
                outputs, kl = model(inputs, return_kl=True)
                total_nll = total_nll + criterion(outputs, targets)
                total_kl = total_kl + kl

            loss = (total_nll + kl_weight * total_kl) / inputs.size(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += outputs.max(1)[1].eq(targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": total_loss / len(train_loader.dataset),
                    "train/accuracy": 100.0 * correct / total,
                    "train/lr": scheduler.get_last_lr()[0],
                }
            )


def evaluate_mc_accuracy(model, test_loader, device, mc_samples):
    """Test accuracy with MC averaging of the predictive distribution.

    The model is put in ``train()`` mode inside the loop so that the Bayesian
    layers draw fresh posterior samples; this is intentional (the layers have no
    BatchNorm/Dropout that ``eval()`` would otherwise need to disable).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = _unpack(batch)
            inputs, targets = inputs.to(device), targets.to(device)

            probs = []
            for _ in range(mc_samples):
                model.train()  # enable stochastic forward
                probs.append(F.softmax(model(inputs, return_kl=False), dim=1))
            pred = torch.stack(probs).mean(0).argmax(1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    model.eval()
    return 100.0 * correct / total
