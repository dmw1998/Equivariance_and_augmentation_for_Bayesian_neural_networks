"""Equivariance and accuracy metrics.

Consolidates the metric functions that were previously split across ``equivariance_val.py`` (single-sample and the core defect), the unified trainer (MC-averaged variants), and the Theorem-4 script (T-sweep and K-fold defect).

For classification the group acts trivially on the label space, so the equivariance defect measures how much the predictive distribution changes under input rotation:

    Delta_F^eq = (1/N) sum_i (1/|G|) sum_g || F_mc(g x_i) - F_mc(x_i) ||^2,

where ``F_mc(x) = (1/T) sum_t softmax(f(x; theta^(t)))``.
"""

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# MC posterior prediction
# --------------------------------------------------------------------------- #
def predict_probs_mc(model, inputs, eval_samples):
    """Mean softmax over ``eval_samples`` posterior draws (model must be in a stochastic mode)."""
    probs_mc = []
    for _ in range(eval_samples):
        logits = model(inputs, return_kl=False)
        probs_mc.append(F.softmax(logits, dim=1))
    return torch.stack(probs_mc, dim=0).mean(dim=0)


# --------------------------------------------------------------------------- #
# Accuracy / loss
# --------------------------------------------------------------------------- #
def _iter_eval_batches(test_loader, equiv_loader, eval_source):
    if eval_source == "test":
        for batch in test_loader:
            yield batch[0], batch[1]
        return
    if eval_source == "equiv_all":
        for batch in equiv_loader:
            if len(batch) != 5:
                continue
            img_0, img_90, img_180, img_270, targets = batch
            for img in (img_0, img_90, img_180, img_270):
                yield img, targets
        return
    raise ValueError(f"Unknown eval_source: {eval_source}")


def eval_pass(
    model, test_loader, equiv_loader, device, eval_samples, eval_source="test", return_distribution=False
):
    """MC-averaged accuracy + NLL over the chosen evaluation source."""
    model.eval()
    correct = count = 0
    loss_sum = 0.0
    pred_labels = []
    with torch.no_grad():
        for inputs, targets in _iter_eval_batches(test_loader, equiv_loader, eval_source):
            inputs = inputs.to(device)
            targets = targets.to(device)
            probs_avg = predict_probs_mc(model, inputs, eval_samples)
            loss_sum += F.nll_loss(torch.log(probs_avg.clamp_min(1e-12)), targets).item() * inputs.size(0)
            pred = probs_avg.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            count += targets.size(0)
            if return_distribution:
                pred_labels.append(pred.detach().cpu().numpy())
    acc = 100.0 * correct / count
    avg_loss = loss_sum / count
    if not return_distribution:
        return acc, avg_loss
    pred_labels = np.concatenate(pred_labels) if pred_labels else np.array([])
    return acc, avg_loss, {"pred_labels": pred_labels}


def evaluate_classification_accuracy(model, device, test_loader, mc_samples=1):
    """Top-1 accuracy. ``mc_samples > 1`` averages the predictive distribution over posterior draws."""
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            img, label = batch[0], batch[1]
            img, label = img.to(device), label.to(device)
            if mc_samples > 1:
                model.train()  # enable stochastic sampling
                preds = [F.softmax(model(img, return_kl=False), dim=1) for _ in range(mc_samples)]
                pred = torch.stack(preds).mean(dim=0).argmax(dim=1)
            else:
                model.eval()  # use mean weights
                pred = model(img, return_kl=False).argmax(dim=1)
            correct += pred.eq(label).sum().item()
            total += label.size(0)
    return correct / total


# --------------------------------------------------------------------------- #
# Single-sample (mean-weight) equivariance metrics
# --------------------------------------------------------------------------- #
def compute_orbits_same_pred(model, device, equiv_loader):
    """Mean over samples of (1/|G|) sum_g 1[pred(g x) == pred(x)] using mean weights."""
    total_indicator_sum = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch in equiv_loader:
            if len(batch) != 5:
                continue
            img_0, img_90, img_180, img_270, _ = batch
            preds = []
            for img in (img_0, img_90, img_180, img_270):
                img = img.to(device)
                preds.append(model(img, return_kl=False).argmax(dim=1).cpu())
            preds = torch.stack(preds)  # [4, batch]
            for i in range(preds.shape[1]):
                pred_0 = preds[0, i]
                indicator_sum = sum(0.25 for j in range(4) if preds[j, i] == pred_0)
                total_indicator_sum += indicator_sum
                total_samples += 1
    return total_indicator_sum / total_samples if total_samples > 0 else 0.0


def compute_symmetric_kl_divergence(model, device, equiv_loader):
    """Symmetric KL between the 0-degree and each rotated predictive distribution (mean weights)."""
    kl_values = []
    model.eval()
    eps = 1e-8
    with torch.no_grad():
        for batch in equiv_loader:
            if len(batch) != 5:
                continue
            img_0, img_90, img_180, img_270, _ = batch
            probs = []
            for img in (img_0, img_90, img_180, img_270):
                img = img.to(device)
                probs.append(F.softmax(model(img, return_kl=False), dim=1).cpu())
            probs = torch.stack(probs)  # [4, batch, num_classes]
            for i in range(probs.shape[1]):
                p0 = probs[0, i] + eps
                for j in (1, 2, 3):
                    p_rot = probs[j, i] + eps
                    kl_01 = F.kl_div(p0.log(), p_rot, reduction="batchmean")
                    kl_10 = F.kl_div(p_rot.log(), p0, reduction="batchmean")
                    kl_values.append(float(kl_01 + kl_10))
    if kl_values:
        return {
            "mean": float(np.mean(kl_values)),
            "variance": float(np.var(kl_values)),
            "std": float(np.std(kl_values)),
            "all_kl": kl_values,
        }
    return {"mean": 0.0, "variance": 0.0, "std": 0.0, "all_kl": []}


# --------------------------------------------------------------------------- #
# MC-averaged equivariance metrics
# --------------------------------------------------------------------------- #
def compute_orbits_same_pred_mc(model, equiv_loader, device, eval_samples):
    """MC-averaged orbit consistency: mean over samples of (max orbit-vote count) / |G|."""
    total_indicator_sum = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch in equiv_loader:
            if len(batch) != 5:
                continue
            img_0, img_90, img_180, img_270, _ = batch
            preds = []
            for img in (img_0, img_90, img_180, img_270):
                img = img.to(device)
                preds.append(predict_probs_mc(model, img, eval_samples).argmax(dim=1).cpu())
            preds = torch.stack(preds)
            for i in range(preds.shape[1]):
                _, counts = torch.unique(preds[:, i], return_counts=True)
                total_indicator_sum += counts.max().item() / 4.0
                total_samples += 1
    return total_indicator_sum / total_samples if total_samples > 0 else 0.0


def compute_symmetric_kl_divergence_mc(model, equiv_loader, device, eval_samples):
    """MC-averaged symmetric KL between the 0-degree and each rotated predictive distribution."""
    kl_values = []
    model.eval()
    eps = 1e-8
    with torch.no_grad():
        for batch in equiv_loader:
            if len(batch) != 5:
                continue
            img_0, img_90, img_180, img_270, _ = batch
            probs = []
            for img in (img_0, img_90, img_180, img_270):
                img = img.to(device)
                probs.append(predict_probs_mc(model, img, eval_samples).cpu())
            probs = torch.stack(probs)
            for i in range(probs.shape[1]):
                p0 = probs[0, i] + eps
                for j in (1, 2, 3):
                    p_rot = probs[j, i] + eps
                    kl_01 = F.kl_div(p0.log(), p_rot, reduction="batchmean")
                    kl_10 = F.kl_div(p_rot.log(), p0, reduction="batchmean")
                    kl_values.append(float(kl_01 + kl_10))
    if kl_values:
        return {"mean": float(np.mean(kl_values)), "std": float(np.std(kl_values))}
    return {"mean": 0.0, "std": 0.0}


def run_validation(model, test_loader, equiv_loader, device, eval_samples, val_source):
    """Bundle accuracy/loss with the MC equivariance metrics. Returns (metrics, distribution)."""
    val_acc, val_loss, val_dist = eval_pass(
        model,
        test_loader,
        equiv_loader,
        device,
        eval_samples,
        eval_source=val_source,
        return_distribution=True,
    )
    osp = compute_orbits_same_pred_mc(model, equiv_loader, device, eval_samples)
    kl = compute_symmetric_kl_divergence_mc(model, equiv_loader, device, eval_samples)
    metrics = {
        "val/accuracy": val_acc,
        "val/loss": val_loss,
        "val/osp": osp,
        "val/symmetric_kl_div_mean": kl["mean"],
        "val/symmetric_kl_div_std": kl["std"],
    }
    return metrics, val_dist


# --------------------------------------------------------------------------- #
# Equivariance defect (Delta_F^eq)
# --------------------------------------------------------------------------- #
def compute_equivariance_defect(model, device, equiv_loader, mc_samples=10):
    """Empirical equivariance defect with MC posterior averaging.

    Returns a dict with keys ``mean``, ``std`` and ``per_sample``.
    """
    per_sample_defects = []
    for batch in equiv_loader:
        if len(batch) != 5:
            continue
        img_0, img_90, img_180, img_270, _ = batch
        imgs = [img_0, img_90, img_180, img_270]

        all_preds = []
        for img in imgs:
            img = img.to(device)
            mc_preds = []
            for _ in range(mc_samples):
                model.train()  # enable stochastic sampling
                with torch.no_grad():
                    prob = torch.softmax(model(img, return_kl=False), dim=1)
                mc_preds.append(prob.cpu())
            all_preds.append(torch.stack(mc_preds, dim=0))  # [T, batch, num_classes]

        f_mc = torch.stack([p.mean(dim=0) for p in all_preds], dim=0)  # [4, batch, num_classes]
        f_ref = f_mc[0]
        for i in range(f_ref.shape[0]):
            defect = sum(torch.sum((f_mc[g, i] - f_ref[i]) ** 2).item() for g in range(4)) / 4.0
            per_sample_defects.append(defect)

    model.eval()
    per_sample_defects = np.array(per_sample_defects)
    return {
        "mean": float(per_sample_defects.mean()),
        "std": float(per_sample_defects.std()),
        "per_sample": per_sample_defects.tolist(),
    }


def compute_equivariance_defect_T_sweep(model, device, equiv_loader, T_list):
    """Equivariance defect for a range of MC sample sizes T in a single pass.

    Draws ``T_max`` posterior samples per rotation and uses a cumulative mean
    along the T axis to recover ``F_mc^(T)`` for every T without re-running
    forward passes.  Returns ``{T: {'mean', 'std', 'per_sample'}}``.
    """
    T_list = sorted({int(T) for T in T_list})
    T_max = T_list[-1]
    per_sample_defects = {T: [] for T in T_list}

    for batch in equiv_loader:
        if len(batch) != 5:
            continue
        img_0, img_90, img_180, img_270, _ = batch
        imgs = [img_0.to(device), img_90.to(device), img_180.to(device), img_270.to(device)]

        all_probs = []
        for img in imgs:
            mc_preds = []
            for _ in range(T_max):
                model.train()  # enable stochastic sampling
                with torch.no_grad():
                    mc_preds.append(torch.softmax(model(img, return_kl=False), dim=1))
            all_probs.append(torch.stack(mc_preds, dim=0))  # [T_max, B, C]

        denom = torch.arange(1, T_max + 1, device=device).view(-1, 1, 1).float()
        cum_means = [p.cumsum(dim=0) / denom for p in all_probs]  # each [T_max, B, C]

        for T in T_list:
            f_mc = torch.stack([cm[T - 1] for cm in cum_means], dim=0)  # [4, B, C]
            f_ref = f_mc[0]
            diff = f_mc - f_ref.unsqueeze(0)
            sq = (diff**2).sum(dim=-1)  # [4, B]
            per_sample_defects[T].extend(sq.mean(dim=0).cpu().tolist())

    model.eval()
    results = {}
    for T in T_list:
        arr = np.asarray(per_sample_defects[T], dtype=np.float64)
        results[T] = {
            "mean": float(arr.mean()) if arr.size > 0 else 0.0,
            "std": float(arr.std()) if arr.size > 0 else 0.0,
            "per_sample": arr.tolist(),
        }
    return results


def compute_equivariance_defect_K_fold(model, device, equiv_loader, T_list, K=30):
    """K-fold MC evaluation of the equivariance defect.

    Re-runs the T-sweep K times with independent posterior samples and reports
    per-T mean and std across the K realisations.  The std across realisations
    is an empirical estimate of the McDiarmid-type MC deviation, predicted to
    scale as O(1/sqrt(T)).  Returns ``{T: {'mc_mean', 'mc_std_K', 'per_run_means'}}``.
    """
    T_list = sorted({int(T) for T in T_list})
    T_max = T_list[-1]
    per_run_per_T = [{T: [] for T in T_list} for _ in range(K)]

    for batch in equiv_loader:
        if len(batch) != 5:
            continue
        img_0, img_90, img_180, img_270, _ = batch
        imgs = [img_0.to(device), img_90.to(device), img_180.to(device), img_270.to(device)]

        for k in range(K):
            all_probs = []
            for img in imgs:
                mc_preds = []
                for _ in range(T_max):
                    model.train()  # enable stochastic sampling
                    with torch.no_grad():
                        mc_preds.append(torch.softmax(model(img, return_kl=False), dim=1))
                all_probs.append(torch.stack(mc_preds, dim=0))  # [T_max, B, C]

            denom = torch.arange(1, T_max + 1, device=device).view(-1, 1, 1).float()
            cum_means = [p.cumsum(dim=0) / denom for p in all_probs]

            for T in T_list:
                f_mc = torch.stack([cm[T - 1] for cm in cum_means], dim=0)
                f_ref = f_mc[0]
                diff = f_mc - f_ref.unsqueeze(0)
                sq = (diff**2).sum(dim=-1)
                per_run_per_T[k][T].extend(sq.mean(dim=0).cpu().tolist())

    model.eval()
    results = {}
    for T in T_list:
        per_run_means = np.array([np.mean(per_run_per_T[k][T]) for k in range(K)], dtype=np.float64)
        results[T] = {
            "mc_mean": float(per_run_means.mean()),
            "mc_std_K": float(per_run_means.std(ddof=1)) if K > 1 else 0.0,
            "per_run_means": per_run_means.tolist(),
        }
    return results
