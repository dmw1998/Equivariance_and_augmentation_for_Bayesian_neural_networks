"""Theorem-4 verification: train a baseline Bayesian CNN, then estimate the equivariance defect as a function of the number of posterior MC samples T.

Two sweeps are produced:

* a single-realisation T-sweep (held-out and training equiv sets), and
* a K-fold MC sweep that re-evaluates the same trained model K times to estimate the per-realisation MC deviation std at each T (predicted to scale as O(1/sqrt(T))).

Example:
    python scripts/train_theorem4.py --dataset FashionMNIST --train_size 5000 --epochs 500 --eval_samples 1024 --K_mc_runs 10 --skip_single_sweep
"""

import argparse

import numpy as np
import torch
import torchvision
import wandb
from torch.utils.data import DataLoader

from ebnn.data import EquivarianceEvalSet, get_base_transform, get_loaders
from ebnn.metrics import compute_equivariance_defect_K_fold, compute_equivariance_defect_T_sweep
from ebnn.models import BayesianCNN
from ebnn.training import evaluate_mc_accuracy, train_baseline_bnn

_DATASETS = {
    "MNIST": torchvision.datasets.MNIST,
    "FashionMNIST": torchvision.datasets.FashionMNIST,
    "CIFAR10": torchvision.datasets.CIFAR10,
}


def _build_train_equiv_loader(dataset_name, train_size, seed, batch_size):
    """Build a separate equiv-format loader on the same N_0 training samples.

    The standard equiv_loader (from the held-out test split) gives a proxy for
    Delta^eq; this loader gives the empirical defect \\hat{Delta}^eq on the
    training set.  A fixed local generator samples uniformly from the train
    split -- the resulting estimate is still over N_0 i.i.d. training-like
    samples, which is what the theorem requires.
    """
    transform = get_base_transform(dataset_name)
    if dataset_name not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    base_train = _DATASETS[dataset_name](root="./data", train=True, download=True, transform=transform)

    g = torch.Generator().manual_seed(seed)
    n = min(train_size, len(base_train))
    indices = torch.randperm(len(base_train), generator=g)[:n].tolist()
    subset = torch.utils.data.Subset(base_train, indices)

    eq_set = EquivarianceEvalSet(subset, n_samples=n, cache_path=None)
    return DataLoader(eq_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)


def main():
    parser = argparse.ArgumentParser(description="Theorem 4 verification: MC sample-complexity of the defect")
    parser.add_argument(
        "--dataset", type=str, default="FashionMNIST", choices=["MNIST", "FashionMNIST", "CIFAR10"]
    )
    parser.add_argument(
        "--train_size",
        type=int,
        required=True,
        help="N_0: number of training samples (before C4 augmentation)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--conv_channels", type=int, nargs="+", default=None)
    parser.add_argument("--prior_std", type=float, default=1.0)
    parser.add_argument("--rho_init", type=float, default=-3.0)
    parser.add_argument("--random_prior", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--train_samples", type=int, default=10)
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=1024,
        help="T_max: number of posterior samples for the largest T in the sweep.",
    )
    parser.add_argument(
        "--T_list",
        type=int,
        nargs="+",
        default=None,
        help="List of T values to evaluate. Defaults to powers of 2 up to --eval_samples.",
    )
    parser.add_argument(
        "--K_mc_runs",
        type=int,
        default=10,
        help="Independent MC re-evaluations to estimate the MC deviation std at each T. 0 to skip.",
    )
    parser.add_argument(
        "--K_eval_split",
        type=str,
        default="test",
        choices=["test", "train", "both"],
        help="Which equiv set(s) to run the K-fold sweep on.",
    )
    parser.add_argument(
        "--skip_single_sweep",
        action="store_true",
        help="Skip the single-realisation T-sweep (the K-fold mc_mean estimates the same quantity).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="theorem4_verification")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    if args.T_list is None:
        T_list = []
        T = 1
        while T <= args.eval_samples:
            T_list.append(T)
            T *= 2
        if T_list[-1] != args.eval_samples:
            T_list.append(args.eval_samples)
        args.T_list = T_list

    print("=" * 50)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 50)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, equiv_loader = get_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train_size=args.train_size,
        use_augmentation=True,
    )
    print(
        f"Train samples (after aug): {len(train_loader.dataset)}, "
        f"Test: {len(test_loader.dataset)}, Equiv (test): {len(equiv_loader.dataset)}"
    )

    train_equiv_loader = _build_train_equiv_loader(
        dataset_name=args.dataset,
        train_size=args.train_size,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(f"Equiv (train, N_0): {len(train_equiv_loader.dataset)}")

    sample_input = next(iter(train_loader))[0]
    in_channels = sample_input.shape[1]
    input_size = (sample_input.shape[2], sample_input.shape[3])

    if args.conv_channels is None:
        args.conv_channels = [64, 128] if args.dataset == "CIFAR10" else [32, 64]

    model = BayesianCNN(
        in_channels=in_channels,
        num_classes=10,
        conv_channels=args.conv_channels,
        input_size=input_size,
        prior_std=args.prior_std,
        rho_init=args.rho_init,
        random_prior=args.random_prior,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    use_wandb = not args.no_wandb
    if use_wandb:
        if args.run_name is None:
            prior_tag = "rndprior" if args.random_prior else "isoprior"
            args.run_name = f"{args.dataset}_n{args.train_size}_{prior_tag}_seed{args.seed}"
        wandb.init(project=args.project, name=args.run_name, config=vars(args))

    train_baseline_bnn(
        model=model,
        train_loader=train_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        train_samples=args.train_samples,
        weight_decay=args.weight_decay,
        use_wandb=use_wandb,
    )

    test_acc = evaluate_mc_accuracy(model, test_loader, device, mc_samples=min(64, args.eval_samples))
    print(f"Test accuracy: {test_acc:.2f}%")

    # Rebuild equiv loaders for evaluation only (batch size does not affect the
    # per-sample defect; the model has no BatchNorm).
    eval_batch_size = 128
    equiv_loader = DataLoader(
        equiv_loader.dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    train_equiv_loader = DataLoader(
        train_equiv_loader.dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    if not args.skip_single_sweep:
        print(f"\nRunning T-sweep on held-out equiv set (T_list={args.T_list})...")
        delta_eq_sweep = compute_equivariance_defect_T_sweep(model, device, equiv_loader, T_list=args.T_list)
        print(f"Running T-sweep on training equiv set (N_0={args.train_size})...")
        hat_delta_eq_sweep = compute_equivariance_defect_T_sweep(
            model, device, train_equiv_loader, T_list=args.T_list
        )

        T_star = max(args.T_list)
        delta_eq_proxy = delta_eq_sweep[T_star]["mean"]
        hat_delta_eq_proxy = hat_delta_eq_sweep[T_star]["mean"]

        print("\n=== Theorem 4 quantities ===")
        print(f"Delta^eq proxy        (T={T_star}, held-out): {delta_eq_proxy:.6f}")
        print(f"hat Delta^eq proxy    (T={T_star}, train N_0): {hat_delta_eq_proxy:.6f}")
        print(f"Generalization gap (Delta - hat Delta): {delta_eq_proxy - hat_delta_eq_proxy:.6f}")
        print("\nPer-T values (held-out):")
        for T in args.T_list:
            r = delta_eq_sweep[T]
            print(
                f"  T={T:5d}: mean={r['mean']:.6f}, std={r['std']:.6f}, "
                f"|mean - proxy|={abs(r['mean'] - delta_eq_proxy):.6f}"
            )

        if use_wandb:
            for T in args.T_list:
                r_test = delta_eq_sweep[T]
                r_train = hat_delta_eq_sweep[T]
                wandb.log(
                    {
                        "sweep/T": T,
                        "sweep/delta_eq_test_mean": r_test["mean"],
                        "sweep/delta_eq_test_std": r_test["std"],
                        "sweep/delta_eq_test_gap_to_proxy": abs(r_test["mean"] - delta_eq_proxy),
                        "sweep/hat_delta_eq_train_mean": r_train["mean"],
                        "sweep/hat_delta_eq_train_std": r_train["std"],
                        "sweep/hat_delta_eq_train_gap_to_proxy": abs(r_train["mean"] - hat_delta_eq_proxy),
                        "sweep/gen_gap": r_test["mean"] - r_train["mean"],
                    }
                )
            wandb.log(
                {
                    "final/test_accuracy": test_acc,
                    "final/delta_eq_proxy": delta_eq_proxy,
                    "final/hat_delta_eq_proxy": hat_delta_eq_proxy,
                    "final/gen_gap_proxy": delta_eq_proxy - hat_delta_eq_proxy,
                    "final/train_size": args.train_size,
                    "final/T_max": T_star,
                    "final/random_prior": args.random_prior,
                }
            )
            for T in args.T_list:
                wandb.summary[f"delta_eq_T{T}"] = delta_eq_sweep[T]["mean"]
                wandb.summary[f"hat_delta_eq_T{T}"] = hat_delta_eq_sweep[T]["mean"]
    else:
        print("\nSkipping single-realisation T-sweep (--skip_single_sweep).")
        if use_wandb:
            wandb.log(
                {
                    "final/test_accuracy": test_acc,
                    "final/train_size": args.train_size,
                    "final/T_max": max(args.T_list),
                    "final/random_prior": args.random_prior,
                }
            )

    # --- K-fold MC deviation sweep ---
    if args.K_mc_runs > 0:
        if args.K_eval_split in ("test", "both"):
            print(f"\nRunning K={args.K_mc_runs}-fold MC sweep on held-out equiv set...")
            mc_test = compute_equivariance_defect_K_fold(
                model, device, equiv_loader, T_list=args.T_list, K=args.K_mc_runs
            )
            T_star = max(args.T_list)
            test_proxy = mc_test[T_star]["mc_mean"]
            print("=== K-fold MC sweep (held-out) ===")
            for T in args.T_list:
                r = mc_test[T]
                print(
                    f"  T={T:5d}: mc_mean={r['mc_mean']:.6e}  mc_std_K={r['mc_std_K']:.6e}  "
                    f"|mc_mean - proxy|={abs(r['mc_mean'] - test_proxy):.6e}"
                )
            if use_wandb:
                for T in args.T_list:
                    r = mc_test[T]
                    wandb.log(
                        {
                            "mc_std_sweep/T": T,
                            "mc_std_sweep/test_mc_mean": r["mc_mean"],
                            "mc_std_sweep/test_mc_std_K": r["mc_std_K"],
                            "mc_std_sweep/test_gap_to_proxy": abs(r["mc_mean"] - test_proxy),
                        }
                    )
                    wandb.summary[f"mc_test_mean_T{T}"] = r["mc_mean"]
                    wandb.summary[f"mc_test_std_K_T{T}"] = r["mc_std_K"]
                wandb.summary["mc_test_proxy"] = test_proxy

        if args.K_eval_split in ("train", "both"):
            print(
                f"\nRunning K={args.K_mc_runs}-fold MC sweep on training equiv set (N_0={args.train_size})..."
            )
            mc_train = compute_equivariance_defect_K_fold(
                model, device, train_equiv_loader, T_list=args.T_list, K=args.K_mc_runs
            )
            T_star = max(args.T_list)
            train_proxy = mc_train[T_star]["mc_mean"]
            print("=== K-fold MC sweep (train) ===")
            for T in args.T_list:
                r = mc_train[T]
                print(
                    f"  T={T:5d}: mc_mean={r['mc_mean']:.6e}  mc_std_K={r['mc_std_K']:.6e}  "
                    f"|mc_mean - proxy|={abs(r['mc_mean'] - train_proxy):.6e}"
                )
            if use_wandb:
                for T in args.T_list:
                    r = mc_train[T]
                    wandb.log(
                        {
                            "mc_std_sweep/T": T,
                            "mc_std_sweep/train_mc_mean": r["mc_mean"],
                            "mc_std_sweep/train_mc_std_K": r["mc_std_K"],
                            "mc_std_sweep/train_gap_to_proxy": abs(r["mc_mean"] - train_proxy),
                        }
                    )
                    wandb.summary[f"mc_train_mean_T{T}"] = r["mc_mean"]
                    wandb.summary[f"mc_train_std_K_T{T}"] = r["mc_std_K"]
                wandb.summary["mc_train_proxy"] = train_proxy

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
