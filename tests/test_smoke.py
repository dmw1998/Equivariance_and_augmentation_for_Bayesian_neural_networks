"""CPU smoke tests for the ``ebnn`` package.

These do NOT reproduce paper numbers. They check that the refactored code is internally consistent: every model imports, builds, and runs a forward/backward pass on random tensors; the C4 projection is idempotent; the GCNN expansion runs; and the optimiser builders work for both AdamW and SGD.

Run with:  python -m pytest tests/test_smoke.py   (or)   python tests/test_smoke.py
"""

import sys
import types

# Allow the package (whose __init__ imports ebnn.data -> torchvision) to import
# even when torchvision is absent; the data path is not exercised here.
try:
    import torchvision  # noqa: F401
except Exception:  # pragma: no cover - test-environment shim only
    stub = types.ModuleType("torchvision")
    stub.datasets = types.SimpleNamespace(MNIST=object, FashionMNIST=object, CIFAR10=object, CIFAR100=object)
    transforms = types.ModuleType("torchvision.transforms")

    class _T:
        @staticmethod
        def Compose(*a, **k):
            return None

        @staticmethod
        def ToTensor(*a, **k):
            return None

        @staticmethod
        def Normalize(*a, **k):
            return None

    transforms.Compose = _T.Compose
    transforms.ToTensor = _T.ToTensor
    transforms.Normalize = _T.Normalize
    transforms.functional = types.SimpleNamespace(rotate=lambda img, angle: img)
    stub.transforms = transforms
    sys.modules["torchvision"] = stub
    sys.modules["torchvision.transforms"] = transforms

import torch  # noqa: E402

import ebnn  # noqa: E402
from ebnn.models import (  # noqa: E402
    BayesianCNN,
    BayesianCNNConfigurable,
    BayesianFullyConvCNN,
    BayesianGroupConvCNN,
)
from ebnn.symmetrize import (  # noqa: E402
    apply_one_shot_c4_posterior_average,
    apply_one_shot_c4_projection,
    build_optimizer,
    expand_small_model_to_gcnn,
    initialize_gaussian_posterior_scale,
    initialize_posterior_mean,
)

torch.manual_seed(0)


def _forward_backward(model, x, family="gaussian"):
    out = model(x, return_kl=True)
    logits, kl = out[0], out[-1]
    loss = logits.float().pow(2).mean() + kl
    loss.backward()
    assert torch.isfinite(loss), "loss is not finite"
    return logits


def test_theorem_cnn():
    model = BayesianCNN(1, 10, conv_channels=[8, 16], input_size=(28, 28))
    logits = _forward_backward(model, torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 10)


def test_configurable_families():
    for family in ("gaussian", "laplace", "lognormal"):
        model = BayesianCNNConfigurable(
            in_channels=1,
            num_classes=10,
            family=family,
            conv_channels=[8, 16],
            input_size=(28, 28),
        )
        logits = _forward_backward(model, torch.randn(4, 1, 28, 28))
        assert logits.shape == (4, 10), family


def test_fullyconv_forward():
    model = BayesianFullyConvCNN(1, 10, conv_channels=[8, 16])
    logits = _forward_backward(model, torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 10)


def test_projection_is_idempotent():
    # Channel counts must be divisible by 4 for the C4 projection.
    model = BayesianFullyConvCNN(1, 12, conv_channels=[8, 16])
    apply_one_shot_c4_projection(model)
    w1 = model.conv_layers[1].weight_mu.detach().clone()
    apply_one_shot_c4_projection(model)
    w2 = model.conv_layers[1].weight_mu.detach().clone()
    assert torch.allclose(w1, w2, atol=1e-5), "projection should be idempotent on the mean"


def test_posterior_average_runs():
    model = BayesianFullyConvCNN(1, 10, conv_channels=[8, 16])
    for method in ("geometric", "arithmetic"):
        apply_one_shot_c4_posterior_average(model, method=method)
    logits = model(torch.randn(2, 1, 28, 28), return_kl=False)
    assert logits.shape == (2, 10)


def test_gcnn_expansion_runs():
    small = BayesianFullyConvCNN(1, 10, conv_channels=[8, 16])
    initialize_posterior_mean(small, posterior_init="kaiming")
    initialize_gaussian_posterior_scale(small, rho_init=-3.0)
    gcnn = BayesianGroupConvCNN(1, 10, base_channels=[8, 16], group_size=4)
    expand_small_model_to_gcnn(small, gcnn, group_size=4)
    logits = gcnn(torch.randn(2, 1, 28, 28), return_kl=False)
    assert logits.shape == (2, 10)


def test_optimizer_builders():
    model = BayesianFullyConvCNN(1, 10, conv_channels=[8, 16])
    adamw = build_optimizer(model, optimizer="adamw", lr=1e-3)
    sgd = build_optimizer(model, optimizer="sgd", lr=1e-2)
    assert len(adamw.param_groups) == 1
    assert len(sgd.param_groups) == 2  # base + rho groups


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed. Public API exports: {len(ebnn.__all__)} names.")


if __name__ == "__main__":
    _run_all()
