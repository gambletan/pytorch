"""Torchlite training end-to-end tests.

Tests the full torchlite.compile() pipeline (trace -> run_passes -> codegen)
by comparing loss values and parameter updates against eager PyTorch training
for several model architectures and optimizer types (SGD, AdamW).
"""

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase
from test_torchlite_utils import TrainStep, SimpleLinear, TwoLayerMLP, ThreeLayerSinCos
from torch._torchlite import compile


class TestTorchliteTraining(TestCase):
    def _train_eager_sgd(self, model, x, target, lr, num_steps):
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        losses = []
        for _ in range(num_steps):
            opt.zero_grad()
            out = model(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    def _train_eager_adamw(self, model, x, target, lr, num_steps,
                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr,
            betas=betas, eps=eps, weight_decay=weight_decay,
        )
        losses = []
        for _ in range(num_steps):
            opt.zero_grad()
            out = model(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    def _train_torchlite(self, model, x, target, lr, num_steps,
                         optimizer_type="sgd"):
        train_step = TrainStep(model)
        compiled = compile(
            train_step, [x, target],
            lr=lr, optimizer_type=optimizer_type,
        )
        losses = []
        for _ in range(num_steps):
            loss = compiled(x, target)
            losses.append(loss.item())
        return losses

    def _make_pair(self, model_fn):
        torch.manual_seed(42)
        model_eager = model_fn()
        torch.manual_seed(42)
        model_torchlite = model_fn()
        return model_eager, model_torchlite

    def _compare_sgd(self, model_fn, in_dim, out_dim,
                     lr=0.01, num_steps=5, batch=4,
                     atol=1e-5, rtol=1e-5):
        model_eager, model_torchlite = self._make_pair(model_fn)
        x = torch.randn(batch, in_dim)
        target = torch.randn(batch, out_dim)

        eager_losses = self._train_eager_sgd(
            model_eager, x, target, lr, num_steps,
        )
        torchlite_losses = self._train_torchlite(
            model_torchlite, x, target, lr, num_steps, optimizer_type="sgd",
        )

        for step, (el, tl) in enumerate(zip(eager_losses, torchlite_losses)):
            self.assertAlmostEqual(
                el, tl, places=4,
                msg=f"SGD loss mismatch at step {step}",
            )

        for i, (pe, pt) in enumerate(zip(
            model_eager.parameters(), model_torchlite.parameters(),
        )):
            self.assertEqual(
                pe, pt, atol=atol, rtol=rtol,
                msg=f"SGD param {i} mismatch after {num_steps} steps",
            )

    def _compare_adamw(self, model_fn, in_dim, out_dim,
                       lr=0.001, num_steps=5, batch=4,
                       atol=1e-4, rtol=1e-4):
        model_eager, model_torchlite = self._make_pair(model_fn)
        x = torch.randn(batch, in_dim)
        target = torch.randn(batch, out_dim)

        eager_losses = self._train_eager_adamw(
            model_eager, x, target, lr, num_steps,
        )
        torchlite_losses = self._train_torchlite(
            model_torchlite, x, target, lr, num_steps, optimizer_type="adamw",
        )

        for step, (el, tl) in enumerate(zip(eager_losses, torchlite_losses)):
            self.assertAlmostEqual(
                el, tl, places=3,
                msg=f"AdamW loss mismatch at step {step}",
            )

        for i, (pe, pt) in enumerate(zip(
            model_eager.parameters(), model_torchlite.parameters(),
        )):
            self.assertEqual(
                pe, pt, atol=atol, rtol=rtol,
                msg=f"AdamW param {i} mismatch after {num_steps} steps",
            )

    # ──────────────────────────────────────────────────────────────────
    # SGD
    # ──────────────────────────────────────────────────────────────────

    def test_simple_linear_sgd(self):
        self._compare_sgd(lambda: SimpleLinear(10, 5), 10, 5)

    def test_two_layer_mlp_sgd(self):
        self._compare_sgd(lambda: TwoLayerMLP(10, 20, 5), 10, 5)

    def test_three_layer_sincos_sgd(self):
        self._compare_sgd(
            lambda: ThreeLayerSinCos(10, 20, 5), 10, 5,
            lr=0.001, num_steps=3,
        )

    def test_sgd_multi_step(self):
        self._compare_sgd(
            lambda: TwoLayerMLP(10, 20, 5), 10, 5,
            lr=0.01, num_steps=10,
        )

    # ──────────────────────────────────────────────────────────────────
    # AdamW
    # ──────────────────────────────────────────────────────────────────

    def test_simple_linear_adamw(self):
        self._compare_adamw(lambda: SimpleLinear(10, 5), 10, 5)

    def test_two_layer_mlp_adamw(self):
        self._compare_adamw(lambda: TwoLayerMLP(10, 20, 5), 10, 5)

    def test_three_layer_sincos_adamw(self):
        self._compare_adamw(
            lambda: ThreeLayerSinCos(10, 20, 5), 10, 5,
            lr=0.001, num_steps=3,
        )

    # ──────────────────────────────────────────────────────────────────
    # Sanity checks
    # ──────────────────────────────────────────────────────────────────

    def test_loss_decreases(self):
        torch.manual_seed(42)
        model = TwoLayerMLP(10, 20, 5)
        x = torch.randn(8, 10)
        target = torch.randn(8, 5)
        losses = self._train_torchlite(model, x, target, lr=0.01, num_steps=20)
        self.assertLess(losses[-1], losses[0])

    def test_parameters_change_after_one_step(self):
        torch.manual_seed(42)
        model = SimpleLinear(10, 5)
        initial_params = [p.clone() for p in model.parameters()]
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        self._train_torchlite(model, x, target, lr=0.01, num_steps=1)
        for p_init, p_curr in zip(initial_params, model.parameters()):
            self.assertFalse(torch.equal(p_init, p_curr))

    def test_larger_batch(self):
        self._compare_sgd(
            lambda: TwoLayerMLP(10, 20, 5), 10, 5,
            batch=32, num_steps=3,
        )

    def test_single_step_sgd(self):
        self._compare_sgd(
            lambda: SimpleLinear(10, 5), 10, 5,
            num_steps=1,
        )

    def test_single_step_adamw(self):
        self._compare_adamw(
            lambda: SimpleLinear(10, 5), 10, 5,
            num_steps=1,
        )


if __name__ == "__main__":
    run_tests()
