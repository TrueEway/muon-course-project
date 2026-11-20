import unittest

import numpy as np
import torch

from experiments.language import TinyTransformer, batch_from_starts, make_optimizers, parameter_groups
from experiments.optim import MatrixMuon, ns_numpy, ns_torch, polar_numpy, polar_torch
from experiments.synthetic import problem, value_grad


class PolarAndDescentTests(unittest.TestCase):
    def test_polar_duality_including_rank_deficiency(self):
        rng = np.random.default_rng(7)
        for m, n in [(8, 4), (4, 8), (4, 4)]:
            for rank in [0, 1, min(m, n)]:
                matrix = rng.normal(size=(m, rank)) @ rng.normal(size=(rank, n))
                q = polar_numpy(matrix)
                self.assertAlmostEqual(float(np.sum(matrix * q)), float(np.linalg.svd(matrix, compute_uv=False).sum()), places=10)
                self.assertLessEqual(np.linalg.norm(q, 2), 1 + 1e-12)
                self.assertLessEqual(np.linalg.norm(q) ** 2, min(m, n) + 1e-10)
                np.testing.assert_allclose(q, polar_torch(torch.tensor(matrix)).numpy(), atol=1e-11)

    def test_stochastic_descent_with_tracking_error(self):
        h, target, w = problem(4, 10)
        rng = np.random.default_rng(3)
        eta = 0.001
        for _ in range(40):
            loss, gradient = value_grad(w, target, h)
            momentum = gradient + 0.02 * rng.normal(size=w.shape)
            q = polar_numpy(momentum)
            next_w = w - eta * q
            new_loss, _ = value_grad(next_w, target, h)
            nuclear = np.linalg.svd(gradient, compute_uv=False).sum()
            rhs = loss - eta * nuclear + 2 * eta * np.sqrt(16) * np.linalg.norm(momentum - gradient) + eta ** 2 * 16 / 2
            self.assertLessEqual(new_loss, rhs + 1e-12)
            w = next_w

    def test_objective_gradient_finite_difference(self):
        h, target, w = problem(2, 100)
        _, gradient = value_grad(w, target, h)
        direction = np.random.default_rng(1).normal(size=w.shape)
        epsilon = 1e-6
        finite = (value_grad(w + epsilon * direction, target, h)[0] - value_grad(w - epsilon * direction, target, h)[0]) / (2 * epsilon)
        self.assertAlmostEqual(finite, float(np.sum(gradient * direction)), places=8)

    def test_finite_ns_matches_scalar_polynomial_and_zero(self):
        for steps in [1, 3, 5, 10]:
            diag = np.array([1.0, 0.1, 1e-5, 0.0])
            matrix = np.diag(diag)
            x = diag / (np.linalg.norm(diag) + 1e-7)
            for _ in range(steps):
                x = 3.4445 * x - 4.7750 * x ** 3 + 2.0315 * x ** 5
            np.testing.assert_allclose(ns_numpy(matrix, steps), np.diag(x), atol=1e-12)
        np.testing.assert_array_equal(ns_numpy(np.zeros((3, 7))), np.zeros((3, 7)))
        matrix = np.random.default_rng(1).normal(size=(7, 3)).astype(np.float32)
        np.testing.assert_allclose(ns_numpy(matrix), ns_torch(torch.from_numpy(matrix)).numpy(), atol=2e-5, rtol=2e-5)

    def test_reference_update_no_gradient_mutation(self):
        param = torch.nn.Parameter(torch.eye(3))
        gradient = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        param.grad = gradient.clone()
        optimizer = MatrixMuon([param], lr=0.02, mode='svd', weight_decay=0.01)
        expected = torch.eye(3) * (1 - 0.02 * 0.01) - 0.02 * polar_torch(gradient)
        optimizer.step()
        torch.testing.assert_close(param.grad, gradient)
        torch.testing.assert_close(param, expected)
        torch.testing.assert_close(optimizer.state[param]['momentum'], gradient * 0.05)


class ModelTests(unittest.TestCase):
    def test_parameter_groups_are_disjoint_complete(self):
        model = TinyTransformer(65)
        for method in ['adamw', 'muon_svd', 'muon_ns']:
            muon, decay, nodecay, names = parameter_groups(model, method)
            ids = [id(p) for p in muon + decay + nodecay]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in model.parameters()})
            self.assertFalse(any('embedding' in name or 'head' in name for name in names['muon']))
            if method != 'adamw':
                self.assertEqual(len(muon), 8)

    def test_causal_attention_does_not_see_future(self):
        torch.manual_seed(1)
        model = TinyTransformer(65).eval()
        tokens = torch.randint(0, 65, (2, 16))
        changed = tokens.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 65
        with torch.no_grad():
            before, _ = model(tokens)
            after, _ = model(changed)
        torch.testing.assert_close(before[:, :8], after[:, :8])

    def test_repeatable_training_step(self):
        outputs = []
        for _ in range(2):
            torch.manual_seed(3)
            model = TinyTransformer(65)
            optimizers, _ = make_optimizers(model, 'muon_ns', 0.02, 0.001)
            tokens = torch.arange(400) % 65
            x, y = batch_from_starts(tokens, torch.tensor([0, 70]))
            _, loss = model(x, y)
            loss.backward()
            for optimizer, _ in optimizers:
                optimizer.step()
            outputs.append(model.head.weight.detach().clone())
        torch.testing.assert_close(*outputs, rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
