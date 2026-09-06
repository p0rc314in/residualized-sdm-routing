from __future__ import annotations

import unittest

try:
    import torch
    from shared_residual_routing.router import ResidualizedRouter
except ImportError:
    torch = None
    ResidualizedRouter = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RouterTests(unittest.TestCase):
    def test_zero_residuals_begin_at_the_native_mean(self) -> None:
        assert torch is not None and ResidualizedRouter is not None
        torch.manual_seed(7)
        read = torch.nn.Linear(5, 6, bias=True)
        write = torch.nn.Linear(5, 6, bias=True)
        expected_weight = 0.5 * (read.weight.detach() + write.weight.detach())
        expected_bias = 0.5 * (read.bias.detach() + write.bias.detach())
        router = ResidualizedRouter(read, write)
        x = torch.randn(3, 5)

        self.assertTrue(torch.equal(router.shared_weight, expected_weight))
        self.assertTrue(torch.equal(router.shared_bias, expected_bias))
        self.assertTrue(torch.equal(read(x), write(x)))
        self.assertTrue(torch.equal(read(x), torch.nn.functional.linear(x, expected_weight, expected_bias)))
        self.assertFalse(read.weight.requires_grad)
        self.assertFalse(write.weight.requires_grad)

        native_trainable = 2 * (read.weight.numel() + read.bias.numel())
        routed_trainable = sum(p.numel() for p in router.parameters() if p.requires_grad)
        self.assertEqual(routed_trainable - native_trainable, read.weight.numel() + read.bias.numel())

    def test_role_residuals_specialize_independently(self) -> None:
        assert torch is not None and ResidualizedRouter is not None
        read = torch.nn.Linear(4, 4, bias=True)
        write = torch.nn.Linear(4, 4, bias=True)
        router = ResidualizedRouter(read, write)
        with torch.no_grad():
            router.read_residual_bias.add_(1.0)
        x = torch.zeros(2, 4)
        self.assertTrue(torch.equal(read(x), write(x) + 1.0))


if __name__ == "__main__":
    unittest.main()
