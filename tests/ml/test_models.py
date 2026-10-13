import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rptu_ai_toolkit import models


HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestRuntimeSelection(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        self.torch_patch = patch.object(models, "_import_torch", return_value=self.torch)
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

    def test_models_import_does_not_import_ml_dependencies(self):
        source = Path(models.__file__).resolve().parents[1]
        environment = dict(os.environ, PYTHONPATH=str(source))
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import rptu_ai_toolkit.models; "
             "assert 'torch' not in sys.modules; assert 'warpconvnet' not in sys.modules"],
            env=environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_auto_cpu_does_not_load_warp(self):
        with patch.object(models, "_load_warp") as warp:
            self.assertEqual(models.resolve_runtime(), ("cpu", "torch"))
        warp.assert_not_called()

    def test_auto_prefers_warp_on_cuda(self):
        self.torch.cuda.is_available = lambda: True
        with patch.object(models, "_load_warp"):
            self.assertEqual(models.resolve_runtime(), ("cuda", "warpconvnet"))

    def test_unavailable_warp_falls_back_only_when_backend_auto(self):
        self.torch.cuda.is_available = lambda: True
        with patch.object(models, "_load_warp", side_effect=models.MLRuntimeError("missing")):
            self.assertEqual(models.resolve_runtime(), ("cpu", "torch"))
            self.assertEqual(models.resolve_runtime("cuda"), ("cuda", "torch"))
            with self.assertRaisesRegex(models.MLRuntimeError, "missing"):
                models.resolve_runtime(backend="warpconvnet")

    def test_explicit_cpu_ignores_available_cuda(self):
        self.torch.cuda.is_available = lambda: self.fail("must not query CUDA")
        with patch.object(models, "_load_warp") as warp:
            self.assertEqual(models.resolve_runtime("cpu"), ("cpu", "torch"))
            self.assertEqual(models.resolve_runtime("cpu", "torch"), ("cpu", "torch"))
        warp.assert_not_called()

    def test_explicit_torch_can_run_on_cuda(self):
        self.torch.cuda.is_available = lambda: True
        with patch.object(models, "_load_warp") as warp:
            self.assertEqual(models.resolve_runtime(backend="torch"), ("cuda", "torch"))
        warp.assert_not_called()

    def test_explicit_unavailable_devices_and_incompatible_backend_fail(self):
        for device, backend in (("cuda", "auto"), ("auto", "warpconvnet"), ("cpu", "warpconvnet")):
            with self.subTest(device=device, backend=backend):
                with self.assertRaisesRegex(models.MLRuntimeError, "CUDA"):
                    models.resolve_runtime(device, backend)

    def test_runtime_errors_are_not_masked_as_fallback(self):
        self.torch.cuda.is_available = lambda: True
        with patch.object(models, "_load_warp", side_effect=RuntimeError("CUDA kernel fault")):
            with self.assertRaisesRegex(RuntimeError, "kernel fault"):
                models.resolve_runtime()

    def test_invalid_selection_is_rejected(self):
        for device, backend in (("mps", "auto"), ("cpu", "minkowski")):
            with self.subTest(device=device, backend=backend):
                with self.assertRaises(models.MLRuntimeError):
                    models.resolve_runtime(device, backend)

    def test_missing_torch_has_install_guidance(self):
        self.torch_patch.stop()
        with patch.object(models.importlib, "import_module", side_effect=ModuleNotFoundError("torch")):
            with self.assertRaisesRegex(models.MLRuntimeError, "ml extra"):
                models.resolve_runtime()

    def test_warp_requires_compiled_extension_and_pinned_version(self):
        packages = (
            SimpleNamespace(__version__="1.8.2", _C=None),
            SimpleNamespace(__version__="1.7.0", _C=object()),
        )
        for package in packages:
            with self.subTest(package=package):
                with patch.object(models.importlib, "import_module", return_value=package):
                    with self.assertRaises(models.MLRuntimeError):
                        models._load_warp()


@unittest.skipUnless(HAS_TORCH, "requires the ml extra")
class TestTorchModel(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(3)
        self.coords = torch.tensor([[3, -2, 0], [0, 8, 4], [10, 10, 10]], dtype=torch.int64)
        self.features = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)

    def test_model_trains_and_retains_input_order(self):
        torch = self.torch
        model = models.build_model("torch", 2, 3, "cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model(self.coords, self.features)
        self.assertEqual(before.shape, (3, 3))
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(model(self.coords[permutation], self.features[permutation]), before[permutation])
        loss = torch.nn.functional.cross_entropy(before, torch.tensor([0, 1, 2]))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model(self.coords, self.features)))

    def test_checkpoint_recreates_identical_predictions(self):
        model = models.build_model("torch", 2, 3, "cpu")
        restored = models.build_model("torch", 2, 3, "cpu")
        restored.load_state_dict(model.state_dict(), strict=True)
        self.torch.testing.assert_close(model(self.coords, self.features), restored(self.coords, self.features))

    def test_single_voxel_training_works(self):
        model = models.build_model("torch", 2, 3, "cpu")
        logits = model(self.coords[:1], self.features[:1])
        logits.sum().backward()
        self.assertEqual(logits.shape, (1, 3))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_invalid_feature_and_coordinate_contracts_fail(self):
        torch = self.torch
        model = models.build_model("torch", 2, 3, "cpu")
        cases = (
            (self.coords.to(torch.int32), self.features),
            (self.coords[:, :2], self.features),
            (self.coords[:0], self.features[:0]),
            (self.coords, self.features.to(torch.float64)),
            (self.coords, self.features[:, :1]),
            (self.coords, torch.full_like(self.features, float("inf"))),
        )
        for coords, features in cases:
            with self.subTest(coords=coords, features=features):
                with self.assertRaises(models.MLRuntimeError):
                    model(coords, features)

    def test_invalid_model_configuration_fails(self):
        for backend, channels, classes, device in (
            ("auto", 2, 3, "cpu"), ("torch", 0, 3, "cpu"),
            ("torch", 2, True, "cpu"), ("torch", 2, 3, "auto"),
        ):
            with self.subTest(backend=backend, channels=channels, classes=classes, device=device):
                with self.assertRaises(models.MLRuntimeError):
                    models.build_model(backend, channels, classes, device)


@unittest.skipUnless(HAS_TORCH, "requires the ml extra")
class TestWarpAdapterContract(unittest.TestCase):
    """CPU adapter checks with a stand-in geometry/model, without CUDA kernels."""

    def setUp(self):
        import torch
        self.torch = torch
        self.created = []
        created = self.created

        class Voxels:
            def __init__(self, batched_coordinates, batched_features, offsets):
                self.coordinate_tensor = batched_coordinates
                self.feature_tensor = batched_features
                self.offsets = offsets
                created.append(self)

        class MinkUNet18(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.linear = torch.nn.Linear(in_channels, out_channels)
                self.norm = torch.nn.BatchNorm1d(out_channels)

            def forward(self, voxels):
                order = torch.arange(len(voxels.coordinate_tensor) - 1, -1, -1)
                return SimpleNamespace(
                    coordinate_tensor=voxels.coordinate_tensor[order],
                    feature_tensor=self.norm(self.linear(voxels.feature_tensor))[order],
                )

        self.warp_patch = patch.object(models, "_load_warp", return_value=(Voxels, MinkUNet18))
        self.warp_patch.start()
        self.addCleanup(self.warp_patch.stop)
        self.model = models._build_warp_model(torch, 2, 3)
        self.coords = torch.tensor([[3, -2, 0], [0, 8, 4], [10, 10, 10]], dtype=torch.int64)
        self.features = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)

    def test_geometry_uses_integer_coordinates_and_cpu_offsets(self):
        logits = self.model(self.coords, self.features)
        self.assertEqual(logits.shape, (3, 3))
        geometry = self.created[0]
        self.assertEqual(geometry.coordinate_tensor.dtype, self.torch.int32)
        self.assertEqual(geometry.offsets.device.type, "cpu")
        self.assertEqual(geometry.offsets.tolist(), [0, 3])
        self.torch.testing.assert_close(geometry.feature_tensor, self.features)

    def test_output_reordering_preserves_correspondence_and_gradients(self):
        expected = self.model.network.norm(self.model.network.linear(self.features))
        logits = self.model(self.coords, self.features)
        self.torch.testing.assert_close(logits, expected)
        logits[0, 0].backward()
        self.assertIsNotNone(self.model.network.linear.weight.grad)

    def test_tiny_scene_keeps_norm_frozen_in_train_mode(self):
        self.model.eval()
        self.model.train()
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.network.linear.training)
        self.assertFalse(self.model.network.norm.training)
        before = self.model.network.norm.running_mean.clone()
        self.model(self.coords[:1], self.features[:1]).sum().backward()
        self.torch.testing.assert_close(self.model.network.norm.running_mean, before)
        self.assertIsNotNone(self.model.network.norm.weight.grad)

    def test_coordinate_bounds_checked_before_narrowing_or_model_execution(self):
        for invalid in (-131073, 131072, 2**40):
            with self.subTest(invalid=invalid):
                coords = self.coords.clone()
                coords[0, 0] = invalid
                with self.assertRaisesRegex(models.MLRuntimeError, "131072"):
                    self.model(coords, self.features)
        self.assertEqual(self.created, [])

    def test_coordinate_mismatch_and_duplicates_are_rejected(self):
        logits = self.torch.ones((3, 3))
        actual = self.coords.clone()
        actual[0, 0] += 1
        with self.assertRaisesRegex(models.MLRuntimeError, "different voxel set"):
            models._restore_coordinate_order(self.torch, self.coords, actual, logits)
        duplicate = self.coords.clone()
        duplicate[1] = duplicate[0]
        with self.assertRaisesRegex(models.MLRuntimeError, "unique"):
            models._restore_coordinate_order(self.torch, duplicate, duplicate, logits)
        with self.assertRaisesRegex(models.MLRuntimeError, "different voxel set"):
            models._restore_coordinate_order(self.torch, self.coords, self.coords[:2], logits[:2])

    def test_network_failure_propagates_without_fallback(self):
        with patch.object(self.model.network, "forward", side_effect=RuntimeError("CUDA fault")):
            with self.assertRaisesRegex(RuntimeError, "CUDA fault"):
                self.model(self.coords, self.features)


if __name__ == "__main__":
    unittest.main()
