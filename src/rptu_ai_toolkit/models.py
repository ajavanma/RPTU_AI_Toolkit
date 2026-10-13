"""Lazy Torch and WarpConvNet models for prepared voxel scenes.

The Warp adapter targets WarpConvNet 1.8.2's native ``Voxels``/``MinkUNet18``
API. BatchNorm uses fixed running statistics, with trainable affine parameters,
so training a scene with a single voxel at the deepest resolution is valid.
Torch and CUDA are imported only when an ML runtime is requested.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import sys
from typing import Any


class MLRuntimeError(ValueError):
    """The requested ML runtime or its inputs cannot be used."""


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise MLRuntimeError(
            "PyTorch is unavailable. Install the toolkit's ml extra to train or predict."
        ) from exc


def _load_warp() -> tuple[Any, Any]:
    # Upstream prints its version on import. Keep command JSON stdout clean.
    try:
        with redirect_stdout(sys.stderr):
            package = importlib.import_module("warpconvnet")
            if package.__version__.split("+", 1)[0] != "1.8.2":
                raise MLRuntimeError("The WarpConvNet backend requires WarpConvNet 1.8.2.")
            if getattr(package, "_C", None) is None:
                raise MLRuntimeError("WarpConvNet's compiled CUDA extension is unavailable.")
            voxels = importlib.import_module("warpconvnet.geometry.types.voxels").Voxels
            mink_unet = importlib.import_module("warpconvnet.models.mink_unet").MinkUNet18
        return voxels, mink_unet
    except (ImportError, OSError, AttributeError) as exc:
        raise MLRuntimeError(
            "WarpConvNet is unavailable. Install the pinned GPU dependencies and the "
            "WarpConvNet 1.8.2 wheel matching PyTorch and CUDA. "
            f"Import failed: {exc}"
        ) from exc


def resolve_runtime(device: str = "auto", backend: str = "auto") -> tuple[str, str]:
    """Choose (device, backend) before constructing or executing a model.

    Automatic selection prefers WarpConvNet on CUDA, otherwise Torch on CPU.
    An explicit CUDA device remains CUDA when only the backend is automatic.
    An explicit backend is never substituted; in particular a Warp checkpoint
    cannot be loaded into the CPU MLP. Execution failures never trigger fallback.
    """
    if device not in {"auto", "cpu", "cuda"}:
        raise MLRuntimeError("device must be auto, cpu, or cuda")
    if backend not in {"auto", "torch", "warpconvnet"}:
        raise MLRuntimeError("backend must be auto, torch, or warpconvnet")
    if device == "cpu" and backend == "warpconvnet":
        raise MLRuntimeError("WarpConvNet requires CUDA; its checkpoints cannot run on CPU.")
    torch = _import_torch()
    cuda_available = torch.cuda.is_available() if device != "cpu" else False
    if device == "cuda" and not cuda_available:
        raise MLRuntimeError("CUDA was requested but is unavailable in this PyTorch runtime.")
    if backend == "warpconvnet":
        if not cuda_available:
            raise MLRuntimeError("WarpConvNet requires an available CUDA device.")
        _load_warp()
        return "cuda", "warpconvnet"
    if backend == "torch":
        return ("cuda" if cuda_available else "cpu"), "torch"
    if cuda_available:
        try:
            _load_warp()
        except MLRuntimeError:
            return ("cuda" if device == "cuda" else "cpu"), "torch"
        return "cuda", "warpconvnet"
    return "cpu", "torch"


def _validate_inputs(torch: Any, coords: Any, features: Any, in_channels: int) -> None:
    if not isinstance(coords, torch.Tensor) or coords.dtype != torch.int64:
        raise MLRuntimeError("voxel coordinates must be an int64 tensor")
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise MLRuntimeError("voxel coordinates must have nonempty shape [N, 3]")
    if not isinstance(features, torch.Tensor) or features.dtype != torch.float32:
        raise MLRuntimeError("voxel features must be a float32 tensor")
    if features.shape != (coords.shape[0], in_channels):
        raise MLRuntimeError(f"voxel features must have shape [N, {in_channels}]")
    if features.device != coords.device:
        raise MLRuntimeError("voxel coordinates and features must be on the same device")
    if not bool(torch.isfinite(features).all()):
        raise MLRuntimeError("voxel features must be finite")


def _lexicographic_order(torch: Any, coords: Any) -> Any:
    order = torch.arange(coords.shape[0], device=coords.device)
    for axis in (2, 1, 0):
        order = order[torch.argsort(coords[order, axis], stable=True)]
    return order


def _restore_coordinate_order(torch: Any, expected: Any, actual: Any, logits: Any) -> Any:
    """Match integer triples exactly, without an overflow-prone flattened key."""
    if actual.shape != expected.shape or logits.ndim != 2 or len(logits) != len(expected):
        raise MLRuntimeError("WarpConvNet returned a different voxel set from its input.")
    if actual.device != expected.device or logits.device != expected.device:
        raise MLRuntimeError("WarpConvNet returned coordinates or logits on the wrong device.")
    if actual.dtype not in (torch.int32, torch.int64):
        raise MLRuntimeError("WarpConvNet returned noninteger voxel coordinates.")
    expected_order = _lexicographic_order(torch, expected)
    sorted_expected = expected[expected_order]
    if bool((sorted_expected[1:] == sorted_expected[:-1]).all(dim=1).any()):
        raise MLRuntimeError("voxel coordinates must be unique")
    if torch.equal(expected, actual):
        return logits
    actual_order = _lexicographic_order(torch, actual)
    if not torch.equal(sorted_expected, actual[actual_order]):
        raise MLRuntimeError("WarpConvNet returned a different voxel set from its input.")
    output_rows = torch.empty_like(expected_order)
    output_rows[expected_order] = actual_order
    return logits[output_rows]


def _build_torch_model(torch: Any, in_channels: int, num_classes: int) -> Any:
    class VoxelMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(in_channels, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, num_classes),
            )

        def forward(self, voxel_coords: Any, features: Any) -> Any:
            _validate_inputs(torch, voxel_coords, features, in_channels)
            return self.network(features)

    return VoxelMLP()


def _build_warp_model(torch: Any, in_channels: int, num_classes: int) -> Any:
    Voxels, MinkUNet18 = _load_warp()

    class WarpVoxelUNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = MinkUNet18(in_channels=in_channels, out_channels=num_classes)
            self.train()

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            # A whole scene is one sparse sample; deeper levels may contain only
            # one voxel. Keep running statistics fixed even after model.train().
            for module in self.network.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.eval()
            return self

        def forward(self, voxel_coords: Any, features: Any) -> Any:
            _validate_inputs(torch, voxel_coords, features, in_channels)
            # v1.8.2's PackedHashTable stores signed 18-bit spatial coordinates.
            # Check before int32 conversion so large int64 values cannot wrap.
            if bool(((voxel_coords < -131072) | (voxel_coords > 131071)).any()):
                raise MLRuntimeError(
                    "WarpConvNet voxel coordinates must be within [-131072, 131071]; "
                    "recenter the source coordinates or increase the voxel size."
                )
            coords = voxel_coords.to(dtype=torch.int32).contiguous()
            voxels = Voxels(
                batched_coordinates=coords,
                batched_features=features.contiguous(),
                offsets=torch.tensor([0, len(coords)], dtype=torch.int64, device="cpu"),
            )
            result = self.network(voxels)
            logits = _restore_coordinate_order(
                torch, voxel_coords, result.coordinate_tensor, result.feature_tensor,
            )
            if logits.shape[1] != num_classes:
                raise MLRuntimeError("WarpConvNet returned an unexpected number of classes.")
            return logits

    return WarpVoxelUNet()


def build_model(backend: str, in_channels: int, num_classes: int, device: str) -> Any:
    """Construct a model with logits aligned to its prepared input coordinate order."""
    if type(in_channels) is not int or in_channels < 1:
        raise MLRuntimeError("in_channels must be a positive integer")
    if type(num_classes) is not int or num_classes < 1:
        raise MLRuntimeError("num_classes must be a positive integer")
    if backend not in {"torch", "warpconvnet"} or device not in {"cpu", "cuda"}:
        raise MLRuntimeError("build_model requires an explicitly resolved device and backend")
    resolve_runtime(device, backend)
    torch = _import_torch()
    model = (
        _build_torch_model(torch, in_channels, num_classes)
        if backend == "torch"
        else _build_warp_model(torch, in_channels, num_classes)
    )
    return model.to(device)
