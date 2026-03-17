"""Tests for aieng.llm_interp.utils."""

from unittest.mock import MagicMock, patch

import torch
from aieng.llm_interp.utils import get_device, release_memory


class TestGetDevice:
    """Tests for :func:`get_device`."""

    def test_returns_torch_device(self) -> None:
        """get_device should always return a torch.device instance."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_returns_cuda_when_available(self) -> None:
        """get_device should return a CUDA device when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = get_device()
        assert device == torch.device("cuda")

    def test_returns_mps_when_cuda_unavailable_and_mps_available(self) -> None:
        """get_device should return MPS when CUDA is absent but MPS is present."""
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps", mock_mps),
        ):
            device = get_device()

        assert device == torch.device("mps")

    def test_returns_cpu_when_only_cpu_available(self) -> None:
        """get_device should fall back to CPU when neither CUDA nor MPS is available."""
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = False

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps", mock_mps),
        ):
            device = get_device()

        assert device == torch.device("cpu")

    def test_returns_cpu_when_mps_attribute_missing(self) -> None:
        """get_device should return CPU when torch.backends has no mps attribute."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "torch.backends.mps",
                new_callable=lambda: type("Broken", (), {"is_available": MagicMock(side_effect=AttributeError)}),
            ),
        ):
            device = get_device()

        assert device == torch.device("cpu")


class TestReleaseMemory:
    """Tests for :func:`release_memory`."""

    def test_returns_none(self) -> None:
        """release_memory should return None."""
        with (
            patch("gc.collect"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            result = release_memory()

        assert result is None

    def test_calls_gc_collect(self) -> None:
        """release_memory should always call gc.collect."""
        with (
            patch("gc.collect") as mock_gc,
            patch("torch.cuda.is_available", return_value=False),
        ):
            release_memory()

        mock_gc.assert_called_once()

    def test_calls_cuda_empty_cache_when_available(self) -> None:
        """release_memory should call torch.cuda.empty_cache when CUDA is available."""
        with (
            patch("gc.collect"),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_cuda_cache,
        ):
            release_memory()

        mock_cuda_cache.assert_called_once()

    def test_does_not_call_cuda_empty_cache_when_unavailable(self) -> None:
        """release_memory should skip torch.cuda.empty_cache when CUDA is absent."""
        with (
            patch("gc.collect"),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.empty_cache") as mock_cuda_cache,
        ):
            release_memory()

        mock_cuda_cache.assert_not_called()

    def test_calls_mps_empty_cache_when_available(self) -> None:
        """release_memory should call torch.mps.empty_cache when MPS is available."""
        mock_mps_backends = MagicMock()
        mock_mps_backends.is_available.return_value = True
        mock_mps_module = MagicMock()

        with (
            patch("gc.collect"),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps", mock_mps_backends),
            patch("torch.mps", mock_mps_module),
        ):
            release_memory()

        mock_mps_module.empty_cache.assert_called_once()

    def test_does_not_call_mps_empty_cache_when_unavailable(self) -> None:
        """release_memory should skip torch.mps.empty_cache when MPS is absent."""
        mock_mps_backends = MagicMock()
        mock_mps_backends.is_available.return_value = False
        mock_mps_module = MagicMock()

        with (
            patch("gc.collect"),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps", mock_mps_backends),
            patch("torch.mps", mock_mps_module),
        ):
            release_memory()

        mock_mps_module.empty_cache.assert_not_called()

    def test_handles_mps_attribute_error_gracefully(self) -> None:
        """release_memory should not raise when torch.backends.mps is missing."""

        class _BrokenMPS:
            @staticmethod
            def is_available() -> bool:
                raise AttributeError("mps not supported")

        with (
            patch("gc.collect"),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps", _BrokenMPS()),
        ):
            # Should not raise
            release_memory()
