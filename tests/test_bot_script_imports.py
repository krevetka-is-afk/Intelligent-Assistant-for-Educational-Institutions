import importlib.util
import sys
import types
from pathlib import Path


def test_service_package_import_does_not_mask_missing_dependency(monkeypatch):
    fake_package = types.ModuleType("fakepkg")
    fake_package.__path__ = []
    monkeypatch.setitem(sys.modules, "fakepkg", fake_package)

    app_runtime_module = types.ModuleType("app_runtime")
    app_runtime_module.log_extra = lambda **kwargs: kwargs
    app_runtime_module.setup_logging = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "app_runtime", app_runtime_module)

    module_path = Path(__file__).resolve().parents[1] / "src" / "bot" / "service.py"
    spec = importlib.util.spec_from_file_location("fakepkg.service", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        assert exc.name == "fakepkg.api_client"
        return

    raise AssertionError("Expected ModuleNotFoundError for package import path")
