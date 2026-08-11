from pathlib import Path
from types import SimpleNamespace

from trhash.backends.local import LocalBackend


def test_torch_checkpoint_is_exported_before_autonomous_serving(monkeypatch, tmp_path: Path):
    backend = LocalBackend.__new__(LocalBackend)
    backend.device = SimpleNamespace(type="cpu")
    exported = tmp_path / "service" / "bundle"
    calls = {}

    def export(*, output):
        calls["export"] = Path(output)
        return exported

    def run_server(model, **options):
        calls["server"] = (model, options)

    backend.export = export
    monkeypatch.setattr("trhash.server.runner.run_server", run_server)

    backend.serve(host="0.0.0.0", port=9000, api_key="secret", jobs_root=tmp_path / "service")

    assert calls["export"] == exported
    assert calls["server"] == (
        exported,
        {"device": "cpu", "host": "0.0.0.0", "port": 9000, "api_key": "secret"},
    )
