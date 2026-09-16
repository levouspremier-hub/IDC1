"""M1.3f-b 测试：公开数据源获取、许可冻结与可复现实物化。

改前缺陷（本文件在实现前必须为红）：

- 不存在 `scripts.fetch_m13f_public_sources` 模块，也没有 `--fetch` / `--verify`；
- 没有 hash / 许可 / schema / 大小 / URL 校验；
- 没有原子下载与「已冻结文件不同则拒绝覆盖」；
- 没有把四条红线固化成可测试的守卫：2026 光伏 profile 冒充 2024 真值、
  年度碳因子冒充半小时真值、未批准参数生成正式 PV/风电、
  2019 trace 静默重放成 2024 arrival。

**网络纪律**：本文件的非 slow 用例**一律不联网**，全部走本地 HTTP fixture
（127.0.0.1 的临时端口）或注入的 opener。真实网络的用例只允许标 `slow`。
"""

import hashlib
import http.server
import importlib
import json
import pathlib
import threading

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = "scripts.fetch_m13f_public_sources"
MANIFEST = REPO_ROOT / "data/manifest/m13f_public_sources.json"
READY = REPO_ROOT / "data/raw/public_benchmarks/README.md"
EXPECTED_READINESS = {
    "public_source_frozen": True,
    "local_pv_kw_ready": False,
    "wind_generation_kw_ready": False,
    "carbon_intensity_ready": False,
    "arrival_ready": False,
    "formal_scenario_bundle_ready": False,
    "formal_training_ready": False,
}


def mod():
    return importlib.import_module(MODULE)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --- 本地 HTTP fixture（不联网） ----------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    payloads: dict = {}
    fail_with: dict = {}

    def do_GET(self):  # noqa: N802
        if self.path in self.fail_with:
            self.send_response(self.fail_with[self.path])
            self.end_headers()
            return
        body = self.payloads.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 静音
        return


@pytest.fixture
def http_server():
    """本机临时端口上的静态服务器：**不访问外网**。"""
    _Handler.payloads = {}
    _Handler.fail_with = {}
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _spec(**over):
    base = {
        "source_id": "unit_test_source",
        "route": "C",
        "classification": "modeled_scenario",
        "url": "https://example.invalid/x.csv",
        "license": "MIT",
        "license_url": "https://example.invalid/LICENSE",
        "pinned_ref": "v1",
        "local_name": "unit_test_source.csv",
        "bytes": 0,
        "sha256": "0" * 64,
        "data_year": None,
        "resolution": "n/a",
        "required_columns": (),
        "license_marker": "MIT",
    }
    base.update(over)
    return base


# --- 1. 模块与 manifest 存在性 -----------------------------------------------

def test_module_and_entry_points_exist():
    module = mod()
    for name in ("MAX_SOURCE_BYTES", "SOURCE_SPECS", "main", "verify_local",
                 "download_to_temp", "build_manifest", "SourcePolicyError"):
        assert hasattr(module, name), name


def test_manifest_exists_with_frozen_schema():
    assert MANIFEST.exists(), "缺少 data/manifest/m13f_public_sources.json"
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] == "m1.3f-public-sources-v1"
    assert set(payload["readiness"]) == set(EXPECTED_READINESS)
    assert payload["readiness"] == EXPECTED_READINESS
    assert payload["max_source_bytes"] == 16 * 1024 * 1024


def test_manifest_paths_are_repo_relative_posix():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    text = MANIFEST.read_text(encoding="utf-8")
    assert "/Users/" not in text
    assert str(REPO_ROOT) not in text
    for entry in payload["sources"]:
        path = entry["logical_path"]
        if path is None:
            # 未冻结来源（blocked / refused_over_size_cap）本就没有本地路径
            assert entry["status"] != "frozen", entry["source_id"]
            continue
        assert not path.startswith("/") and "\\" not in path
        assert not any(seg in (".", "..", "") for seg in path.split("/"))
    # 未冻结来源必须写明原因
    for entry in payload["sources"]:
        if entry["status"] != "frozen":
            assert entry.get("blocked_reason") or entry.get("refused_reason"), (
                entry["source_id"])


def test_frozen_files_exist_and_match_the_manifest():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert any(e["status"] == "frozen" for e in payload["sources"])
    for entry in payload["sources"]:
        if entry["status"] != "frozen":
            continue
        path = REPO_ROOT / entry["logical_path"]
        assert path.exists(), entry["logical_path"]
        body = path.read_bytes()
        assert len(body) == entry["bytes"], entry["logical_path"]
        assert _sha256(body) == entry["sha256"], entry["logical_path"]


def test_readme_lists_the_non_frozen_sources():
    text = READY.read_text(encoding="utf-8")
    assert "carbon_intensity" in text
    assert "arrival" in text
    assert "2020" in text or "coverageEnd" in text
    assert "/Users/" not in text


# --- 2. 大小上限与 URL / HTTP 校验 -------------------------------------------

def test_size_cap_is_anchored_and_enforced():
    module = mod()
    assert module.MAX_SOURCE_BYTES == 16 * 1024 * 1024
    oversize = _spec(bytes=module.MAX_SOURCE_BYTES + 1)
    with pytest.raises(module.SourcePolicyError):
        module.assert_within_size_cap(oversize)


def test_http_error_fails_closed(http_server):
    module = mod()
    _Handler.fail_with["/missing.csv"] = 404
    with pytest.raises((module.SourcePolicyError,)) as error:
        module.fetch_bytes(f"{http_server}/missing.csv")
    assert "404" in str(error.value)


def test_non_https_non_loopback_url_is_rejected(monkeypatch):
    """非 HTTPS 且非 loopback 的 URL 必须拒绝（且不得发起任何请求）。"""
    module = mod()

    def forbidden(*args, **kwargs):
        raise AssertionError("非 HTTPS 的 URL 不得发起请求")

    monkeypatch.setattr(module.urllib.request, "urlopen", forbidden)
    with pytest.raises(module.SourcePolicyError, match="HTTPS"):
        module.fetch_bytes("http://example.com/x.csv")


# --- 3. hash / 许可 / schema 校验 --------------------------------------------

def test_hash_mismatch_fails_closed(tmp_path):
    module = mod()
    target = tmp_path / "x.csv"
    target.write_bytes(b"tampered")
    spec = _spec(sha256=_sha256(b"expected"), bytes=8)
    with pytest.raises(module.SourcePolicyError):
        module.validate_frozen_file(spec, target)


def test_byte_count_mismatch_fails_closed(tmp_path):
    module = mod()
    body = b"a,b,c\n1,2,3\n"
    target = tmp_path / "x.csv"
    target.write_bytes(body)
    spec = _spec(sha256=_sha256(body), bytes=len(body) + 1)
    with pytest.raises(module.SourcePolicyError):
        module.validate_frozen_file(spec, target)


def test_license_marker_mismatch_fails_closed(tmp_path):
    module = mod()
    body = b"Apache License 2.0"
    target = tmp_path / "LICENSE"
    target.write_bytes(body)
    spec = _spec(local_name="LICENSE", sha256=_sha256(body), bytes=len(body),
                 license="MIT", license_marker="MIT")
    with pytest.raises(module.SourcePolicyError):
        module.validate_frozen_file(spec, target)


def test_required_columns_are_enforced(tmp_path):
    module = mod()
    body = b"wrong,header\n1,2\n"
    target = tmp_path / "x.csv"
    target.write_bytes(body)
    spec = _spec(sha256=_sha256(body), bytes=len(body),
                 required_columns=("turbine_type", "0.0"))
    with pytest.raises(module.SourcePolicyError):
        module.validate_frozen_file(spec, target)


# --- 4. 原子下载与防覆盖 ------------------------------------------------------

def test_download_is_atomic_and_leaves_no_partial_file(tmp_path, http_server, monkeypatch):
    module = mod()
    body = b"hello,world\n"
    _Handler.payloads["/a.csv"] = body

    def boom(*args, **kwargs):
        raise OSError("injected install failure")

    monkeypatch.setattr(module, "_atomic_install", boom)
    spec = _spec(url=f"{http_server}/a.csv", bytes=len(body), sha256=_sha256(body))
    with pytest.raises(OSError):
        module.download_to_temp(spec, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_existing_different_file_is_not_overwritten(tmp_path):
    module = mod()
    target = tmp_path / "a.csv"
    target.write_bytes(b"frozen-content")
    before = (target.stat().st_mtime_ns, _sha256(target.read_bytes()))
    spec = _spec(local_name="a.csv", bytes=len(b"other"), sha256=_sha256(b"other"))
    with pytest.raises(module.SourcePolicyError):
        module.install_frozen(spec, b"other", tmp_path)
    after = (target.stat().st_mtime_ns, _sha256(target.read_bytes()))
    assert before == after


def test_verify_is_idempotent(tmp_path):
    module = mod()
    result = module.verify_local(MANIFEST, REPO_ROOT)
    again = module.verify_local(MANIFEST, REPO_ROOT)
    assert result["ok"] is True
    assert result == again


def test_default_mode_does_not_touch_the_network(monkeypatch):
    """默认运行（无 `--fetch`）不得联网。"""
    module = mod()

    def forbidden(*args, **kwargs):
        raise AssertionError("默认模式发起了网络访问")

    monkeypatch.setattr(module, "fetch_bytes", forbidden)
    assert module.main(["--verify"]) == 0


# --- 5. 四条红线的守卫 --------------------------------------------------------

def test_2026_solar_profile_cannot_masquerade_as_2024_truth():
    module = mod()
    source = _spec(source_id="ema_solar_profile_2026", data_year=2026,
                   classification="modeled_scenario")
    with pytest.raises(module.SourcePolicyError, match="年份"):
        module.assert_year_matches(source, target_year=2024)


def test_annual_carbon_factor_cannot_masquerade_as_half_hourly_truth():
    module = mod()
    source = _spec(source_id="ema_gef_annual", resolution="annual",
                   classification="external_low_resolution")
    with pytest.raises(module.SourcePolicyError, match="分辨率"):
        module.assert_resolution_matches(source, required="half_hourly")


def test_unapproved_parameters_cannot_generate_formal_pv_or_wind():
    module = mod()
    params = {
        "pv_capacity_kw": "UNAPPROVED",
        "tilt_deg": "UNAPPROVED",
        "azimuth_deg": "UNAPPROVED",
        "array_type": "UNAPPROVED",
        "losses_pct": "UNAPPROVED",
        "hub_height_m": "UNAPPROVED",
        "shear_exponent": "UNAPPROVED",
        "turbine_model": "UNAPPROVED",
        "rated_capacity_kw": "UNAPPROVED",
    }
    for route in ("B", "C"):
        with pytest.raises(module.SourcePolicyError, match="UNAPPROVED"):
            module.assert_parameters_approved(params, route=route)


def test_legacy_pv_capacity_is_not_inherited():
    """`pv_capacity_kw=500` 是旧仿真假设，不得继承为正式口径。"""
    module = mod()
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved({"pv_capacity_kw": 500.0}, route="B")


def test_2019_trace_cannot_be_silently_replayed_as_2024_arrival():
    module = mod()
    with pytest.raises(module.SourcePolicyError, match="replay|重放"):
        module.assert_no_silent_replay(trace_year=2019, target_year=2024)


def test_arrival_source_is_refused_over_the_size_cap():
    """官方 trace 包 142,968,140 B 超 16 MiB 上限 → 必须拒绝，且只登记元数据。"""
    module = mod()
    arrival = [s for s in module.SOURCE_SPECS if s["route"] == "D"]
    assert arrival, "缺少 arrival 来源登记"
    for spec in arrival:
        assert spec["status"] in ("refused_over_size_cap", "blocked")
        assert spec["observed_content_length"] > module.MAX_SOURCE_BYTES


def test_carbon_source_is_blocked_not_fabricated():
    module = mod()
    carbon = [s for s in module.SOURCE_SPECS if s["route"] == "A"]
    assert carbon
    for spec in carbon:
        assert spec["status"] in ("blocked", "not_frozen")
        assert spec["data_year"] != 2024 or spec["status"] != "frozen"


def test_no_readiness_flag_is_prematurely_true():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    readiness = payload["readiness"]
    for key in ("local_pv_kw_ready", "wind_generation_kw_ready",
                "carbon_intensity_ready", "arrival_ready",
                "formal_scenario_bundle_ready", "formal_training_ready"):
        assert readiness[key] is False, key


def test_reserved_split_names_are_not_created():
    for name in ("train.json", "validation.json", "test.json"):
        assert not (REPO_ROOT / "data/manifest" / name).exists(), name


# --- 6. slow：真实网络（可选，不进入 make check） -----------------------------

@pytest.mark.slow
def test_real_sources_match_the_frozen_manifest():
    module = mod()
    for spec in module.SOURCE_SPECS:
        if spec["status"] != "frozen":
            continue
        body = module.fetch_bytes(spec["url"])
        assert len(body) == spec["bytes"]
        assert hashlib.sha256(body).hexdigest() == spec["sha256"]
