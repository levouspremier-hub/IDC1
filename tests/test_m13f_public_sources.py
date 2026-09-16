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
        path = entry.get("logical_path")
        if path is None:
            # 未冻结来源（blocked / refused_over_size_cap）按键集合**不带**该键
            assert entry["status"] != "frozen", entry["source_id"]
            assert "logical_path" not in entry, entry["source_id"]
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
    """**M1.3f-b-R2 迁移**：R2 起参数名集合必须精确等于该路线的冻结批准组，
    因此「路径正确但状态为 UNAPPROVED」与「参数组不匹配」都要分别验证。"""
    module = mod()

    # (a) 完整的 B 组，但每项状态被改成 UNAPPROVED → 拒绝（消息含 UNAPPROVED）
    unapproved = {
        name: dict(entry, status="UNAPPROVED", decision_id=None, approved_on=None)
        for name, entry in _group(B_GROUP).items()
    }
    with pytest.raises(module.SourcePolicyError, match="UNAPPROVED"):
        module.assert_parameters_approved(unapproved, route="B")

    # (b) 把 C 组参数喂给 route B（以及反向）→ 参数名集合不符，拒绝
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved(_approved_params(C_GROUP), route="B")
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved(_approved_params(B_GROUP), route="C")

    # (c) 非结构化标量 → 拒绝
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved({"pv_capacity_kw": 500.0}, route="B")


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


# --- 7. M1.3f-b-R1：严格 manifest 校验（单一入口） ---------------------------

import copy  # noqa: E402

TOP_KEYS = (
    "schema", "contract_version", "max_source_bytes", "max_source_bytes_anchor",
    "frozen_at_utc", "sources", "human_approved_parameters",
    "unapproved_parameters", "red_lines", "readiness",
)
SOURCE_COMMON_KEYS = (
    "source_id", "route", "role", "classification", "status", "url", "license",
    "license_url", "pinned_ref", "data_year", "resolution",
)
STATUS_EXTRA_KEYS = {
    "frozen": ("logical_path", "bytes", "sha256"),
    "blocked": ("blocked_reason",),
    "refused_over_size_cap": ("refused_reason", "observed_content_length"),
}
PARAM_KEYS = ("value", "unit", "status", "decision_id", "approved_on")


def _load():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _put(tmp_path, payload):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _rejects(tmp_path, payload):
    module = mod()
    with pytest.raises(module.SourcePolicyError):
        module.validate_public_source_manifest(_put(tmp_path, payload), REPO_ROOT)


def _entry(payload, source_id):
    return next(e for e in payload["sources"] if e["source_id"] == source_id)


def test_validate_entry_point_exists_and_accepts_the_real_manifest():
    module = mod()
    result = module.validate_public_source_manifest(MANIFEST, REPO_ROOT)
    assert result["ok"] is True
    assert result["sources"] == len(module.SOURCE_SPECS)


def test_validate_also_checks_blocked_and_refused_sources():
    """§二：blocked / refused 来源同样必须被校验，**不得被 verify 跳过**。"""
    module = mod()
    payload = _load()
    assert any(e["status"] != "frozen" for e in payload["sources"])
    skipped = module.validated_source_ids(MANIFEST, REPO_ROOT)
    for entry in payload["sources"]:
        assert entry["source_id"] in skipped, entry["source_id"]


# --- 7.1 source 集合与顺序 ----------------------------------------------------

def test_empty_sources_is_rejected(tmp_path):
    payload = _load()
    payload["sources"] = []
    _rejects(tmp_path, payload)


MISSING_SOURCE_IDS: tuple[str, ...] = (
    "pvlib_pvwatts_license", "windpowerlib_power_curves",
    "ema_grid_emission_factor_annual", "azure_functions_2019_trace",
)


@pytest.mark.parametrize("source_id", MISSING_SOURCE_IDS)
def test_missing_source_id_is_rejected(tmp_path, source_id):
    payload = _load()
    payload["sources"] = [e for e in payload["sources"] if e["source_id"] != source_id]
    _rejects(tmp_path, payload)


def test_duplicated_source_id_is_rejected(tmp_path):
    payload = _load()
    payload["sources"] = payload["sources"] + [copy.deepcopy(payload["sources"][0])]
    _rejects(tmp_path, payload)


def test_unknown_source_id_is_rejected(tmp_path):
    payload = _load()
    entry = copy.deepcopy(payload["sources"][0])
    entry["source_id"] = "shadow_source"
    payload["sources"] = payload["sources"] + [entry]
    _rejects(tmp_path, payload)


def test_reordered_sources_are_rejected(tmp_path):
    payload = _load()
    payload["sources"] = list(reversed(payload["sources"]))
    _rejects(tmp_path, payload)


# --- 7.2 逐字段篡改矩阵 -------------------------------------------------------

def _t_url(p):
    _entry(p, "pvlib_pvwatts_model")["url"] = "https://evil.example.invalid/x.py"


def _t_license(p):
    _entry(p, "pvlib_pvwatts_license")["license"] = "Apache-2.0"


def _t_pinned_ref(p):
    _entry(p, "windpowerlib_power_curves")["pinned_ref"] = "v9.9.9"


def _t_logical_path(p):
    _entry(p, "windpowerlib_turbine_data")["logical_path"] = "data/raw/other.csv"


def _t_bytes(p):
    _entry(p, "pvlib_pvwatts_model")["bytes"] = 1


def _t_sha256(p):
    _entry(p, "windpowerlib_power_curves")["sha256"] = "0" * 64


def _t_classification(p):
    _entry(p, "pvlib_pvwatts_model")["classification"] = "observed"


def _t_status_frozen(p):
    _entry(p, "ema_grid_emission_factor_annual")["status"] = "frozen"


def _t_status_refused(p):
    _entry(p, "azure_functions_2019_trace")["status"] = "frozen"


def _t_resolution(p):
    _entry(p, "ema_grid_emission_factor_annual")["resolution"] = "half_hourly"


def _t_drop_blocked_reason(p):
    _entry(p, "ema_grid_emission_factor_annual").pop("blocked_reason")


def _t_drop_refused_reason(p):
    _entry(p, "azure_functions_2019_trace").pop("refused_reason")


def _t_carbon_frozen(p):
    entry = _entry(p, "ema_grid_emission_factor_annual")
    entry["status"] = "frozen"
    entry["bytes"] = 1
    entry["sha256"] = "0" * 64
    entry["logical_path"] = "data/raw/public_benchmarks/fake.json"
    entry.pop("blocked_reason")


def _t_readiness(p):
    p["readiness"]["carbon_intensity_ready"] = True


def _t_red_lines(p):
    p["red_lines"] = p["red_lines"][:-1]


def _t_params(p):
    p["human_approved_parameters"]["local_pv_kw"]["pv_capacity_kw"]["value"] = 999.0


def _t_unapproved(p):
    p["unapproved_parameters"] = {}


def _t_cap(p):
    p["max_source_bytes"] = 32 * 1024 * 1024


def _t_schema(p):
    p["schema"] = "m1.3f-public-sources-v0"


def _t_contract_version(p):
    p["contract_version"] = "contract-v7"


def _t_anchor(p):
    p["max_source_bytes_anchor"] = "arbitrary"


def _t_frozen_at(p):
    p["frozen_at_utc"] = "2026-09-16T00:00:00"


def _t_unknown_top(p):
    p["future_extension"] = 1


def _t_unknown_entry(p):
    _entry(p, "pvlib_pvwatts_license")["extra_note"] = "x"


def _t_unknown_readiness(p):
    p["readiness"]["extra_ready"] = False


def _t_unknown_param_key(p):
    p["human_approved_parameters"]["local_pv_kw"]["tilt_deg"]["extra"] = 1


def _t_drop_param_field(p):
    p["human_approved_parameters"]["local_pv_kw"]["tilt_deg"].pop("decision_id")


def _t_legacy_inherited(p):
    entry = p["human_approved_parameters"]["local_pv_kw"]["pv_capacity_kw"]
    entry["status"] = "legacy_inherited"


def _t_bool_as_int(p):
    _entry(p, "pvlib_pvwatts_license")["bytes"] = True


MANIFEST_TAMPERINGS = [
    ("url", _t_url), ("license", _t_license), ("pinned_ref", _t_pinned_ref),
    ("logical_path", _t_logical_path), ("bytes", _t_bytes), ("sha256", _t_sha256),
    ("classification", _t_classification), ("carbon_status_frozen", _t_status_frozen),
    ("arrival_status_frozen", _t_status_refused), ("resolution", _t_resolution),
    ("drop_blocked_reason", _t_drop_blocked_reason),
    ("drop_refused_reason", _t_drop_refused_reason),
    ("carbon_frozen", _t_carbon_frozen), ("readiness", _t_readiness),
    ("red_lines", _t_red_lines), ("params", _t_params), ("unapproved", _t_unapproved),
    ("cap", _t_cap), ("schema", _t_schema), ("contract_version", _t_contract_version),
    ("anchor", _t_anchor), ("frozen_at", _t_frozen_at), ("unknown_top", _t_unknown_top),
    ("unknown_entry", _t_unknown_entry), ("unknown_readiness", _t_unknown_readiness),
    ("unknown_param_key", _t_unknown_param_key),
    ("drop_param_field", _t_drop_param_field),
    ("legacy_inherited", _t_legacy_inherited), ("bool_as_int", _t_bool_as_int),
]


@pytest.mark.parametrize("name,tamper", MANIFEST_TAMPERINGS)
def test_manifest_tampering_is_rejected(tmp_path, name, tamper):
    payload = copy.deepcopy(_load())
    tamper(payload)
    _rejects(tmp_path, payload)


MALFORMED_MANIFESTS: tuple = (
    None, 42, True, "text", [], [1, 2], {"schema": "x"},
    {"sources": []}, {"sources": {}}, {"sources": [None]},
    {"sources": ["x"]}, {"sources": [{"source_id": 1}]},
    {"readiness": []}, {"red_lines": "x"},
)

@pytest.mark.parametrize("bad", MALFORMED_MANIFESTS)
def test_malformed_manifest_never_leaks_builtin_errors(tmp_path, bad):
    """§一.8：只抛 SourcePolicyError，不泄漏 KeyError/TypeError/AttributeError。"""
    module = mod()
    path = tmp_path / "m.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(module.SourcePolicyError):
        module.validate_public_source_manifest(path, REPO_ROOT)


def test_missing_manifest_file_is_a_policy_error(tmp_path):
    module = mod()
    with pytest.raises(module.SourcePolicyError):
        module.validate_public_source_manifest(tmp_path / "nope.json", REPO_ROOT)


# --- 7.3 不可变性与重复 --fetch ----------------------------------------------

def test_write_manifest_refuses_a_semantically_different_manifest(tmp_path):
    module = mod()
    path = tmp_path / "m.json"
    payload = _load()
    module.write_manifest_atomic(payload, path)
    changed = copy.deepcopy(payload)
    changed["max_source_bytes"] = 32
    with pytest.raises(module.SourcePolicyError):
        module.write_manifest_atomic(changed, path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_write_manifest_is_a_noop_for_identical_content(tmp_path):
    module = mod()
    path = tmp_path / "m.json"
    payload = _load()
    module.write_manifest_atomic(payload, path)
    st = path.stat()
    before = (st.st_size, _sha256(path.read_bytes()), st.st_mtime_ns)
    module.write_manifest_atomic(payload, path)
    st = path.stat()
    assert (st.st_size, _sha256(path.read_bytes()), st.st_mtime_ns) == before


def test_repeated_fetch_reuses_the_existing_frozen_at_utc(tmp_path, monkeypatch):
    """§一.11：重复 --fetch 必须复用现存合法 frozen_at_utc，不得随墙钟漂移。"""
    module = mod()
    dest = tmp_path / "root" / "data/raw/public_benchmarks"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "root" / "m.json"

    frozen_dir = REPO_ROOT / "data/raw/public_benchmarks"
    monkeypatch.setattr(
        module, "fetch_bytes",
        lambda url, **kw: (frozen_dir / _local_for(url)).read_bytes(),
    )
    monkeypatch.setattr(module, "_now_utc", lambda: "2026-09-16T00:00:00+00:00")
    module.run_fetch(dest_dir=dest, manifest_path=manifest)
    first = manifest.read_bytes()
    assert json.loads(first)["frozen_at_utc"] == "2026-09-16T00:00:00+00:00"

    monkeypatch.setattr(module, "_now_utc", lambda: "2027-01-01T00:00:00+00:00")
    module.run_fetch(dest_dir=dest, manifest_path=manifest)
    second = manifest.read_bytes()
    assert json.loads(second)["frozen_at_utc"] == "2026-09-16T00:00:00+00:00"
    assert first == second


def _local_for(url: str) -> str:
    spec = next(s for s in mod().SOURCE_SPECS if s["url"] == url)
    return spec["local_name"]


# --- 7.4 审批守卫（可延续） ---------------------------------------------------

_APPROVED = {
    "value": 500.0, "unit": "kW", "status": "human_approved",
    "decision_id": "B4", "approved_on": "2026-09-16",
}


def test_structured_human_approval_is_accepted():
    """**M1.3f-b-R2 迁移**：只有**完整**的冻结批准组才被接受（子集不再通过）。"""
    module = mod()
    module.assert_parameters_approved(_approved_params(B_GROUP), route="B")
    module.assert_parameters_approved(_approved_params(C_GROUP), route="C")


@pytest.mark.parametrize("mutation", [
    {"decision_id": ""},
    {"decision_id": None},
    {"status": "legacy_inherited"},
    {"status": "UNAPPROVED"},
    {"approved_on": "2026-09-16T00:00:00"},
])
def test_incomplete_or_legacy_approvals_are_rejected(mutation):
    module = mod()
    entry = dict(_APPROVED)
    entry.update(mutation)
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved({"pv_capacity_kw": entry}, route="B")


@pytest.mark.parametrize("bad", [500.0, "500", None, True, [500.0]])
def test_unstructured_parameter_values_are_rejected(bad):
    module = mod()
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved({"pv_capacity_kw": bad}, route="B")


def test_the_real_approved_parameters_block_validates():
    module = mod()
    payload = _load()
    for route, group in (("B", "local_pv_kw"), ("C", "wind_generation_kw")):
        module.assert_parameters_approved(
            payload["human_approved_parameters"][group], route=route)


# --- 8. M1.3f-b-R2：路径绑定、reason 恒等、审批参数组、特例上限 ---------------

REAL_LICENSE = "data/raw/public_benchmarks/pvlib_v0.15.2_LICENSE.txt"


def _frozen_entry(payload, source_id="pvlib_pvwatts_license"):
    return next(e for e in payload["sources"] if e["source_id"] == source_id)


def _accepts(tmp_path, payload):
    mod().validate_public_source_manifest(_put(tmp_path, payload), REPO_ROOT)


# --- 8.1 logical_path 必须逐字绑定 spec.local_name ---------------------------

def test_correct_logical_path_is_accepted(tmp_path):
    _accepts(tmp_path, _load())


@pytest.mark.parametrize("bad", [
    str(REPO_ROOT / REAL_LICENSE),                      # 绝对路径
    REAL_LICENSE.replace("pvlib_v0.15.2", "./pvlib_v0.15.2"),
    REAL_LICENSE.replace("public_benchmarks", "public_benchmarks/.."),
    REAL_LICENSE.replace("public_benchmarks", "public_benchmarks/./x/.."),
    "data\\raw\\public_benchmarks\\pvlib_v0.15.2_LICENSE.txt",
    "data//raw/public_benchmarks/pvlib_v0.15.2_LICENSE.txt",
    "data/raw/../raw/public_benchmarks/pvlib_v0.15.2_LICENSE.txt",
    "data/raw/public_benchmarks/windpowerlib_v0.2.2_LICENSE.txt",  # 另一份文件
    "pvlib_v0.15.2_LICENSE.txt",                        # 缺前缀
    "data/raw/public_benchmarks/",                      # 目录
])
def test_logical_path_must_be_bound_to_the_frozen_local_name(tmp_path, bad):
    payload = _load()
    _frozen_entry(payload)["logical_path"] = bad
    _rejects(tmp_path, payload)


def test_decoy_file_with_identical_content_is_still_rejected(tmp_path):
    """把 frozen 文件复制成**另一个名字**（内容/hash 相同）仍必须拒绝。"""
    root = tmp_path / "root"
    target = root / REAL_LICENSE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((REPO_ROOT / REAL_LICENSE).read_bytes())
    decoy = target.parent / "decoy_copy.txt"
    decoy.write_bytes(target.read_bytes())
    assert _sha256(decoy.read_bytes()) == _sha256(target.read_bytes())

    payload = _load()
    _frozen_entry(payload)["logical_path"] = "data/raw/public_benchmarks/decoy_copy.txt"
    path = _put(tmp_path, payload)
    with pytest.raises(mod().SourcePolicyError):
        mod().validate_public_source_manifest(path, root)


# --- 8.2 reason 与所有 status-specific 字段逐字恒等 --------------------------

def test_forged_blocked_reason_is_rejected(tmp_path):
    payload = _load()
    _frozen_entry(payload, "ema_grid_emission_factor_annual")["blocked_reason"] = "x"
    _rejects(tmp_path, payload)


def test_forged_refused_reason_is_rejected(tmp_path):
    payload = _load()
    _frozen_entry(payload, "azure_functions_2019_trace")["refused_reason"] = "x"
    _rejects(tmp_path, payload)


@pytest.mark.parametrize("field", SOURCE_COMMON_KEYS)
def test_every_common_field_is_binding(tmp_path, field):
    payload = _load()
    entry = _frozen_entry(payload)
    entry[field] = 12345 if field not in ("pinned_ref", "resolution") else 12345
    _rejects(tmp_path, payload)


@pytest.mark.parametrize("field,value", [
    ("observed_content_length", 1),
    ("exception_cap_bytes", 1),
    ("decision_id", "B999"),
])
def test_refused_status_fields_are_binding(tmp_path, field, value):
    payload = _load()
    _frozen_entry(payload, "azure_functions_2019_trace")[field] = value
    _rejects(tmp_path, payload)


# --- 8.3 审批参数组必须完整恒等 ----------------------------------------------

B_GROUP = "local_pv_kw"
C_GROUP = "wind_generation_kw"


def _group(group):
    return mod().HUMAN_APPROVED_PARAMETERS[group]


def _approved_params(group):
    return copy.deepcopy(_group(group))


def test_the_complete_frozen_groups_are_accepted():
    module = mod()
    module.assert_parameters_approved(_approved_params(B_GROUP), route="B")
    module.assert_parameters_approved(_approved_params(C_GROUP), route="C")


def test_empty_parameter_group_is_rejected():
    module = mod()
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved({}, route="B")


def test_missing_extra_or_renamed_parameter_is_rejected():
    module = mod()
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved(
            {k: v for k, v in _approved_params(B_GROUP).items() if k != "tilt_deg"},
            route="B")
    extra = _approved_params(B_GROUP)
    extra["shadow_kw"] = mod()._approved(1.0, "kW", "B4")
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved(extra, route="B")
    renamed = {k: v for k, v in _approved_params(B_GROUP).items() if k != "tilt_deg"}
    renamed["tilt"] = _approved_params(B_GROUP)["tilt_deg"]
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved(renamed, route="B")
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved(_approved_params(C_GROUP), route="B")


@pytest.mark.parametrize("bad", [
    True, False, float("nan"), float("inf"), float("-inf"),
    [500.0], {"a": 1}, None, "500",
])
def test_non_finite_or_non_numeric_values_are_rejected(bad):
    module = mod()
    params = _approved_params(B_GROUP)
    params["pv_capacity_kw"]["value"] = bad
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved(params, route="B")


@pytest.mark.parametrize("over", [
    {"unit": None}, {"unit": ""}, {"decision_id": "WHATEVER"},
    {"decision_id": ""}, {"approved_on": "2026-13-99"},
    {"approved_on": "2026-09-17"}, {"value": 501.0}, {"value": 500},
])
def test_approval_record_must_match_the_frozen_decision(over):
    module = mod()
    params = _approved_params(B_GROUP)
    params["pv_capacity_kw"].update(over)
    if over.get("value") == 500:
        # 500 与 500.0 数值相等，且必须仍然通过（int 是合法数值）
        module.assert_parameters_approved(params, route="B")
        return
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved(params, route="B")


def test_five_key_set_is_enforced(tmp_path):
    module = mod()
    params = _approved_params(B_GROUP)
    params["pv_capacity_kw"]["extra"] = 1
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved(params, route="B")
    params = _approved_params(B_GROUP)
    params["pv_capacity_kw"].pop("unit")
    with pytest.raises(module.SourcePolicyError):
        module.assert_parameters_approved(params, route="B")


# --- 8.4 特例上限只绑定精确的 Azure 来源 -------------------------------------

def test_exception_cap_is_bound_to_the_exact_azure_source():
    module = mod()
    azure = next(s for s in module.SOURCE_SPECS
                 if s["source_id"] == "azure_functions_2019_trace")
    assert module.effective_cap(azure) == 160 * 1024 * 1024

    for field, value in (("url", "https://evil.invalid/x.tar.xz"),
                         ("pinned_ref", "other"),
                         ("decision_id", "B999"),
                         ("exception_cap_bytes", 200 * 1024 * 1024)):
        forged = dict(azure)
        forged[field] = value
        assert module.effective_cap(forged) == module.MAX_SOURCE_BYTES, field

    self_made = {"source_id": "not_azure", "url": azure["url"],
                 "pinned_ref": azure["pinned_ref"], "decision_id": "B3",
                 "exception_cap_bytes": 160 * 1024 * 1024}
    assert module.effective_cap(self_made) == module.MAX_SOURCE_BYTES


def test_self_made_spec_cannot_bypass_the_default_cap():
    module = mod()
    forged = {"source_id": "not_azure", "url": "https://evil.invalid/x",
              "pinned_ref": "v1", "decision_id": "B3",
              "exception_cap_bytes": 160 * 1024 * 1024,
              "bytes": 100 * 1024 * 1024}
    with pytest.raises(module.SourcePolicyError):
        module.assert_within_size_cap(forged)


# --- 8.5 run_fetch 的复核 root 必须与 dest_dir 一致 --------------------------

def _tmp_root(tmp_path):
    root = tmp_path / "root"
    dest = root / "data/raw/public_benchmarks"
    dest.mkdir(parents=True, exist_ok=True)
    return root, dest


def test_validation_root_follows_the_logical_prefix(tmp_path):
    module = mod()
    root, dest = _tmp_root(tmp_path)
    assert module.validation_root_for(dest) == root
    with pytest.raises(module.SourcePolicyError):
        module.validation_root_for(tmp_path / "wrong/place")


def test_run_fetch_validates_the_temporary_download(tmp_path, monkeypatch):
    """§一.12：临时 dest_dir 的复核必须校验**刚下载的**文件，不得依赖仓库正式文件。"""
    module = mod()
    real = REPO_ROOT / "data/raw/public_benchmarks"
    monkeypatch.setattr(
        module, "fetch_bytes",
        lambda url, **kw: (real / _local_for(url)).read_bytes(),
    )
    root, dest = _tmp_root(tmp_path)
    manifest = root / "manifest.json"
    module.run_fetch(dest_dir=dest, manifest_path=manifest)

    victim = dest / "pvlib_v0.15.2_pvsystem.py"
    assert victim.exists()
    victim.unlink()
    assert (real / "pvlib_v0.15.2_pvsystem.py").exists()  # 仓库正式文件仍在
    with pytest.raises(module.SourcePolicyError):
        module.validate_public_source_manifest(manifest, module.validation_root_for(dest))


def test_run_fetch_fails_when_a_downloaded_file_is_broken(tmp_path, monkeypatch):
    module = mod()
    real = REPO_ROOT / "data/raw/public_benchmarks"
    monkeypatch.setattr(
        module, "fetch_bytes",
        lambda url, **kw: (real / _local_for(url)).read_bytes(),
    )
    root, dest = _tmp_root(tmp_path)
    manifest = root / "manifest.json"
    module.run_fetch(dest_dir=dest, manifest_path=manifest)
    (dest / "windpowerlib_v0.2.2_turbine_data.csv").write_bytes(b"tampered")
    with pytest.raises(module.SourcePolicyError):
        module.validate_public_source_manifest(manifest, module.validation_root_for(dest))


# --- 8.6 分类声明必须只有一套 ------------------------------------------------

def test_classifications_cover_every_declared_source():
    module = mod()
    declared = {spec["classification"] for spec in module.SOURCE_SPECS}
    assert declared <= set(module.CLASSIFICATIONS), declared
    assert "human_approved_external_low_resolution" in module.CLASSIFICATIONS
