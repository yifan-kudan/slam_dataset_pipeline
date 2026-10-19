"""config/paths.yaml resolution.

The point of paths.py is that no script carries an absolute path, so what needs
covering is the resolution itself: environment variables win over fallbacks, a
fallback can reference another key, and a key that cannot be resolved says which
key and which variable rather than raising a bare KeyError.
"""
import pytest

import paths


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Write a paths file and make it the one paths.py reads."""
    def write(text: str):
        file = tmp_path / "paths.yaml"
        file.write_text(text, encoding="utf-8")
        monkeypatch.setenv("SDP_PATHS", str(file))
        paths.reload()
        return file
    yield write
    paths.reload()


# --------------------------------------------------------------------------
# Expansion
# --------------------------------------------------------------------------
def test_environment_variable_wins_over_the_fallback(cfg, monkeypatch):
    cfg("a:\n  b: ${SDP_TEST_ROOT:-/fallback}\n")
    assert paths.get("a.b") == "/fallback"
    monkeypatch.setenv("SDP_TEST_ROOT", "/from/env")
    assert paths.get("a.b") == "/from/env"


def test_fallback_may_itself_reference_a_variable(cfg, monkeypatch):
    monkeypatch.setenv("HOME", "/home/someone")
    cfg("a: ${SDP_UNSET:-${HOME}/prj}\n")
    assert paths.get("a") == "/home/someone/prj"


def test_a_key_can_reference_another_key(cfg):
    cfg("datasets:\n"
        "  d:\n"
        "    root: /data/d\n"
        "    gt: ${datasets.d.root}/groundtruth\n")
    assert paths.get("datasets.d.gt") == "/data/d/groundtruth"


def test_sdp_repo_points_at_the_checkout(cfg):
    cfg("profile: ${SDP_REPO}/config/dds/x.xml\n")
    assert paths.get("profile") == f"{paths.REPO}/config/dds/x.xml"


def test_tilde_is_expanded(cfg, monkeypatch):
    monkeypatch.setenv("HOME", "/home/someone")
    cfg("a: ~/thing\n")
    assert paths.get("a") == "/home/someone/thing"


# --------------------------------------------------------------------------
# Failure modes -- the message has to name the key, or a broken config is a hunt
# --------------------------------------------------------------------------
def test_unset_variable_without_a_fallback_names_the_key_and_the_variable(cfg):
    cfg("a:\n  b: ${SDP_DEFINITELY_UNSET}\n")
    with pytest.raises(paths.PathsError) as exc:
        paths.get("a.b")
    assert "a.b" in str(exc.value) and "SDP_DEFINITELY_UNSET" in str(exc.value)


def test_a_reference_cycle_is_reported_not_recursed(cfg):
    cfg("a: ${b}\nb: ${a}\n")
    with pytest.raises(paths.PathsError, match="cycle"):
        paths.get("a")


def test_unknown_key_is_an_error_unless_a_default_is_given(cfg):
    cfg("a: 1\n")
    with pytest.raises(paths.PathsError):
        paths.get("nope.missing")
    assert paths.get("nope.missing", "fallback") == "fallback"


def test_a_mapping_or_list_is_not_a_value(cfg):
    cfg("a:\n  b: 1\nlist:\n  - x\n")
    with pytest.raises(paths.PathsError):
        paths.get("a")
    with pytest.raises(paths.PathsError):
        paths.get("list")


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------
def test_get_list_expands_every_element(cfg, monkeypatch):
    monkeypatch.setenv("SDP_TEST_WS", "/ws")
    cfg("ros:\n  overlays:\n    - ${SDP_TEST_WS}/a/setup.bash\n    - /plain/b\n")
    assert paths.get_list("ros.overlays") == ["/ws/a/setup.bash", "/plain/b"]


def test_section_and_dataset_expand_scalars_and_skip_nested(cfg):
    cfg("datasets:\n"
        "  d:\n"
        "    root: /data/d\n"
        "    kdm: ${datasets.d.root}/maps\n"
        "    nested:\n"
        "      x: 1\n")
    assert paths.dataset("d") == {"root": "/data/d", "kdm": "/data/d/maps"}


def test_path_returns_a_pathlib_path(cfg):
    cfg("a: /x/y\n")
    assert paths.path("a").name == "y"


# --------------------------------------------------------------------------
# The committed template has to resolve on its own, or a clone cannot run
# --------------------------------------------------------------------------
def test_the_shipped_template_resolves_every_key():
    paths.reload()
    template = paths.CONFIG_DIR / "examples" / "paths.example.yaml"
    resolved = {}
    for dotted, _ in paths._flatten(paths.load(template)):
        base, _, index = dotted.partition("[")
        resolved[dotted] = (paths.get_list(base, config=template)[int(index[:-1])]
                            if index else paths.get(dotted, config=template))
    assert resolved["ros.dds_profile"].endswith("config/dds/fastdds_shm_big.xml")
    assert all(v.startswith("/") for v in resolved.values())
