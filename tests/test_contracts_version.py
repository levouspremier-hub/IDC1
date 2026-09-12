"""M0.3 gate: the contract version constant exists and is well-formed."""

from contracts import CONTRACT_VERSION, CONTRACT_VERSION_ID


def test_contract_version_is_nonempty_string() -> None:
    assert isinstance(CONTRACT_VERSION, str)
    assert CONTRACT_VERSION.strip() == CONTRACT_VERSION
    assert CONTRACT_VERSION


def test_contract_version_id_is_positive_int() -> None:
    assert isinstance(CONTRACT_VERSION_ID, int)
    assert CONTRACT_VERSION_ID > 0
