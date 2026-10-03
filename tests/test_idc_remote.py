from pathlib import Path

import pytest

from scripts import idc_remote as remote


def test_identifiers_and_argv_are_not_shell_programs():
    assert remote.identifier('check-001') == 'check-001'
    for value in ['../escape', '-option', 'a b', 'x;touch bad']:
        with pytest.raises(ValueError):
            remote.identifier(value)
    assert remote.remote_command('/home/u/a b', ['status', 'j1']).startswith('python3 ')
    with pytest.raises(ValueError):
        remote.validate_request({'job': 'ok', 'revision': 'main', 'argv': ['make', 'check']})


def test_receipt_detects_corruption_and_missing_files(tmp_path):
    (tmp_path / 'checkpoint.pt').write_bytes(b'weights')
    receipt = remote.make_receipt(tmp_path)
    remote.verify_receipt(tmp_path, receipt)
    (tmp_path / 'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        remote.verify_receipt(tmp_path, receipt)


def test_receipt_rejects_paths_and_external_symlinks(tmp_path):
    outside = tmp_path.parent / 'secret.txt'
    outside.write_text('never copy')
    (tmp_path / 'escape').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        remote.make_receipt(tmp_path)
    with pytest.raises(ValueError):
        remote.verify_receipt(tmp_path, {'../secret.txt': {'sha256': '0', 'size': 10}})


def test_existing_local_delivery_cannot_be_overwritten(tmp_path):
    folder = tmp_path / 'job'; folder.mkdir()
    remote.check_destination(folder, 'a' * 40)
    remote.dump(folder / 'remote_status.json', {'revision': 'a' * 40})
    remote.check_destination(folder, 'a' * 40)
    with pytest.raises(ValueError):
        remote.check_destination(folder, 'b' * 40)
    (folder / 'other.txt').write_text('user data')
    (folder / 'remote_status.json').unlink()
    with pytest.raises(ValueError):
        remote.check_destination(folder, 'a' * 40)


def test_failed_task_keeps_every_output(tmp_path):
    output = tmp_path / 'output'; output.mkdir()
    (output / 'failure.pt').write_bytes(b'evidence')
    remote.finalize(output, {'job': 'j', 'revision': 'a' * 40}, 3, 'failed')
    assert (output / 'failure.pt').read_bytes() == b'evidence'
    remote.verify_receipt(output, remote.read(output / 'receipt.json'))
    assert remote.read(output / 'remote_status.json')['exit_code'] == 3


def test_no_tracked_source_is_changed_by_metadata(tmp_path):
    source = tmp_path / 'policy.py'; source.write_text('frozen')
    before = remote.sha(source)
    remote.dump(tmp_path / 'status.json', {'state': 'queued'})
    assert remote.sha(source) == before
