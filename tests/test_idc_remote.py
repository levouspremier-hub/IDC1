
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
    folder = tmp_path / 'job'
    folder.mkdir()
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
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'failure.pt').write_bytes(b'evidence')
    remote.finalize(output, {'job': 'j', 'revision': 'a' * 40}, 3, 'failed')
    assert (output / 'failure.pt').read_bytes() == b'evidence'
    remote.verify_receipt(output, remote.read(output / 'receipt.json'))
    assert remote.read(output / 'remote_status.json')['exit_code'] == 3


def test_no_tracked_source_is_changed_by_metadata(tmp_path):
    source = tmp_path / 'policy.py'
    source.write_text('frozen')
    before = remote.sha(source)
    remote.dump(tmp_path / 'status.json', {'state': 'queued'})
    assert remote.sha(source) == before


def test_detached_jobs_queue_pin_code_and_keep_failed_checkpoint(tmp_path, monkeypatch):
    import io
    import os
    import shutil
    import subprocess
    import sys
    import time

    repo = tmp_path / 'source'
    repo.mkdir()
    (repo / 'runs').mkdir()
    (repo / 'runs/.gitkeep').write_text('')
    (repo / 'uv.lock').write_text('fixture')
    (repo / 'policy.py').write_text('version one')
    (repo / '.gitignore').write_text('.venv/\nruns/*\n!runs/.gitkeep\n')
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    for args in [['config', 'user.name', 'Fixture'], ['config', 'user.email', 'f@example.test'],
                 ['add', 'policy.py', 'uv.lock', '.gitignore', 'runs/.gitkeep'],
                 ['commit', '-m', 'fixture']]:
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    base = tmp_path / 'host'
    base.mkdir()
    shutil.copyfile(remote.__file__, base / 'idc_remote.py')
    fakebin = tmp_path / 'bin'
    fakebin.mkdir()
    uv = fakebin / 'uv'
    uv.write_text('#!' + sys.executable + '\n'
                  'import pathlib,sys,subprocess\n'
                  'a=sys.argv[1:]\n'
                  'if a[0]=="sync": pathlib.Path(".venv").mkdir(exist_ok=True)\n'
                  'else:\n'
                  ' a=a[2:]\n'
                  ' if a[0]=="python": a[0]=sys.executable\n'
                  ' sys.exit(subprocess.call(a))\n')
    uv.chmod(0o755)
    monkeypatch.setenv('PATH', str(fakebin) + os.pathsep + os.environ['PATH'])
    asset = 'd' * 64
    assets = base / 'assets' / asset
    assets.mkdir(parents=True)
    remote.dump(assets / 'asset_receipt.json', {})
    for job in ['one', 'two', 'changed']:
        program = ('from pathlib import Path; import time,sys; '
                   'p=Path("runs/result"); p.mkdir(); '
                   'p.joinpath("started").write_text(str(time.time())); '
                   'time.sleep(.25); '
                   'p.joinpath("failure.pt").write_bytes(b"retained"); '
                   'p.joinpath("ended").write_text(str(time.time())); '
                   + ('Path("policy.py").write_text("mutated"); ' if job == 'changed' else '')
                   + ('sys.exit(3)' if job == 'one' else 'sys.exit(0)'))
        request = {'job': job, 'revision': revision, 'asset_set': asset,
                   'repository': str(repo), 'argv': [sys.executable, '-c', program]}
        monkeypatch.setattr(sys, 'stdin', io.StringIO(__import__('json').dumps(request)))
        remote.server(base, ['submit'])
    deadline = time.monotonic() + 15
    while not all((base / 'jobs' / j / 'output/receipt.json').exists()
                  for j in ['one', 'two', 'changed']):
        assert time.monotonic() < deadline, [
            (p, p.read_text()) for p in base.glob('jobs/*/worker.log')]
        time.sleep(.05)
    intervals = []
    for job in ['one', 'two', 'changed']:
        output = base / 'jobs' / job / 'output'
        status = remote.read(output / 'remote_status.json')
        assert status['state'] == ('succeeded' if job == 'two' else 'failed')
        assert status['exit_code'] == (3 if job == 'one' else 0)
        if job == 'changed':
            assert 'tracked source' in status['error']
        assert status['revision'] == revision
        assert (output / 'runs/result/failure.pt').read_bytes() == b'retained'
        assert all((output / p).exists() for p in [
            'config.yaml', 'metrics.parquet', 'report.json', 'manifest.json', 'figures'])
        remote.verify_receipt(output, remote.read(output / 'receipt.json'))
        intervals.append([float((output / 'runs/result' / p).read_text())
                          for p in ['started', 'ended']])
    intervals.sort()
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:], strict=False))
    assert (repo / 'policy.py').read_text() == 'version one'


def test_seed_ignores_running_run_but_retains_failed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, 'ROOT', tmp_path)
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/frozen.bin').write_bytes(b'data')
    for name, status in [('live', 'running'), ('bad', 'failed')]:
        folder = tmp_path / 'runs' / name
        folder.mkdir(parents=True)
        remote.dump(folder / 'manifest.json', {'status': status})
        (folder / 'checkpoint.pt').write_bytes(b'weights')
    records = remote.asset_records()
    assert 'runs/bad/checkpoint.pt' in records
    assert not any(name.startswith('runs/live/') for name in records)


def test_symlink_output_is_failed_without_reading_target(tmp_path):
    outside = tmp_path.parent / 'private.bin'
    outside.write_bytes(b'secret')
    (tmp_path / 'escape').symlink_to(outside)
    remote.finalize(tmp_path, {'job': 'j', 'revision': 'a' * 40}, 0, 'succeeded')
    assert remote.read(tmp_path / 'remote_status.json')['state'] == 'failed'
    receipt = remote.read(tmp_path / 'receipt.json')
    assert 'escape' not in receipt
    remote.verify_receipt(tmp_path, receipt)


@pytest.mark.resume
def test_orphan_preserves_partial_run_and_publishes_interruption(tmp_path, monkeypatch):
    import io
    import sys

    job = tmp_path / 'jobs/j'
    (job / 'output').mkdir(parents=True)
    partial = job / 'workspace/runs/partial'
    partial.mkdir(parents=True)
    (partial / 'trace.jsonl').write_text('evidence')
    request = {'job': 'j', 'revision': 'a' * 40, 'asset_set': 'b' * 64, 'argv': ['make']}
    remote.dump(job / 'request.json', request)
    remote.dump(job / 'pid.json', {'pid': 12345})
    remote.dump(job / 'output/remote_status.json', {**request, 'state': 'running'})
    monkeypatch.setattr(remote, 'process_alive', lambda *args: False)
    monkeypatch.setattr(sys, 'stdout', io.StringIO())
    remote.server(tmp_path, ['status', 'j'])
    assert (job / 'output/runs/partial/trace.jsonl').read_text() == 'evidence'
    assert remote.read(job / 'output/remote_status.json')['state'] == 'interrupted'
    remote.verify_receipt(job / 'output', remote.read(job / 'output/receipt.json'))


def test_real_rsync_returns_checkpoint_and_replaces_same_size_corruption(tmp_path):
    import os
    import shutil

    if not shutil.which('rsync'):
        pytest.skip('rsync not installed')
    source = tmp_path / 'source'
    target = tmp_path / 'target'
    source.mkdir()
    target.mkdir()
    (source / 'checkpoint.pt').write_bytes(b'correct')
    remote.finalize(source, {'job': 'j', 'revision': 'a' * 40}, 0, 'succeeded')
    remote.rsync(str(source) + '/', str(target) + '/')
    remote.verify_receipt(target, remote.read(target / 'receipt.json'))
    checkpoint = target / 'checkpoint.pt'
    checkpoint.write_bytes(b'corrupt')
    timestamp = (source / 'checkpoint.pt').stat().st_mtime
    os.utime(checkpoint, (timestamp, timestamp))
    remote.rsync(str(source) + '/', str(target) + '/')
    remote.verify_receipt(target, remote.read(target / 'receipt.json'))
    assert checkpoint.read_bytes() == b'correct'
