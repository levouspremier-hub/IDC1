"""Pinned SSH jobs with immutable inputs and full, verified artifact retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / '.idc_remote.json'
TERMINAL = {'succeeded', 'failed', 'interrupted'}


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', value):
        raise ValueError('Invalid job/host identifier')
    return value


def validate_request(request):
    identifier(request['job'])
    if not re.fullmatch(r'[a-f0-9]{40}', request['revision']):
        raise ValueError('A full immutable Git SHA is required')
    argv = request['argv']
    if not isinstance(argv, list) or not argv or any(
        not isinstance(x, str) or '\0' in x for x in argv
    ):
        raise ValueError('Command must be a nonempty argv list')


def make_receipt(folder, *, skip_symlinks=False):
    result = {}
    for p in sorted(Path(folder).rglob('*')):
        if p.is_symlink():
            if skip_symlinks:
                continue
            raise ValueError('Output symlink is not transferable: ' + str(p))
        if p.is_file() and p.name not in {'receipt.json', 'remote_status.json.tmp'}:
            result[str(p.relative_to(folder))] = {'sha256': sha(p), 'size': p.stat().st_size}
    return result


def verify_receipt(folder, receipt):
    folder = Path(folder).resolve()
    for name, record in receipt.items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or not p.parts:
            raise ValueError('Unsafe receipt path: ' + name)
        p = folder / p
        if not p.resolve().is_relative_to(folder) or p.is_symlink():
            raise ValueError('Unsafe receipt symlink: ' + name)
        if not p.is_file() or p.stat().st_size != record['size'] or sha(p) != record['sha256']:
            raise ValueError('Incomplete or corrupt file: ' + name)


def check_destination(folder, revision):
    folder = Path(folder)
    status = folder / 'remote_status.json'
    if status.exists():
        if read(status)['revision'] != revision:
            raise ValueError('Destination belongs to a different revision')
    elif folder.exists() and any(folder.iterdir()):
        raise ValueError('Refusing to overwrite unrelated local files')


def finalize(output, request, exit_code, state, error=None):
    dump(Path(output) / 'remote_status.json', {
        'job': request['job'], 'revision': request['revision'], 'state': state,
        'exit_code': exit_code, 'error': error, 'finished_at': time.time(),
    })
    try:
        receipt = make_receipt(output)
    except ValueError as exc:
        status = read(Path(output) / 'remote_status.json')
        status.update(state='failed', error=str(exc))
        dump(Path(output) / 'remote_status.json', status)
        for name in ['report.json', 'manifest.json']:
            path = Path(output) / name
            if path.exists():
                value = read(path)
                value.update(state='failed', status='failed', error=str(exc),
                             failure_classification=str(exc))
                dump(path, value)
        dump(Path(output) / 'transfer_failure.json', {'reason': str(exc)})
        receipt = make_receipt(output, skip_symlinks=True)
    dump(Path(output) / 'receipt.json', receipt)


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def capture(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def remote_command(base, argv):
    if not Path(base).is_absolute() or '\n' in base:
        raise ValueError('Remote root must be an absolute Linux path')
    return shlex.join(['python3', str(Path(base) / 'idc_remote.py'), '--server', base, *argv])


def ssh(config, argv, payload=None):
    return capture(['ssh', '-o', 'BatchMode=yes', config['host'],
                    remote_command(config['remote_root'], argv)],
                   input=None if payload is None else json.dumps(payload))


def rsync(source, destination, *extra):
    # No --delete; failed copies remain resumable. Both Macs' rsync 2.6 and Linux supported.
    run(['rsync', '-rtc', '--partial', *extra, source, destination])


def asset_records():
    records = {}
    folders = [ROOT / 'data']
    for p in (ROOT / 'runs').iterdir():
        if p.is_dir() and (p / 'manifest.json').exists():
            if read(p / 'manifest.json').get('status') in {'running', 'queued'}:
                continue  # Never snapshot a live run.
            folders.append(p)
    for folder in folders:
        for p in sorted(folder.rglob('*')):
            if '__pycache__' in p.parts or p.name == '.DS_Store':
                continue
            if p.is_symlink():
                raise ValueError('Input symlink is unsupported: ' + str(p))
            if p.is_file():
                if '\n' in str(p.relative_to(ROOT)):
                    raise ValueError('Input filename contains newline')
                records[str(p.relative_to(ROOT))] = {'sha256': sha(p), 'size': p.stat().st_size}
    return records


def seed(config):
    records = asset_records()
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    ssh(config, ['asset-init', digest])
    listing = ROOT / '.idc_asset_files'
    listing.write_text('\n'.join(records) + '\n')
    try:
        rsync(str(ROOT) + '/', config['host'] + ':' + config['remote_root']
              + '/assets/' + digest + '/', '--files-from=' + str(listing))
    finally:
        listing.unlink()
    # Recheck after transfer so a local mutation cannot become a valid asset set.
    verify_receipt(ROOT, records)
    ssh(config, ['asset-finish', digest], records)
    config['asset_set'] = digest
    dump(CONFIG, config)
    print('Verified asset set', digest, 'files', len(records))


def config_load():
    config = read(CONFIG)
    identifier(config['host'])  # SSH alias only, no options or shell fragments.
    remote_command(config['remote_root'], [])
    if not re.fullmatch(r'/[A-Za-z0-9_/.-]+', config['remote_root']):
        raise ValueError('Use a remote root without spaces or shell metacharacters')
    return config


def server(base, argv):
    base = Path(base)
    action = argv[0]
    if action == 'doctor':
        print(json.dumps({'platform': sys.platform, 'python': sys.version,
                          'git': shutil.which('git'), 'rsync': shutil.which('rsync'),
                          'uv': shutil.which('uv') or str(Path.home() / '.local/bin/uv'),
                          'root': str(base), 'cpu_count': os.cpu_count()}))
    elif action == 'asset-init':
        digest = argv[1]
        if not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('Invalid asset set')
        folder = base / 'assets' / digest
        if (folder / 'asset_receipt.json').exists():
            raise ValueError('Asset set already sealed; select it using configure --asset-set')
        folder.mkdir(parents=True, exist_ok=True)
    elif action == 'asset-finish':
        digest = argv[1]
        if not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('Invalid asset set')
        records = json.load(sys.stdin)
        assert hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() == digest
        folder = base / 'assets' / digest
        verify_receipt(folder, records)
        dump(folder / 'asset_receipt.json', records)
        print(json.dumps({'asset_set': digest, 'verified_files': len(records)}))
    elif action == 'submit':
        request = json.load(sys.stdin)
        validate_request(request)
        digest = request['asset_set']
        if not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('Invalid asset set')
        if not (base / 'assets' / digest / 'asset_receipt.json').exists():
            raise ValueError('Seed and verify inputs before submitting')
        job = base / 'jobs' / request['job']
        job.mkdir(parents=True, exist_ok=False)
        output = job / 'output'
        output.mkdir()
        dump(job / 'request.json', request)
        dump(output / 'remote_status.json', {**request, 'state': 'queued'})
        shutil.copyfile(base / 'idc_remote.py', job / 'controller.py')
        with (job / 'worker.log').open('ab') as log:
            proc = subprocess.Popen(
                [sys.executable, str(job / 'controller.py'), '--server', str(base),
                 'worker', request['job']], stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True,
            )
        dump(job / 'pid.json', {'pid': proc.pid})
        print(json.dumps({'job': request['job'], 'pid': proc.pid, 'state': 'queued'}))
    elif action == 'worker':
        worker(base, identifier(argv[1]))
    elif action == 'status':
        job = base / 'jobs' / identifier(argv[1])
        status = read(job / 'output/remote_status.json')
        if (status['state'] not in TERMINAL or not (
            job / 'output/receipt.json').exists()) and (job / 'pid.json').exists():
            pid = read(job / 'pid.json')['pid']
            alive = process_alive(pid, job / 'controller.py')
            if not alive:
                gather_runs(job)
                status['state'] = 'interrupted'
                status['error'] = 'Worker disappeared; inspect retained outputs, no automatic rerun'
                output = job / 'output'
                request = read(job / 'request.json')
                dump(output / 'config.yaml', request)
                dump(output / 'report.json', status)
                dump(output / 'manifest.json', {
                    'run_id': request['job'], 'revision': request['revision'],
                    'status': 'interrupted', 'data_hash': request['asset_set'],
                    'dependency_lock_hash': None, 'scenario_hash': None, 'seed': None,
                    'command': request['argv'], 'failure_classification': status['error'],
                })
                (output / 'figures').mkdir(exist_ok=True)
                if not (output / 'metrics.parquet').exists():
                    dump(output / 'metrics_unavailable.json', {'reason': status['error']})
                if (job / 'worker.log').exists():
                    shutil.copyfile(job / 'worker.log', output / 'worker.log')
                finalize(output, request, None, 'interrupted', status['error'])
        if status['state'] in TERMINAL and not (job / 'output/receipt.json').exists():
            status['state'] = 'finalizing'
        print(json.dumps(status))
    else:
        raise ValueError('Unknown server action')


def process_alive(pid, controller):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    proc = Path('/proc') / str(pid) / 'cmdline'
    if proc.exists():
        return str(controller).encode() in proc.read_bytes().split(b'\0')
    return True


def gather_runs(job):
    work = job / 'workspace'
    inherited = set(read(job / 'inherited_runs.json')) if (
        job / 'inherited_runs.json').exists() else set()
    target = job / 'output/runs'
    target.mkdir(exist_ok=True)
    for p in (work / 'runs').iterdir() if (work / 'runs').exists() else []:
        if p.is_dir() and p.name not in inherited and p.name != '__pycache__':
            if (target / p.name).exists():
                raise ValueError('Artifact destination already exists: ' + p.name)
            shutil.move(str(p), target / p.name)


def task_environment(uv, output, job_id, revision):
    return {**os.environ, 'PATH': str(Path(uv).parent) + os.pathsep
            + os.environ.get('PATH', ''), 'IDC_JOB_OUTPUT': str(output),
            'IDC_JOB_ID': job_id, 'IDC_JOB_REVISION': revision}


def worker(base, job_id):
    import fcntl
    job = base / 'jobs' / job_id
    request = read(job / 'request.json')
    output = job / 'output'
    state, code, error = 'failed', None, None
    uv = shutil.which('uv') or str(Path.home() / '.local/bin/uv')
    work = job / 'workspace'
    with (base / 'execution.lock').open('a') as lock, (output / 'console.log').open('ab') as log:
        fcntl.flock(lock, fcntl.LOCK_EX)  # One resource-sensitive job, including setup.
        dump(output / 'remote_status.json', {**request, 'state': 'preparing'})
        try:
            repo = base / 'repo.git'
            if not repo.exists():
                run(['git', 'clone', '--bare', request['repository'], str(repo)],
                    stdout=log, stderr=log)
            run(['git', '--git-dir', str(repo), 'fetch', 'origin', request['revision']],
                stdout=log, stderr=log)
            run(['git', '--git-dir', str(repo), 'worktree', 'add', '--detach', str(work),
                 request['revision']], stdout=log, stderr=log)
            assert capture(['git', 'rev-parse', 'HEAD'], cwd=work) == request['revision']
            assets = base / 'assets' / request['asset_set']
            asset_manifest = read(assets / 'asset_receipt.json')
            verify_receipt(assets, asset_manifest)
            for name, record in asset_manifest.items():
                if name.startswith('data/manifest/') and (work / name).exists():
                    if sha(work / name) != record['sha256']:
                        raise ValueError('Asset manifest differs from pinned code: ' + name)
            # Real copies keep an errant task from mutating other jobs' inputs.
            for name in ['raw', 'processed']:
                src = assets / 'data' / name
                if src.exists():
                    shutil.copytree(src, work / 'data' / name, dirs_exist_ok=True)
            for p in (assets / 'runs').iterdir() if (assets / 'runs').exists() else []:
                if p.is_dir():
                    shutil.copytree(p, work / 'runs' / p.name)
            inherited = {p.name for p in (work / 'runs').iterdir() if p.is_dir()}
            dump(job / 'inherited_runs.json', sorted(inherited))
            run([uv, 'sync', '--frozen'], cwd=work, stdout=log, stderr=log)
            dump(output / 'remote_status.json', {**request, 'state': 'running'})
            dump(output / 'request.json', request)
            task_env = task_environment(uv, output, job_id, request['revision'])
            code = subprocess.call([uv, 'run', '--frozen', *request['argv']], cwd=work,
                                   stdout=log, stderr=log, env=task_env)
            state = 'succeeded' if code == 0 else 'failed'
        except Exception as exc:
            error = str(exc)
        finally:
            if work.exists():
                try:
                    gather_runs(job)
                    dirty = capture(['git', 'status', '--porcelain'], cwd=work)
                    if dirty:
                        state, error = 'failed', 'Task changed tracked source/config: ' + dirty
                except Exception as exc:
                    state, error = 'failed', 'Artifact/source check failed: ' + str(exc)
            # Logs and worker bookkeeping survive preflight/dependency failures as well.
            dump(output / 'report.json', {'state': state, 'exit_code': code, 'error': error})
            dump(output / 'config.yaml', request)  # JSON is valid YAML.
            dump(output / 'manifest.json', {
                'run_id': job_id, 'revision': request['revision'], 'status': state,
                'dependency_lock_hash': sha(work / 'uv.lock') if (work / 'uv.lock').exists()
                else None, 'data_hash': request['asset_set'], 'scenario_hash': None,
                'seed': request.get('seed'), 'command': request['argv'],
                'failure_classification': error,
            })
            (output / 'figures').mkdir(exist_ok=True)
            if (work / '.venv').exists():
                script = ('import pandas as pd; pd.DataFrame([{"exit_code":'
                          + repr(code) + '}]).to_parquet(' + repr(str(output / 'metrics.parquet'))
                          + ')')
                metric_code = subprocess.call(
                    [uv, 'run', '--frozen', 'python', '-c', script], cwd=work,
                    stdout=log, stderr=log,
                )
                if metric_code:
                    state, error = 'failed', 'Could not write metrics.parquet'
                    dump(output / 'metrics_unavailable.json', {'reason': error})
            else:
                dump(output / 'metrics_unavailable.json', {
                    'reason': 'Environment setup failed; no fabricated Parquet',
                })
            report = read(output / 'report.json')
            report.update(state=state, error=error)
            dump(output / 'report.json', report)
            manifest = read(output / 'manifest.json')
            manifest.update(status=state, failure_classification=error)
            dump(output / 'manifest.json', manifest)
            log.flush()
            if (job / 'worker.log').exists():
                shutil.copyfile(job / 'worker.log', output / 'worker.log')
            finalize(output, request, code, state, error)


def collect(config, job):
    status = json.loads(ssh(config, ['status', job]))
    folder = ROOT / 'runs' / ('remote_' + job)
    check_destination(folder, status['revision'])
    folder.mkdir(parents=True, exist_ok=True)
    rsync(config['host'] + ':' + config['remote_root'] + '/jobs/' + job + '/output/',
          str(folder) + '/')
    if status['state'] in TERMINAL:
        verify_receipt(folder, read(folder / 'receipt.json'))
        dump(folder / 'local_transfer_receipt.json', {
            'verified_at': time.time(), 'remote_status': status, 'all_files_verified': True,
        })
    return status


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--server':
        server(sys.argv[2], sys.argv[3:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    conf = sub.add_parser('configure')
    conf.add_argument('--host', required=True, help='SSH alias from ~/.ssh/config')
    conf.add_argument('--remote-root', required=True)
    conf.add_argument('--asset-set', default=None)
    sub.add_parser('doctor')
    sub.add_parser('seed')
    submit = sub.add_parser('submit')
    submit.add_argument('--job', required=True)
    submit.add_argument('--revision', default='HEAD')
    submit.add_argument('command', nargs=argparse.REMAINDER)
    for action in ['status', 'collect', 'watch']:
        cmd = sub.add_parser(action)
        cmd.add_argument('job')
        if action == 'watch':
            cmd.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    if args.action == 'configure':
        identifier(args.host)
        remote_command(args.remote_root, [])
        if not re.fullmatch(r'/[A-Za-z0-9_/.-]+', args.remote_root):
            raise ValueError('Use a remote root without spaces or shell metacharacters')
        dump(CONFIG, {'host': args.host, 'remote_root': args.remote_root,
                      'asset_set': args.asset_set})
        print('Local config saved; no passwords or keys stored')
        return
    config = config_load()
    if args.action == 'doctor':
        # Explicitly deploy only the control script, never the running checkout.
        run(['ssh', '-o', 'BatchMode=yes', config['host'],
             shlex.join(['mkdir', '-p', config['remote_root']])])
        rsync(str(Path(__file__).resolve()), config['host'] + ':'
              + config['remote_root'] + '/idc_remote.py')
        print(ssh(config, ['doctor']))
    elif args.action == 'seed':
        seed(config)
    elif args.action == 'submit':
        identifier(args.job)
        if capture(['git', 'status', '--porcelain'], cwd=ROOT):
            raise ValueError('Commit/push changes before submitting')
        revision = capture(['git', 'rev-parse', args.revision + '^{commit}'], cwd=ROOT)
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        request = {'job': args.job, 'revision': revision, 'argv': command,
                   'asset_set': config.get('asset_set'),
                   'repository': 'https://github.com/levouspremier-hub/IDC1.git'}
        validate_request(request)
        print(ssh(config, ['submit'], request))
        log_dir = ROOT / '.idc_remote_logs'
        log_dir.mkdir(exist_ok=True)
        with (log_dir / (args.job + '.log')).open('ab') as log:
            watcher = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), 'watch', args.job],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
            )
        print('Automatic full-return watcher PID', watcher.pid)
    else:
        job = identifier(args.job)
        if args.action == 'status':
            print(ssh(config, ['status', job]))
            return
        while True:
            try:
                status = collect(config, job)
            except (subprocess.CalledProcessError, OSError) as exc:
                if args.action != 'watch':
                    raise
                print('Connection/transfer unavailable; retrying: ' + str(exc), flush=True)
                time.sleep(max(5, args.interval))
                continue
            print(json.dumps(status, ensure_ascii=False), flush=True)
            if args.action == 'collect' or status['state'] in TERMINAL:
                break
            time.sleep(max(5, args.interval))


if __name__ == '__main__':
    main()
