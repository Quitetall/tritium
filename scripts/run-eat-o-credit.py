#!/usr/bin/env python3
"""Frozen discrete-credit ablation with durable per-run reports and journal events."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import pathlib
import platform
import shutil
import statistics
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
VARIANTS = {'backprop': ('adaptive', 12), 'probe': ('adaptive', 12)}


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    help_text = subprocess.check_output([str(a.binary.resolve()), '--help'], text=True)
    if '--raw-cache' not in help_text:
        raise RuntimeError('executable lacks --raw-cache; finish the build before starting')
    import blake3
    source_files = ['crates/tritium-train/examples/ternary_lab.rs',
                    *[f'crates/tritium-train/examples/ternary_lab/{n}.rs' for n in ('numeric','model','engine')]]
    expected = blake3.blake3(b''.join((ROOT/name).read_bytes() for name in source_files)).hexdigest()
    probe = json.loads(subprocess.check_output([str(a.binary.resolve()), '--steps', '0', '--eval-limit', '1'], cwd=ROOT, text=True))
    if probe['implementation_digest'] != expected:
        raise RuntimeError('executable/source mismatch; finish the build before starting')
    out = a.output.resolve()
    durable = a.records.resolve()
    out.mkdir(parents=True, exist_ok=False)
    durable.mkdir(parents=True, exist_ok=False)
    parent = a.parent.resolve()
    previous_protocol = json.loads((parent/'protocol.json').read_text())
    shutil.copy2(a.binary, out/'ternary_lab')
    source_hashes = {}
    for name in previous_protocol['source_sha256']:
        target = out/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
        source_hashes[name]=sha(target)
    shutil.copy2(__file__, durable/'runner.py')
    data = a.data.resolve()
    hashes = {name: sha(data/name) for name in previous_protocol['data_sha256']}
    assert hashes == previous_protocol['data_sha256']
    protocol = {
        'campaign': out.name, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'variants': VARIANTS, 'seeds': list(range(1, 6)), 'steps': 20000,
        'common': {'history': 'statistics', 'replay': 0, 'coordinates': 4096,
                   'hysteresis': True, 'threshold': 0, 'incremental': True, 'raw_cache': True},
        'evaluation_offset': 1000, 'evaluation_count': 4000, 'split': 'reused validation',
        'selection': 'none; report all variants', 'workers': 2,
        'primary_comparator': 'backprop', 'control': 'probe seed 1 without raw cache; identical model required', 'interval': 'paired 95% t, df=4; exploratory, no multiplicity adjustment',
        'source_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_dirty': bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
        'binary_sha256': sha(out/'ternary_lab'), 'source_sha256': source_hashes,
        'runner_sha256': sha(durable/'runner.py'), 'data_sha256': hashes,
        'machine': platform.platform(), 'cpu': subprocess.check_output(['lscpu'],text=True),
        'python': sys.version, 'artifact_directory': str(out), 'checkpoint_retention': 'local target only; hashes durable',
        'advance_to_language_model': False,
    }
    save(durable/'protocol.json', protocol)
    save(out/'protocol.json', protocol)
    lock = threading.Lock()

    def event(run, state, **details):
        item = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'campaign': out.name, 'run': run, 'state': state, **details}
        with lock:
            with (durable/'events.jsonl').open('a') as f:
                f.write(json.dumps(item)+'\n'); f.flush(); os.fsync(f.fileno())
            with a.journal.open('a') as f:
                text = f"- {item['utc']} — `{out.name}/{run}` **{state}**"
                if state == 'VERIFIED':
                    text += f"; accuracy {details['accuracy_percent']:.3f}%; mean loss {details['mean_loss']:.3f}; work {details['work']:,}; state {details['state_bytes']:,} bytes; growth {details['growth']}; clips {details['clips']}"
                elif state == 'FAILED': text += '; ' + details['error'].replace('\n', ' ')
                f.write(text+'.\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(item), flush=True)

    def run(name, seed):
        key = f'{name}-{seed}'
        precision, budget = VARIANTS.get(name, ('adaptive',12))
        report = durable/f'{key}.json'
        ckpt = out/f'{key}.checkpoint.json'
        cmd = [str(out/'ternary_lab'),'--data',str(data),'--history','statistics','--replay','0',
               '--coordinates','4096','--hysteresis','yes','--threshold','0','--incremental','yes',
               '--route','probe' if name.startswith('probe') else 'backprop',
               '--raw-cache','no' if name=='probe_reference' else 'yes',
               '--precision',precision,'--budget',str(budget*101632),'--seed',str(seed),
               '--steps','20000','--eval-offset','1000','--eval-limit','4000',
               '--report',str(report),'--checkpoint',str(ckpt)]
        save(durable/f'{key}.command.json', cmd)
        event(key, 'STARTED')
        try:
            with (out/f'{key}.log').open('x') as f:
                subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            result = subprocess.run([sys.executable,str(out/'source/scripts/verify-ternary-lab.py'),
                        str(ckpt),str(report),'--data',str(data)],cwd=ROOT,text=True,capture_output=True,check=True)
            receipt = json.loads(result.stdout)
            assert receipt['arithmetic_and_report'] == 'PASS'
            r = json.loads(report.read_text())
            c = json.loads(ckpt.read_text())
            assert r['config']['precision'].lower() == precision and r['config']['budget'] == budget*101632
            assert r['metrics']['steps']==20000 and r['replay_examples']==0
            if name == 'backprop':
                old = json.loads((parent/f'eval-hysteresis-{seed}.checkpoint.json').read_text())
                assert c['engine']['model'] == old['engine']['model'], 'backprop cache changed model'
            receipt.update({'report_sha256': sha(report),'checkpoint_sha256': sha(ckpt),
                            'binary_sha256': protocol['binary_sha256'],
                            'final_banks': {str(width): sum(b['bank']['digits']==width for b in c['engine']['blocks']) for width in (0,8,16,24)}})
            save(durable/f'{key}.verification.json',receipt)
            event(key,'VERIFIED',accuracy_percent=r['correct']/40,mean_loss=r['loss_sum']/4000,
                  work=r['metrics']['contraction_terms'],state_bytes=r['optimizer_replay_allocated_bytes'],
                  growth=r['metrics']['precision_grows'],clips=r['accumulator_clips'])
            return r
        except Exception as exc:
            event(key,'FAILED',error=str(exc))
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {(name,s): pool.submit(run,name,s) for name in VARIANTS for s in range(1,6)}
        control_job=pool.submit(run,'probe_reference',1)
        rows = {key: job.result() for key,job in jobs.items()}
        control=control_job.result()
        candidate=json.loads((out/'probe-1.checkpoint.json').read_text())['engine']['model']
        reference=json.loads((out/'probe_reference-1.checkpoint.json').read_text())['engine']['model']
        assert candidate==reference, 'raw probe cache changed model'
        save(durable/'control.json',{'identical_model':True,'reference_work':control['metrics']['contraction_terms'], 'raw_work':rows['probe',1]['metrics']['contraction_terms']})
    summaries = {}
    for name in VARIANTS:
        rr = [rows[name,s] for s in range(1,6)]
        acc = [r['correct']/40 for r in rr]
        diffs = [(rows[name,s]['loss_sum']-rows['backprop',s]['loss_sum'])/4000 for s in range(1,6)]
        mean = statistics.mean(diffs)
        half = 2.776445105*statistics.stdev(diffs)/math.sqrt(5)
        summaries[name] = {'accuracy_percent': statistics.mean(acc),'accuracy_sd_pp': statistics.stdev(acc),
            'mean_loss': statistics.mean(r['loss_sum']/4000 for r in rr),
            'mean_work': statistics.mean(r['metrics']['contraction_terms'] for r in rr),
            'mean_state_bytes': statistics.mean(r['optimizer_replay_allocated_bytes'] for r in rr),
            'mean_growth': statistics.mean(r['metrics']['precision_grows'] for r in rr),
            'mean_rescalings': statistics.mean(r['accumulator_rescalings'] for r in rr),
            'mean_reversals': statistics.mean(r['metrics']['reversals'] for r in rr),
            'loss_difference_vs_backprop': mean, 'paired_t_95_ci': [mean-half,mean+half]}
    save(durable/'summary.json', {'methods':summaries,'advance_to_language_model':False})
    event('campaign','COMPLETE',verified_runs=len(rows)+1)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','parent','data','output','records','journal'):p.add_argument('--'+key,type=pathlib.Path,required=True)
    main(p.parse_args())
