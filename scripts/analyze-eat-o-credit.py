#!/usr/bin/env python3
"""Independently check the discrete-credit campaign and emit durable tables."""
import argparse
import csv
import hashlib
import json
import math
import pathlib


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(d):
    protocol=json.loads((d/'protocol.json').read_text())
    artifacts=pathlib.Path(protocol['artifact_directory'])
    assert sha(artifacts/'ternary_lab')==protocol['binary_sha256']
    assert sha(d/'runner.py')==protocol['runner_sha256']
    for path,digest in protocol['source_sha256'].items():assert sha(artifacts/'source'/path)==digest
    events=[json.loads(line) for line in (d/'events.jsonl').read_text().splitlines()]
    assert sum(e['state']=='COMPLETE' for e in events)==1
    assert not any(e['state']=='FAILED' for e in events)
    rows={};models={};table=[];identity=None
    keys=[(name,s) for name in ['backprop','probe'] for s in range(1,6)]+[('probe_reference',1)]
    for name,seed in keys:
        key=f'{name}-{seed}';r=json.loads((d/f'{key}.json').read_text())
        v=json.loads((d/f'{key}.verification.json').read_text())
        ck=artifacts/f'{key}.checkpoint.json'
        assert sha(d/f'{key}.json')==v['report_sha256'] and sha(ck)==v['checkpoint_sha256']
        assert v['arithmetic_and_report']=='PASS' and r['correct']==v['correct'] and r['loss_sum']==v['loss_sum']
        c=json.loads(ck.read_text());models[name,seed]=c['engine']['model']
        assert r['metrics']==c['engine']['metrics'] and r['config']==c['engine']['config']
        expected={'route':'Probe' if name.startswith('probe') else 'Backprop','history':'Statistics',
                  'precision':'Adaptive','hysteresis':True,'simple':False,'threshold':0,'budget':1219584,
                  'replay_limit':0,'coordinates':4096,'incremental':True,'raw_cache':name!='probe_reference'}
        assert r['config']==expected and r['seed']==seed and r['metrics']['steps']==20000
        assert r['evaluation_count']==4000 and r['evaluation_offset']==1000 and r['accumulator_clips']==0
        current=(r['data_digest'],r['evaluation_digest'],r['implementation_digest'])
        if identity is None:identity=current
        assert identity==current
        assert [e['state'] for e in events if e['run']==key]==['STARTED','VERIFIED']
        rows[name,seed]=r
        table.append({'route':name,'seed':seed,'accuracy_percent':r['correct']/40,'mean_loss':r['loss_sum']/4000,
                      'contraction_terms':r['metrics']['contraction_terms'],'state_bytes':r['optimizer_replay_allocated_bytes'],
                      **{k:r['metrics'][k] for k in ['transitions','reversals','precision_grows']}})
    assert models['probe',1]==models['probe_reference',1]
    diffs=[(rows['probe',s]['loss_sum']-rows['backprop',s]['loss_sum'])/4000 for s in range(1,6)]
    mean=sum(diffs)/5;half=2.776445105*math.sqrt(sum((x-mean)**2 for x in diffs)/4/5)
    ci=[mean-half,mean+half]
    summary=json.loads((d/'summary.json').read_text())
    assert math.isclose(mean,summary['methods']['probe']['loss_difference_vs_backprop'],abs_tol=1e-8)
    assert all(math.isclose(a,b,abs_tol=1e-8) for a,b in zip(ci,summary['methods']['probe']['paired_t_95_ci']))
    for name in ['backprop','probe']:
        rr=[rows[name,s] for s in range(1,6)];item=summary['methods'][name]
        assert math.isclose(sum(r['correct']/40 for r in rr)/5,item['accuracy_percent'],abs_tol=1e-8)
        assert math.isclose(sum(r['metrics']['contraction_terms'] for r in rr)/5,item['mean_work'],abs_tol=1e-5)
    result={'independent_checks':'PASS','verified_runs':11,'exact_probe_control_model':True,
            'probe_loss_difference':mean,'paired_t_95_ci':ci,
            'accuracy_difference_pp':sum((rows['probe',s]['correct']-rows['backprop',s]['correct'])/40 for s in range(1,6))/5,
            'analyzer_sha256':sha(pathlib.Path(__file__)),'qualification':'exploratory reused validation, no matched-work or release qualification'}
    with (d/'independent-analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    with (d/'runs.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# EAT-O exact discrete credit results','','All 11 checkpoint evaluations and the paired comparison passed independent checks.',
           'Same 20,000-example recipe; five paired seeds. Validation was reused. No matched-work qualification.','',
           '| Signal | Mean accuracy | SD, pp | Mean loss | Mean counted work |','|---|---:|---:|---:|---:|']
    for name,item in summary['methods'].items():
        lines.append(f"| {name} | {item['accuracy_percent']:.3f}% | {item['accuracy_sd_pp']:.3f} | {item['mean_loss']:.3f} | {item['mean_work']:,.1f} |")
    lines += ['',f'Probe minus backprop loss: {mean:.3f}; paired 95% t interval [{ci[0]:.3f}, {ci[1]:.3f}].',
              'The raw-cache and reference probe seed-1 models are identical. Intervals are exploratory, unadjusted for repeated experiments.',
              '', '## Every run', '', '| Route | Seed | Accuracy | Mean loss | Work | Transitions | Reversals |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['route']} | {r['seed']} | {r['accuracy_percent']:.3f}% | {r['mean_loss']:.3f} | {r['contraction_terms']:,} | {r['transitions']:,} | {r['reversals']:,} |")
    lines += ['', '[Protocol](protocol.json), [CSV](runs.csv), [events](events.jsonl), [control](control.json), [independent checks](independent-analysis.json).',
              'Each run also has a complete report, command and verification receipt. Checkpoints remain local under the protocol artifact path.','']
    with (d/'results.md').open('x') as f:f.write('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('records',type=pathlib.Path)
    main(p.parse_args().records)
