#!/usr/bin/env python3
"""Independent consistency checks and tabulation for a completed precision campaign."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(records):
    protocol = json.loads((records/'protocol.json').read_text())
    artifacts = pathlib.Path(protocol['artifact_directory'])
    assert digest(artifacts/'ternary_lab') == protocol['binary_sha256']
    assert digest(records/'runner.py') == protocol['runner_sha256']
    for name, expected in protocol['source_sha256'].items():
        assert digest(artifacts/'source'/name) == expected
    events = [json.loads(line) for line in (records/'events.jsonl').read_text().splitlines()]
    assert not any(e['state']=='FAILED' for e in events)
    assert sum(e['state']=='COMPLETE' for e in events)==1
    table = []
    rows = {}
    models = {}
    evaluation_identity = None
    for policy, (precision, budget) in protocol['variants'].items():
        for seed in protocol['seeds']:
            name = f'{policy}-{seed}'
            report_path = records/f'{name}.json'
            r = json.loads(report_path.read_text())
            receipt = json.loads((records/f'{name}.verification.json').read_text())
            ckpt_path = artifacts/f'{name}.checkpoint.json'
            assert digest(report_path) == receipt['report_sha256']
            assert digest(ckpt_path) == receipt['checkpoint_sha256']
            c = json.loads(ckpt_path.read_text())
            assert receipt['arithmetic_and_report']=='PASS'
            assert receipt['correct']==r['correct'] and receipt['loss_sum']==r['loss_sum']
            assert r['metrics']==c['engine']['metrics'] and r['config']==c['engine']['config']
            assert r['seed']==seed and r['metrics']['steps']==20000
            assert r['evaluation_count']==4000 and r['evaluation_offset']==1000
            identity = (r['data_digest'], r['evaluation_digest'], r['implementation_digest'])
            if evaluation_identity is None: evaluation_identity=identity
            assert identity==evaluation_identity
            expected={'route':'Backprop','history':'Statistics','precision':precision.capitalize(),
                      'hysteresis':True,'simple':False,'threshold':0,'budget':budget*101632,
                      'replay_limit':0,'coordinates':4096,'incremental':True}
            assert r['config']==expected and r['replay_examples']==0 and r['accumulator_clips']==0
            assert [e['state'] for e in events if e['run']==name]==['STARTED','VERIFIED']
            rows[policy,seed]=r
            models[policy,seed]=c['engine']['model']
            table.append({'policy':policy,'seed':seed,'accuracy_percent':r['correct']/40,
                'loss_sum':r['loss_sum'],'mean_loss':r['loss_sum']/4000,
                'contraction_terms':r['metrics']['contraction_terms'],
                'state_bytes':r['optimizer_replay_allocated_bytes'],
                'peak_state_bytes':r['metrics']['peak_state_bytes'],
                **{k:r['metrics'][k] for k in ['transitions','reversals','precision_grows','precision_shrinks','rounded_small']},
                'rescalings':r['accumulator_rescalings'],'clips':r['accumulator_clips'],
                **{f'banks_{k}':v for k,v in receipt['final_banks'].items()}})
    assert len(table)==25
    existing=json.loads((records/'summary.json').read_text())
    contrasts={}
    for policy in protocol['variants']:
        differences=[(rows[policy,s]['loss_sum']-rows['adaptive24',s]['loss_sum'])/4000 for s in range(1,6)]
        mean=sum(differences)/5
        spread=2.776445105*math.sqrt(sum((x-mean)**2 for x in differences)/4/5)
        ci=[mean-spread,mean+spread]
        item=existing['methods'][policy]
        rr=[rows[policy,s] for s in range(1,6)]
        accuracy=[r['correct']/40 for r in rr]
        average=sum(accuracy)/5
        checks={'accuracy_percent':average,
            'accuracy_sd_pp':math.sqrt(sum((x-average)**2 for x in accuracy)/4),
            'mean_loss':sum(r['loss_sum']/4000 for r in rr)/5,
            'mean_work':sum(r['metrics']['contraction_terms'] for r in rr)/5,
            'mean_state_bytes':sum(r['optimizer_replay_allocated_bytes'] for r in rr)/5,
            'mean_growth':sum(r['metrics']['precision_grows'] for r in rr)/5,
            'mean_rescalings':sum(r['accumulator_rescalings'] for r in rr)/5,
            'mean_reversals':sum(r['metrics']['reversals'] for r in rr)/5}
        assert all(math.isclose(value,item[key],abs_tol=1e-8) for key,value in checks.items())
        assert math.isclose(mean,item['loss_difference_vs_adaptive24'],abs_tol=1e-8)
        assert all(math.isclose(x,y,abs_tol=1e-8) for x,y in zip(ci,item['paired_t_95_ci']))
        contrasts[policy]={'loss_difference':mean,'paired_t_95_ci':ci,
            'accuracy_delta_pp':sum((rows[policy,s]['correct']-rows['adaptive24',s]['correct'])/40 for s in range(1,6))/5}
    result={'analyzer_sha256':digest(pathlib.Path(__file__)), 'consistency_checks':'PASS','verified_runs':25,'contrasts_vs_adaptive24':contrasts,
        'adaptive_budget_models_identical':all(models['adaptive12',s]==models['adaptive24',s] for s in range(1,6)),
        'adaptive24_fixed24_models_identical':all(models['adaptive24',s]==models['fixed24',s] for s in range(1,6)),
        'fixed24_integer32_models_identical':all(models['fixed24',s]==models['integer32',s] for s in range(1,6)),
        'qualification':'exploratory precision-policy ablation; no matched-work or release qualification'}
    with (records/'independent-analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    with (records/'runs.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# EAT-O precision ablation results','',
        'All 25 runs completed and passed independent checkpoint evaluation and consistency checks.',
        'Accuracy is mean ± sample standard deviation over five seeds; work is counted contraction terms.',
        'This is exploratory evaluation on reused validation data, with no matched-work qualification.','',
        '| Policy | Accuracy | Mean loss | Mean work | Mean final state bytes | Loss difference vs adaptive24, 95% CI |',
        '|---|---:|---:|---:|---:|---:|']
    for policy,item in existing['methods'].items():
        ci=item['paired_t_95_ci']
        lines.append(f"| {policy} | {item['accuracy_percent']:.3f}% ± {item['accuracy_sd_pp']:.3f} | {item['mean_loss']:.3f} | {item['mean_work']:,.1f} | {item['mean_state_bytes']:,.1f} | {item['loss_difference_vs_adaptive24']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines += ['', '## Every run', '', '| Policy | Seed | Accuracy | Mean loss | Work | State bytes | Growth | Rescalings |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['policy']} | {row['seed']} | {row['accuracy_percent']:.3f}% | {row['mean_loss']:.3f} | {row['contraction_terms']:,} | {row['state_bytes']:,} | {row['precision_grows']} | {row['rescalings']} |")
    lines += ['', '## Evidence', '',
        '- [Frozen protocol](protocol.json), [per-run CSV](runs.csv), [event log](events.jsonl).',
        '- [Campaign summary](summary.json), [independent analysis](independent-analysis.json).',
        '- Each run has a command, complete JSON report, and verification receipt in this directory.',
        '- Confidence intervals are paired t intervals with four degrees of freedom, without multiplicity adjustment.',
        '- Fixed modes also differ from adaptive in exponent refinement; this does not isolate mantissa width alone.',
        '- Actual state allocation differs despite the common 24-byte cap. Replay is disabled in every run.',
        '- Large checkpoints remain local under the artifact directory in the protocol; compact records here are durable.', '']
    with (records/'results.md').open('x') as f:f.write('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('records',type=pathlib.Path)
    analyze(p.parse_args().records)
