"""Record sequential checkpoint evaluations; run after training is stopped."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def summarize(results):
    n=len(results)
    return dict(episodes=n,successes=sum(bool(r['success']) for r in results),
                success_rate=sum(bool(r['success']) for r in results)/max(1,n),
                reached_grasp=sum(any(t['stage']>=1 for t in r['trace']) for r in results),
                reached_lift=sum(any(t['stage']>=2 for t in r['trace']) for r in results),
                reached_place=sum(any(t['stage']>=3 for t in r['trace']) for r in results),
                unsafe_table_episodes=sum(any('unsafe_table_contact' in t.get('termination',[]) for t in r['trace']) for r in results),
                mean_table_slide_m=sum(max((t['measurements'].get('slide_distance',0) for t in r['trace']),default=0) for r in results)/max(1,n),
                note='Stage visits are not mastery. Fixed-scene deterministic seeds do not establish generalization; inspect videos and contact warnings.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    p.add_argument('--output',default='outputs/four_stage/evaluation_batch')
    args=p.parse_args()
    active=subprocess.run(['pgrep','-af',r'^([^ ]*/)?python[^ ]* .*scripts/train.py'],capture_output=True,text=True)
    if active.stdout.strip():p.error('Training is running. Leave it running and evaluate later to avoid competing for GPU/RAM.')
    checkpoint=Path(args.checkpoint).resolve()
    if not checkpoint.is_file():p.error('Checkpoint not found')
    folder=Path(args.output)
    folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in args.seeds:
        output=folder/f'seed_{seed}.json'
        if output.exists() or output.with_suffix('.mp4').exists():p.error(f'Refusing to overwrite {output}')
        with output.with_suffix('.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('staged_task.py')),
                            '--mode','evaluate','--checkpoint',str(checkpoint),'--headless',
                            '--max-steps','1350','--seed',str(seed),'--record-video','--output',str(output)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        results.append(json.loads(output.read_text()))
    report=summarize(results);report['checkpoint']=str(checkpoint)
    (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
