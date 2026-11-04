"""Reproducible full-season checks against the official game interpreter."""
import argparse
import contextlib
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path
from evaluation_paths import input_path, output_directory


def load(filename):
    spec=importlib.util.spec_from_file_location(Path(filename).stem,filename)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.agent

def run(filename,opponent,seed,save=False,seat=0,output_dir=None):
    # Matrix workers choose their own backend; load the official engine only here.
    with open('/tmp/kaggriculture-import.log', 'w') as log, contextlib.redirect_stdout(log):
        from kaggle_environments import make
    filename=str(input_path(filename))
    opponent=str(input_path(opponent)) if opponent.endswith('.py') else opponent
    directory=Path(output_dir) if output_dir is not None else output_directory('evaluate')
    fn=load(filename)
    opp=load(opponent) if opponent.endswith('.py') else opponent
    elapsed=[]
    def timed(obs,cfg):
        t=time.perf_counter(); a=fn(obs,cfg); elapsed.append(time.perf_counter()-t)
        return a
    env=make('kaggriculture',configuration={'seed':seed},debug=True)
    agents=[timed,opp] if seat==0 else [opp,timed]
    env.run(agents)
    result={'agent':filename,'opponent':opponent,'seed':seed,'seat':seat,
            'money':[s.reward for s in env.steps[-1]],
            'status':[s.status for s in env.steps[-1]],
            'max_action_seconds':max(elapsed),'mean_action_seconds':sum(elapsed)/len(elapsed)}
    daily=[]
    for index,st in enumerate(env.steps):
        o=st[seat].observation
        recorded_step=o.get('step',index)
        if o.hour==0 or recorded_step==718 or recorded_step==719:
            f=o.farms[seat]
            count=Counter(t.get('crop',t.get('animal',t.get('kind'))) for row in f['tiles'] for t in row if isinstance(t,dict))
            daily.append({'day':o.day,'step':recorded_step,'money':f['money'],'crops':dict(count),'prices':o.market.prices,
                          'shed':dict(o.private.shed),'held':sum(sum(v.values()) for v in o.private.inventories)})
    result['daily']=daily
    out=directory/f'{Path(filename).stem}_{Path(opponent).stem}_{seed}_{seat}.json'
    with out.open('x') as stream: stream.write(json.dumps(result,indent=2))
    if save:
        with out.with_suffix('.replay.json').open('x') as stream: stream.write(json.dumps(env.toJSON()))
    print(json.dumps({k:v for k,v in result.items() if k!='daily'}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--agent',default='main.py');p.add_argument('--opponent',default='starter');p.add_argument('--seeds',default='1');p.add_argument('--save',action='store_true');p.add_argument('--seat',type=int,default=0)
    p.add_argument('--output-dir',help='New output directory; defaults to a unique results/runs/ directory')
    args=p.parse_args()
    directory=output_directory('evaluate',args.output_dir)
    print(f'Output: {directory}',flush=True)
    for seed in map(int,args.seeds.split(',')):run(args.agent,args.opponent,seed,args.save,args.seat,directory)
