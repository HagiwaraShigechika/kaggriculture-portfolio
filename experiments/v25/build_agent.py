"""Build a standalone v25 agent with explicit hooks in the frozen v24 source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA256 = '37512a351de225d69e7c1421671e659b7adaa12d0326aeefbee95a57e733a3a7'


def replace_exact(source, old, new, count=1):
    found = source.count(old)
    if found != count:
        raise ValueError(f'hook anchor count {found}, expected {count}: {old[:100]!r}')
    return source.replace(old, new)


def build(weights=None, mode='neutral'):
    path = ROOT / 'strategy_v24/release_candidate.py'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != BASELINE_SHA256:
        raise ValueError('v24 baseline hash changed')
    source = data.decode()
    source = replace_exact(source, "        state['day6']=True\n",
                           "        state['route']=_v25_route_choice(observation,state['route'])\n        state['day6']=True\n")
    source = replace_exact(source, '    if _cxtb_expected_revenue(obs) < _CXTB_MIN_REVENUE:\n',
                           '    if not _v25_tomato_economic_gate(obs):\n')
    source = replace_exact(source, "    _v9_hz=_V9_ITEM_HZ.get(int(obs['player']))\n",
                           "    _v9_hz=_v25_reservation_horizons(obs,_V9_ITEM_HZ.get(int(obs['player'])))\n")
    source = replace_exact(source, '    _v9_hz = _V9_ITEM_HZ.get(int(obs["player"]))\n',
                           '    _v9_hz = _v25_reservation_horizons(obs, _V9_ITEM_HZ.get(int(obs["player"])))\n')
    source = replace_exact(source, 'if prices.get(i, 0) <= V9_RACEGATE_BASE[i] + V9_RACEGATE_MARGIN}',
                           'if prices.get(i, 0) <= V9_RACEGATE_BASE[i] + V9_RACEGATE_MARGIN and not _v25_reservation_learned(obs, i)}')
    source = replace_exact(source, '    for off in range(1,_ADV_LOOK+1):\n',
                           '    for off in range(1,_v25_advance_scan(obs,_ADV_LOOK)+1):\n')
    source = replace_exact(source, "            if o[0]=='SELL' and o[1] in _ADV_ITEMS:\n",
                           "            if o[0]=='SELL' and o[1] in _ADV_ITEMS and _v25_advance_include(obs,o[1],off):\n")
    source = replace_exact(source,
                           "        if n<1:continue\n        if not _v20_allow(obs,item,n,min(t for t,it,q in plan if it==item),protected):continue\n",
                           "        n=_v25_advance_quantity(obs,item,n)\n        if n<1:continue\n        if not _v25_advance_allowed(obs,item,n,min(t for t,it,q in plan if it==item),protected):continue\n")
    from farm_hooks import install
    source = install(source, replace_exact)
    controller = (ROOT / 'strategy_v25/controller.py').read_text()
    source += '\n\n' + controller + '\n\n' + (ROOT / 'strategy_v25/farm_controls.py').read_text()
    model = weights.get('model', weights) if weights is not None else None
    if model is not None:
        required = {'w1', 'b1', 'w2', 'b2', 'wp', 'bp', 'wv', 'bv'}
        if not required.issubset(model):
            raise ValueError('missing model arrays: ' + str(sorted(required - set(model))))
    source += '\nV25_MODE = ' + repr(mode) + '\n'
    if model is not None:
        source += 'V25_MODEL = json.loads(' + repr(json.dumps(model, separators=(',', ':'), allow_nan=False)) + ')\n'
    source += "agent = globals().pop('agent')\n"
    ast.parse(source, feature_version=(3, 9))
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('neutral', 'policy'), default='neutral')
    args = parser.parse_args()
    weights = json.loads(args.weights.read_text()) if args.weights else None
    source = build(weights, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source)
    print(json.dumps({'output': str(args.output), 'mode': args.mode,
                      'baseline_sha256': BASELINE_SHA256,
                      'sha256': hashlib.sha256(source.encode()).hexdigest(),
                      'bytes': len(source.encode())}))


if __name__ == '__main__':
    main()
