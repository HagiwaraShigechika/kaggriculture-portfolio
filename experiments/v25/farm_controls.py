"""Observable, precommit farm decisions embedded into the frozen v24 pipeline.

Every helper runs before the owning layer reserves seeds, workers, or animals.
The first144 turns and the final day keep their native farm decisions.
"""
V25_FARM_DIAGNOSTICS = {}
_V25_FARM_LAST = {}


def _v25_farm_enabled(obs):
    return V25_MODE == 'policy' and 144 <= int(obs['step']) < 696


def _v25_farm_hook(obs, head, legal, context):
    if not _v25_farm_enabled(obs) or head in V25_DISABLED_HEADS or not any(legal[1:]):
        return 0
    player = int(obs['player']); step = int(obs['step'])
    if step < _V25_FARM_LAST.get(player, -1):
        V25_FARM_DIAGNOSTICS.clear()
    _V25_FARM_LAST[player] = step
    choice = _v25_hook_choice(obs, head, legal, list(context)[:64])
    if not 0 <= choice < len(legal) or not legal[choice]:
        raise ValueError('Farm hook selected a masked action')
    for label, amount in (('eligible', 1), ('overrides', int(choice != 0))):
        key = str(head) + '_' + label
        V25_FARM_DIAGNOSTICS[key] = V25_FARM_DIAGNOSTICS.get(key, 0) + amount
        TS_DIAGNOSTICS['v25_farm_' + key] = V25_FARM_DIAGNOSTICS[key]
    return choice


def _v25_farm_cash(obs, market):
    """Cash after purchases, with no assumed proceeds from simultaneous sales."""
    farm = obs['farms'][int(obs['player'])]
    cash = float(farm['money']); hires = int(farm['hires_today'])
    quadrants = len(farm['unlocked_quadrants'])
    for order in market:
        if not order:
            continue
        op = order[0]
        if op == 'HIRE':
            cash -= _v219_fib(hires); hires += 1
        elif op == 'BUY_LAND':
            if quadrants < 4:
                cash -= (1000, 2000, 4000)[quadrants - 1]; quadrants += 1
        elif len(order) >= 3:
            quantity = max(0, int(order[2])); item = order[1]
            if op == 'BUY_SEED':
                cash -= quantity * {'WHEAT': 10, 'CARROT': 20, 'TOMATO': 50, 'STRAWBERRY': 100, 'MELON': 80}[item]
            elif op == 'BUY_ANIMAL':
                cash -= quantity * {'GOOSE': 300, 'COW': 400, 'SHEEP': 500}[item]
            elif op == 'BUY_PRODUCT':
                cash -= quantity * (float(obs['market']['prices'][item]) + 10)
    return cash


def _v25_crop_preference(obs, action, state, native_economic):
    if not _v25_farm_enabled(obs) or not _CA_FROM <= int(obs['step']) // 24 <= _CA_TO:
        return 0
    units = [action.get('farmer') or ['PASS'], *(action.get('hands') or [])]
    planting = sum(c[:2] == ['PLANT', 'WHEAT'] for c in units)
    private = obs['private']; day = int(obs['step']) // 24
    have = int(private['seeds'].get('CARROT', 0)) - sum(c[:2] == ['PLANT', 'CARROT'] for c in units)
    buying = sum(int(o[2]) for o in action.get('market', []) if len(o) >= 3 and o[:2] == ['BUY_SEED', 'CARROT'])
    quantity = max(0, _CA_BUFFER - have - buying)
    can_buy = (day <= _CA_TO - 1 and quantity > 0 and len(action.get('market', [])) < 10
               and _v25_farm_cash(obs, action.get('market', [])) >= _CA_CASH + quantity * 20)
    can_plant = planting > 0 and min(state['spare_carrot'], have) > 0
    if not (can_buy or can_plant):
        return 0
    context = [planting / 20, have / 20, state['spare_carrot'] / 20,
               quantity / 20, float(can_buy), float(can_plant), float(native_economic),
               _ca_wheat_total(obs) / 100, _ca_feed_need(int(obs['player']), int(obs['step']), _CA_FEED_DAYS) / 100]
    return _v25_farm_hook(obs, 16, [True, True, can_buy or can_plant], context)


def _v25_livestock_mode(obs, action, state, quantity, evs):
    native = state.get('mode')
    kinds = ('GOOSE', 'COW', 'SHEEP')
    legal = [True]
    for kind in kinds:
        orders = [([o[0], kind, o[2]] if len(o) >= 3 and o[:2] == ['BUY_ANIMAL', 'GOOSE'] else list(o))
                  for o in action.get('market', [])]
        legal.append(_v25_farm_cash(obs, orders) >= 100)
    choice = _v25_farm_hook(obs, 17, legal,
                            [quantity / 10, float(native == 'COW'), float(native == 'SHEEP')]
                            + [float(evs.get(kind, 0)) / 15000 for kind in kinds])
    if not choice:
        return native
    # The later shop-herd layer must not reverse this committed learned mode.
    state['v25_mode_fixed'] = True
    return None if choice == 1 else kinds[choice - 1]


def _v25_shop_herd_target(obs, action, order, native_target):
    fixed = _HD2_STATE.get(int(obs['player']), {}).get('v25_mode_fixed')
    if fixed:
        return None
    kind = order[1]; quantity = int(order[2]); kinds = ('GOOSE', 'COW', 'SHEEP')
    legal = [True]
    for target in kinds:
        # This layer can turn a planned coop into a pasture, but not the reverse.
        compatible = target != 'GOOSE' or kind == 'GOOSE'
        orders = [[o[0], target, o[2]] if o is order else list(o) for o in action.get('market', [])]
        legal.append(compatible and _v25_farm_cash(obs, orders) >= 100)
    choice = _v25_farm_hook(obs, 17, legal,
                            [1, kinds.index(kind) / 2, quantity / 10,
                             (kinds.index(native_target) + 1) / 3 if native_target in kinds else 0])
    return native_target if not choice else (None if kinds[choice - 1] == kind else kinds[choice - 1])


def _v25_day6_swap(obs, action, state):
    if (not _v25_farm_enabled(obs) or state.get('decided')
            or not _CS_FROM <= int(obs['step']) < _CS_TO):
        return 0, None
    buys = [o for o in action.get('market', []) if len(o) >= 3 and o[:2] == ['BUY_ANIMAL', 'COW'] and int(o[2]) > 0]
    if not buys:
        return 0, None
    quantity = sum(int(o[2]) for o in buys)
    plan = _cs_plan(obs, action, quantity, 'GOOSE')
    orders = [[o[0], 'GOOSE', o[2]] if o in buys else list(o) for o in action.get('market', [])]
    legal = [True, True, plan is not None and _v25_farm_cash(obs, orders) >= 100]
    choice = _v25_farm_hook(obs, 18, legal,
                            [quantity / 10, len(plan['builds']) / 10 if plan else 0,
                             len(plan['pickups']) / 10 if plan else 0])
    return choice, plan


def _v25_fertilizer_plans(obs, action, targets, stock, purchases, topup, native):
    if not _v25_farm_enabled(obs):
        return native
    farm = obs['farms'][int(obs['player'])]; day = int(obs['step']) // 24
    proposals = [([], 0, 0, {'WHEAT': 0, 'CARROT': 0})]
    seen = {()}; path_cache = {}
    for first_crop in (None, 'WHEAT', 'CARROT'):
        for maximum_workers in (1, 2):
            for visit_limit in (4, 8):
                remaining = dict(targets); plans = []; total_q = 0; total_cost = 0
                all_units = {'WHEAT': 0, 'CARROT': 0}
                for actor_index in range(maximum_workers):
                    subset = {xy: t for xy, t in remaining.items() if t['crop'] == first_crop} if actor_index == 0 and first_crop else remaining
                    key = (actor_index, tuple(sorted(subset)))
                    if key not in path_cache:
                        path_cache[key] = _r51_input_path(obs, subset, action, actor_index)[0]
                    path = path_cache[key][:visit_limit]; quantity = len(path)
                    if (quantity < 3 or len(action.get('market', [])) + 2 + actor_index > 10
                            or sum(stock.values()) + purchases + total_q + quantity + topup > 95):
                        break
                    quote = _r37_market_price('FERTILIZER', obs['market']['inventory']['FERTILIZER'] - total_q - quantity - topup)
                    cost = (quantity + (topup if actor_index == 0 else 0)) * (quote + 2) + _v219_fib(int(farm['hires_today']) + actor_index)
                    if _v25_farm_cash(obs, action.get('market', [])) < total_cost + cost + 3000:
                        break
                    ready, position = _r62_input_start(obs, action, actor_index)
                    units = {'WHEAT': 0, 'CARROT': 0}
                    for x, y, crop, birth in path:
                        arrival = ready + abs(position[0] - x) + abs(position[1] - y)
                        units[crop] += _r51_input_gain(targets[(x, y)], arrival, day)
                        ready = arrival + 1; position = (x, y)
                    plans.append({'path': list(path), 'quantity': quantity, 'loaded': False})
                    total_q += quantity; total_cost += cost
                    for item in units:
                        all_units[item] += units[item]
                    for x, y, _, _ in path:
                        remaining.pop((x, y), None)
                signature = tuple(tuple(p['path']) for p in plans)
                if signature not in seen:
                    seen.add(signature); proposals.append((plans, total_q, total_cost, all_units))
    if len(proposals) == 1 and not native[0]:
        return native
    legal = [True] + [True] * len(proposals) + [False] * (15 - len(proposals))
    context = [len(targets) / 25, topup / 20, len(native[0]) / 2, native[1] / 20]
    for plans, quantity, cost, units in proposals:
        context += [len(plans) / 2, quantity / 20, cost / 3000, (units['CARROT'] - units['WHEAT']) / 20]
    choice = _v25_farm_hook(obs, 19, legal, context)
    return native if not choice else proposals[choice - 1]


def _v25_care_command(obs, actor, position, command, units, positions):
    if not _v25_farm_enabled(obs) or not command or command[0] not in ('CARE', 'COLLECT_FERTILIZER'):
        return None
    farm = obs['farms'][int(obs['player'])]; x, y = position
    tile = farm['tiles'][y][x]
    if not isinstance(tile, dict) or tile.get('animal') not in _CH_ANIMALS:
        return None
    quantity = max(0, int(tile.get('yield_units', 0)))
    carried = sum(sum(int(v) for v in inv.values()) for inv in obs['private']['inventories'])
    room = _CH_SHED - sum(obs['private']['shed'].values()) - carried
    legal = [True, quantity > 0 and room >= quantity, not tile.get('cared_today', False),
             bool(tile.get('fertilizer_available')) and room >= 1]
    occupied = {other[0] for index, other in enumerate(units)
                if index != actor and index < len(positions)
                and positions[index] == position and other}
    for index, operation in enumerate(('HARVEST', 'CARE', 'COLLECT_FERTILIZER'), 1):
        legal[index] &= operation not in occupied
    context = [actor / 20, x / 9, y / 9, quantity / 6, room / 100,
               float(tile.get('fed_today', False)), float(tile.get('cared_today', False)),
               tile.get('pending_care_bonus', 0) / 6,
               (int(obs['step']) // 24 - int(tile['placed_day'])) / 30,
               float(command[0] == 'CARE')]
    choice = _v25_farm_hook(obs, 20, legal, context)
    return None if not choice else [['HARVEST'], ['CARE'], ['COLLECT_FERTILIZER']][choice - 1]


def _v25_feed_skip(obs, actor, tile, bonus, native_skip):
    context = [0, actor / 20, tile.get('yield_units', 0) / 6,
               tile.get('pending_care_bonus', 0) / 6, bonus / 6, float(native_skip),
               tile.get('consecutive_unfed', 0) / 2]
    choice = _v25_farm_hook(obs, 21, [True, True, True], context)
    return native_skip if not choice else choice == 2


def _v25_water_skip(obs, actor, tile):
    # Only an established plant watered yesterday may skip today. Its next
    # observed unwatered count then masks skipping, preserving the survival rule.
    day = int(obs['step']) // 24
    can_skip = (int(tile.get('consecutive_unwatered', 1)) == 0
                and day > int(tile.get('planted_day', day)))
    choice = _v25_farm_hook(obs, 21, [True, True, can_skip],
                            [1, actor / 20, tile.get('yield_units', 0) / 8,
                             (day - int(tile.get('planted_day', day))) / 16,
                             tile.get('consecutive_unwatered', 1) / 2,
                             float(tile.get('fertilized_until_day', -1) >= day)])
    return choice == 2


def _v25_assignment(obs, choices, spawns, remaining, workers, fertilizer, native):
    proposals = [None]
    for maximum, total, assignment in sorted(choices):
        proposals.append(dict(paths=assignment, spawns=spawns, remaining=remaining,
                              workers=workers, fertilizer=fertilizer))
    if len(proposals) > 7:
        raise ValueError('Unexpected assignment candidate count')
    legal = [True] + [True] * len(proposals) + [False] * (7 - len(proposals))
    context = [remaining / 24, workers / 3, float(fertilizer), len(choices) / 6]
    for proposal in proposals:
        if proposal is None:
            context += [0] * 8
        else:
            paths = proposal['paths']
            descriptor = [workers / 3, sum(len(p) for p in paths) / 10]
            descriptor += [coordinate / 9 for path in paths for coordinate in path[0]]
            context += (descriptor + [0] * 8)[:8]
    choice = _v25_farm_hook(obs, 22, legal, context)
    return native if not choice else proposals[choice - 1]

# Existing layers catch their own failures. Keep a separate hard-error signal
# so a failed learned decision cannot silently enter a PPO batch as valid data.
def _v25_guard_farm_function(fn):
    def guarded(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            V25_DIAGNOSTICS['errors'] = V25_DIAGNOSTICS.get('errors', 0) + 1
            raise
    return guarded

for _v25_name in ('_v25_crop_preference', '_v25_livestock_mode', '_v25_shop_herd_target',
                  '_v25_day6_swap', '_v25_fertilizer_plans', '_v25_care_command',
                  '_v25_feed_skip', '_v25_water_skip', '_v25_assignment'):
    globals()[_v25_name] = _v25_guard_farm_function(globals()[_v25_name])
