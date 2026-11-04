"""v25 partial PPO controller, embedded after the frozen v24 implementation.

Only observable state and the agent's own history enter the policy.  NumPy
weights use torch.nn.Linear's [out, in] layout.  Native priors are applied in
forward(), never stored in the exported output bias.
"""
import math as _v25_math
import random as _v25_random
import numpy as _v25_np

V25_BASE_FEATURE_DIM = 288
V25_CONTEXT_DIM = 64
FEATURE_DIM = V25_BASE_FEATURE_DIM + V25_CONTEXT_DIM
V25_ITEMS = ('CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL')
V25_PRODUCTS = ('WHEAT',) + V25_ITEMS + ('FERTILIZER',)
V25_CROPS = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON')
V25_ANIMALS = ('COW', 'SHEEP', 'GOOSE')
V25_SHOPS = ('BAKERY', 'PIZZA_SHOP', 'BRUNCH_SPOT', 'YARN_STORE',
             'ICE_CREAM_SHOP', 'PET_CAFE', 'SMOOTHIE_SHOP', 'FARMERS_MARKET')
V25_ROUTES = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 100, 101, 103,
              104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
              116, 117, 118, 120, 121, 122, 123, 124, 126, 128)
# The terminal-only and currently unused paths require separate qualification.
V25_ALLOWED_ROUTES = tuple(r for r in V25_ROUTES if r not in (2, 4))
V25_HEAD_WIDTHS = (38, 3) + (6,) * 7 + (5,) * 7 + (3, 4, 3, 16, 4, 3, 8)
V25_HEAD_COUNT = len(V25_HEAD_WIDTHS)
V25_POLICY_DIM = sum(V25_HEAD_WIDTHS)
V25_MODEL = None
V25_MODE = 'neutral'
V25_TRAINING = False
V25_RNG = None
V25_DISABLED_HEADS = ()
V25_TRAJECTORY = []
V25_STATES = {}
V25_DIAGNOSTICS = {}
_V25_PARENT = globals().get('agent')
_V25_ARRAY_CACHE = None


def _v25_enabled(obs):
    return V25_MODE == 'policy' and 144 <= int(obs['step']) < 696


def _v25_state(obs):
    return V25_STATES.get(int(obs['player']), {})


def _v25_choice(obs, head):
    if not _v25_enabled(obs) or head in V25_DISABLED_HEADS:
        return 0
    return int(_v25_state(obs).get('actions', [0] * V25_HEAD_COUNT)[head])


def _v25_default_route(obs):
    player = int(obs['player'])
    native = _IMPL.chassis.players.get(player, {})
    state = native.get('router_state', {})
    shops = tuple(obs['town']['unlocked_shops'][:2])
    route = (_R108_SHOP_ROUTES.get(shops, 100) if shops.count('YARN_STORE') <= 0
             else _R110_OLD_SHOPS.get(shops, 0))
    route = _V92_TABLE.get(shops, route)
    if 'YARN_STORE' in shops and state.get('rkey') in _V93_ROUTE_BY_RIVAL:
        route = _V93_ROUTE_BY_RIVAL[state['rkey']]
    return {119: 101, 125: 105, 127: 109}.get(route, route)


def _v25_route_choice(obs, native_route):
    choice = _v25_choice(obs, 0)
    if not choice or int(obs['step']) != 144:
        return native_route
    route = V25_ROUTES[choice - 1]
    native = _IMPL.chassis.players.get(int(obs['player']), {})
    if route not in V25_ALLOWED_ROUTES or native.get('pending'):
        V25_DIAGNOSTICS['route_safety_declines'] += 1
        return native_route
    V25_DIAGNOSTICS['route_changed'] += int(route != native_route)
    return route


def _v25_tomato_compatible(obs):
    farm = obs['farms'][int(obs['player'])]
    if len(farm['tiles']) != 10 or set(farm['unlocked_quadrants']) != {'NW', 'NE', 'SW'}:
        return False
    if farm['money'] < 12000:
        return False
    if any(farm['tiles'][y][x] != 'LOCKED' for y in (5, 6) for x in range(5, 10)):
        return False
    private = obs['private']
    if private['seeds'].get('TOMATO', 0) or private['shed'].get('TOMATO', 0):
        return False
    if any(isinstance(t, dict) and t.get('crop') == 'TOMATO' for row in farm['tiles'] for t in row):
        return False
    for tape in _IMPL.chassis.routes.values():
        for action in tape[432:719]:
            if any(o and o[0] == 'BUY_LAND' for o in action.get('market', [])):
                return False
            if any(c == ['PLANT', 'TOMATO'] for c in [action.get('farmer')] + action.get('hands', [])):
                return False
    return True


def _v25_tomato_economic_gate(obs):
    choice = _v25_choice(obs, 1)
    if choice and int(obs['step']) == 432:
        V25_DIAGNOSTICS['investment_overrides'] += 1
        return choice == 2
    return _cxtb_expected_revenue(obs) >= _CXTB_MIN_REVENUE


def _v25_reservation_horizons(obs, original):
    choices = [_v25_choice(obs, 2 + i) for i in range(7)]
    if not any(choices) or int(obs['step']) < 192:
        return original
    native_end = max(original.values()) if original else _R37_HORIZONS.get(int(obs['player']), 2)
    horizons = {item: (original or {}).get(item, native_end) for item in PRODUCTS}
    for item, choice in zip(V25_ITEMS, choices):
        if choice:
            horizons[item] = (0, 0, 4, 12, 24, 48)[choice]
            V25_DIAGNOSTICS['reservation_control_calls'] += 1
    return horizons


def _v25_reservation_learned(obs, item):
    return (int(obs['step']) >= 192 and item in V25_ITEMS
            and bool(_v25_choice(obs, 2 + V25_ITEMS.index(item))))


def _v25_advance_scan(obs, native):
    return 24 if any(_v25_choice(obs, 9 + i) >= 2 for i in range(7)) else native


def _v25_advance_include(obs, item, offset):
    if item not in V25_ITEMS:
        return offset <= _ADV_LOOK
    choice = _v25_choice(obs, 9 + V25_ITEMS.index(item))
    return offset <= _ADV_LOOK if choice == 0 else choice >= 2 and offset <= 24


def _v25_advance_quantity(obs, item, quantity):
    choice = _v25_choice(obs, 9 + V25_ITEMS.index(item)) if item in V25_ITEMS else 0
    if choice == 0:
        return quantity
    if choice == 1:
        return 0
    V25_DIAGNOSTICS['advance_control_calls'] += 1
    return min(quantity, max(1, int(_v25_math.ceil(quantity * (0, 0, .25, .5, 1)[choice]))))


def _v25_advance_allowed(obs, item, quantity, due, protected):
    choice = _v25_choice(obs, 9 + V25_ITEMS.index(item)) if item in V25_ITEMS else 0
    if choice >= 2:
        return True
    return _v20_allow(obs, item, quantity, due, protected)


def _v25_feature_vector(obs, state, context=None):
    p = int(obs['player']); step = int(obs['step'])
    own, rival = obs['farms'][p], obs['farms'][1 - p]
    private = obs['private']; shops = obs['town']['unlocked_shops']
    native = _IMPL.chassis.players.get(p, {})
    route = native.get('route', 0)
    canonical_route = {119: 101, 125: 105, 127: 109}.get(route, route)
    inventories = private.get('inventories', [])
    cash = float(own['money']); other_cash = float(rival['money'])
    values = [step / 719, (step // 24) / 29, (step % 24) / 23,
              _v25_math.sin(2 * _v25_math.pi * (step % 24) / 24),
              _v25_math.cos(2 * _v25_math.pi * (step % 24) / 24),
              cash / 30000, other_cash / 30000, (cash - other_cash) / 10000,
              sum(private['shed'].values()) / 100,
              sum(sum(inv.values()) for inv in inventories) / 100,
              len(own['hands']) / 20, len(rival['hands']) / 20,
              len(own['unlocked_quadrants']) / 4, len(rival['unlocked_quadrants']) / 4,
              own.get('hires_today', 0) / 20, rival.get('hires_today', 0) / 20,
              float(bool(native.get('pending'))), float(step >= 648)]
    values.extend(shops.count(shop) / 8 for shop in V25_SHOPS)
    for item in V25_PRODUCTS + V25_ANIMALS:
        values.append(float(private['shed'].get(item, 0)) / 100)
        values.append(sum(float(inv.get(item, 0)) for inv in inventories) / 100)
    values.extend(float(private['seeds'].get(crop, 0)) / 100 for crop in V25_CROPS)
    prices = obs['market']['prices']; market = obs['market']['inventory']
    base = {'WHEAT': 25, 'CARROT': 35, 'TOMATO': 60, 'STRAWBERRY': 120,
            'MELON': 250, 'EGG': 50, 'MILK': 160, 'WOOL': 200, 'FERTILIZER': 100}
    draw = _v9_town_draw(shops, step)
    tape = _IMPL.chassis.routes.get(route, _IMPL.chassis.routes[0])
    future_sales = {item: 0 for item in V25_PRODUCTS}
    next_sale = {item: 48 for item in V25_PRODUCTS}
    for t in range(step + 1, min(719, step + 49)):
        source = _IMPL.chassis.routes[2] if t >= 648 else tape
        for order in source[t].get('market', []):
            if len(order) >= 3 and order[0] == 'SELL' and order[1] in future_sales:
                item = order[1]
                next_sale[item] = min(next_sale[item], t - step)
                if t - step <= 24:
                    future_sales[item] += max(0, int(order[2]))
    for item in V25_PRODUCTS:
        values.extend([float(prices.get(item, 0)) / base[item], float(market.get(item, 0)) / 400,
                       state.get('delta', {}).get(item, 0) / 20,
                       state.get('ema', {}).get(item, 0) / 20,
                       draw.get(item, 0) / 10, future_sales[item] / 40, next_sale[item] / 48])
    for farm in (own, rival):
        tiles = [tile for row in farm['tiles'] for tile in row if isinstance(tile, dict)]
        values.extend(float(q in farm['unlocked_quadrants']) for q in ('NW', 'NE', 'SW', 'SE'))
        positions = [farm['farmer']] + farm['hands']
        distances = [min(abs(x - a) + abs(y - b) for a, b in ((4, 4), (5, 4), (4, 5), (5, 5))) for x, y in positions]
        values.extend([min(distances) / 10, max(distances) / 10, sum(distances) / (10 * len(distances)),
                       distances[0] / 10, len(farm['hands']) / 20, farm.get('hires_today', 0) / 20,
                       float(farm['money']) / 30000])
        for crop in V25_CROPS:
            matched = [t for t in tiles if t.get('crop') == crop]
            values.extend([len(matched) / 25, sum(t.get('yield_units', 0) for t in matched) / 100,
                           sum(max(0, step // 24 - t.get('planted_day', step // 24)) for t in matched) / 200,
                           sum(not t.get('watered_today', False) for t in matched) / 25])
        for animal in V25_ANIMALS:
            matched = [t for t in tiles if t.get('animal') == animal]
            values.extend([len(matched) / 20, sum(t.get('yield_units', 0) for t in matched) / 100,
                           sum(max(0, step // 24 - t.get('placed_day', step // 24)) for t in matched) / 200,
                           sum(not t.get('fed_today', False) for t in matched) / 20])
        values.extend([farm['farmer'][0] / 9, farm['farmer'][1] / 9,
                       min(x for x, y in positions) / 9, max(x for x, y in positions) / 9,
                       min(y for x, y in positions) / 9, max(y for x, y in positions) / 9,
                       sum(x for x, y in positions) / (9 * len(positions)),
                       sum(y for x, y in positions) / (9 * len(positions))])
    values.extend(float(canonical_route == rid) for rid in V25_ROUTES)
    values.append(float(canonical_route not in V25_ROUTES))
    values.extend(float(a) / max(1, width - 1) for a, width in zip(state.get('actions', [0] * V25_HEAD_COUNT)[:16], V25_HEAD_WIDTHS[:16]))
    tomato = _V219_STATES.get(p, {})
    values.extend([float(bool(tomato.get('committed'))), float(bool(tomato.get('eligible'))),
                   _cxtb_expected_revenue(obs) / 15000])
    debts = native.get('sell_state', {}).get('r36_debts', {})
    values.extend(sum(q.get(item, 0) for due, q in debts.items() if step <= due <= step + 48) / 100 for item in V25_ITEMS)
    if len(values) > V25_BASE_FEATURE_DIM:
        raise ValueError('v25 feature overflow: ' + str(len(values)))
    values.extend([0.0] * (V25_BASE_FEATURE_DIM - len(values)))
    extra = list(context) if context is not None else []
    if len(extra) > V25_CONTEXT_DIM:
        raise ValueError('v25 hook context overflow: ' + str(len(extra)))
    values.extend(extra + [0.] * (V25_CONTEXT_DIM - len(extra)))
    return _v25_np.clip(_v25_np.asarray(values, dtype=_v25_np.float32), -5, 5)


def _v25_action_mask(obs):
    step = int(obs['step']); p = int(obs['player'])
    masks = [[True] + [False] * (width - 1) for width in V25_HEAD_WIDTHS]
    active = [False] * V25_HEAD_COUNT
    native = _IMPL.chassis.players.get(p, {})
    if step == 144 and not native.get('pending') and len(obs['farms'][p]['tiles']) == 10:
        default = _v25_default_route(obs)
        for i, route in enumerate(V25_ROUTES, 1):
            masks[0][i] = route in V25_ALLOWED_ROUTES and route != default
        active[0] = any(masks[0][1:])
    if step == 432 and _v25_tomato_compatible(obs):
        masks[1] = [True] * 3; active[1] = True
    private = obs['private']
    for i, item in enumerate(V25_ITEMS):
        available = int(private['shed'].get(item, 0)) + sum(int(inv.get(item, 0)) for inv in private['inventories'])
        if available > 0 and int(obs['market']['prices'].get(item, 0)) > 1:
            if step >= 192:
                masks[2 + i] = [True] * 6; active[2 + i] = True
            masks[9 + i] = [True] * 5; active[9 + i] = True
    for head in V25_DISABLED_HEADS:
        if not isinstance(head, int) or not 0 <= head < len(V25_HEAD_WIDTHS):
            raise ValueError('invalid disabled v25 head: ' + str(head))
        masks[head] = [True] + [False] * (V25_HEAD_WIDTHS[head] - 1)
        active[head] = False
    return masks, active


def _v25_forward(features):
    global _V25_ARRAY_CACHE
    if V25_MODEL is None:
        logits = _v25_np.zeros(V25_POLICY_DIM, dtype=_v25_np.float32); value = 0.0
    else:
        if _V25_ARRAY_CACHE is None or _V25_ARRAY_CACHE[0] is not V25_MODEL:
            arrays = {k: _v25_np.asarray(V25_MODEL[k], dtype=_v25_np.float32)
                      for k in ('w1', 'b1', 'w2', 'b2', 'wp', 'bp', 'wv', 'bv')}
            _V25_ARRAY_CACHE = (V25_MODEL, arrays)
        w = _V25_ARRAY_CACHE[1]
        h = _v25_np.tanh(features @ w['w1'].T + w['b1'])
        h = _v25_np.tanh(h @ w['w2'].T + w['b2'])
        logits = _v25_np.asarray(h @ w['wp'].T + w['bp'], dtype=_v25_np.float32).copy()
        value = float(_v25_np.asarray(h @ w['wv'].T + w['bv']).reshape(-1)[0])
    offset = 0
    for head, width in enumerate(V25_HEAD_WIDTHS):
        logits[offset] += 5.0 if head == 0 else (3.0 if head < 16 else 4.0)
        offset += width
    if not _v25_np.all(_v25_np.isfinite(logits)) or not _v25_math.isfinite(value):
        raise ValueError('nonfinite v25 model output')
    return logits, value


def _v25_begin(obs):
    p = int(obs['player']); step = int(obs['step'])
    state = V25_STATES.get(p)
    if state is None or step <= state.get('step', -1):
        state = {'step': -1, 'actions': [0] * V25_HEAD_COUNT, 'prev_market': None,
                 'own_sales': {}, 'delta': {}, 'ema': {}}
        V25_STATES[p] = state
        V25_TRAJECTORY.clear()
        V25_DIAGNOSTICS.clear()
        V25_DIAGNOSTICS.update(decisions=0, route_changed=0, route_safety_declines=0,
                               investment_overrides=0, reservation_control_calls=0,
                               advance_control_calls=0, farm_hook_decisions=0,
                               farm_hook_overrides=0, errors=0)
    previous = state.get('prev_market')
    if previous is not None and step == state['step'] + 1:
        draw = _v9_town_draw(state.get('prev_shops', []), step - 1)
        for item in V25_PRODUCTS:
            delta = (obs['market']['inventory'].get(item, 0) - previous.get(item, 0)
                     + draw.get(item, 0) - state['own_sales'].get(item, 0))
            state['delta'][item] = delta
            state['ema'][item] = .85 * state['ema'].get(item, 0) + .15 * delta
    state['step'] = step
    state['prev_market'] = dict(obs['market']['inventory'])
    state['prev_shops'] = list(obs['town']['unlocked_shops'])
    if not _v25_enabled(obs):
        state['actions'] = [0] * V25_HEAD_COUNT
        return
    if step % 4:
        return
    features = _v25_feature_vector(obs, state)
    masks, active = _v25_action_mask(obs)
    logits, value = _v25_forward(features)
    actions = []; old_log_probs = []; offset = 0
    for head, width in enumerate(V25_HEAD_WIDTHS):
        scores = logits[offset:offset + width].astype(_v25_np.float64)
        scores[~_v25_np.asarray(masks[head], dtype=bool)] = -1e30
        scores -= scores.max()
        probabilities = _v25_np.exp(scores); probabilities /= probabilities.sum()
        if V25_TRAINING and active[head]:
            if V25_RNG is None:
                raise ValueError('training requires a private V25_RNG')
            draw = float(V25_RNG.random()); choice = width - 1
            for index, probability in enumerate(probabilities):
                draw -= float(probability)
                if draw <= 0:
                    choice = index; break
        else:
            choice = int(_v25_np.argmax(scores))
        actions.append(choice)
        old_log_probs.append(float(_v25_math.log(max(1e-300, float(probabilities[choice])))) if active[head] else 0.0)
        offset += width
    state['actions'] = actions
    V25_TRAJECTORY.append({'features': features.tolist(), 'actions': actions,
                           'mask': [bool(v) for mask in masks for v in mask],
                           'active': active, 'old_log_probs': old_log_probs,
                           'value': value, 'step': step})
    V25_DIAGNOSTICS['decisions'] += 1


def _v25_hook_choice(obs, head, legal_mask, context=None):
    """Choose at a native controller's decision point, before it commits state.

Only this hook's head is active in the record. Other heads are native-only,
so repeated decisions within a turn remain separate conditional actions.
The caller must only supply candidates it can execute through its own normal
state updates; zero always means the exact native choice.
"""
    if V25_MODE != 'policy' or not 144 <= int(obs['step']) < 696 or head in V25_DISABLED_HEADS:
        return 0
    if not 16 <= head < V25_HEAD_COUNT:
        raise ValueError('invalid farm hook head: ' + str(head))
    allowed = [bool(x) for x in legal_mask]
    if len(allowed) != V25_HEAD_WIDTHS[head] or not allowed[0]:
        raise ValueError('invalid farm hook mask')
    if not any(allowed[1:]):
        return 0
    state = _v25_state(obs)
    if not state or state.get('step') != int(obs['step']):
        raise ValueError('farm hook called outside initialized current turn')
    features = _v25_feature_vector(obs, state, context)
    logits, value = _v25_forward(features)
    offset = sum(V25_HEAD_WIDTHS[:head]); width = V25_HEAD_WIDTHS[head]
    scores = logits[offset:offset + width].astype(_v25_np.float64)
    scores[~_v25_np.asarray(allowed, dtype=bool)] = -1e30
    scores -= scores.max()
    probabilities = _v25_np.exp(scores); probabilities /= probabilities.sum()
    if V25_TRAINING:
        if V25_RNG is None:
            raise ValueError('training requires a private V25_RNG')
        draw = float(V25_RNG.random()); choice = width - 1
        for index, probability in enumerate(probabilities):
            draw -= float(probability)
            if draw <= 0:
                choice = index
                break
    else:
        choice = int(_v25_np.argmax(scores))
    masks = [[True] + [False] * (n - 1) for n in V25_HEAD_WIDTHS]
    masks[head] = allowed
    active = [False] * V25_HEAD_COUNT; active[head] = True
    actions = [0] * V25_HEAD_COUNT; actions[head] = choice
    log_probs = [0.] * V25_HEAD_COUNT
    log_probs[head] = float(_v25_math.log(max(1e-300, float(probabilities[choice]))))
    V25_TRAJECTORY.append({'features': features.tolist(), 'actions': actions,
                           'mask': [v for mask in masks for v in mask],
                           'active': active, 'old_log_probs': log_probs,
                           'value': value, 'step': int(obs['step'])})
    V25_DIAGNOSTICS['farm_hook_decisions'] += 1
    V25_DIAGNOSTICS['farm_hook_overrides'] += int(choice != 0)
    name = 'head_' + str(head)
    V25_DIAGNOSTICS[name + '_decisions'] = V25_DIAGNOSTICS.get(name + '_decisions', 0) + 1
    V25_DIAGNOSTICS[name + '_overrides'] = V25_DIAGNOSTICS.get(name + '_overrides', 0) + int(choice != 0)
    return choice


def _v25_after(obs, action):
    state = _v25_state(obs)
    stock = projected_shed(action, FarmView(obs)); sales = {}
    for order in action.get('market', []):
        if len(order) >= 3 and order[0] == 'SELL':
            item = order[1]
            amount = min(max(0, int(order[2])), max(0, int(stock.get(item, 0)) - sales.get(item, 0)))
            sales[item] = sales.get(item, 0) + amount
        elif len(order) >= 3 and order[0] == 'BUY_PRODUCT':
            # These two inputs are excluded from learned sale control.  Keep
            # their inferred public deltas from being mislabelled as rival sales.
            item = order[1]
            sales[item] = sales.get(item, 0) - max(0, int(order[2]))
    state['own_sales'] = sales


def kaggriculture_v25(observation, configuration=None):
    _v25_begin(observation)
    action = _V25_PARENT(observation, configuration)
    if V25_DIAGNOSTICS.get('errors', 0):
        raise RuntimeError('v25 learned control failed before native state commitment')
    _v25_after(observation, action)
    TS_DIAGNOSTICS.update({'v25_' + k: v for k, v in V25_DIAGNOSTICS.items()})
    return action


agent = kaggriculture_v25
agent = globals().pop('agent')
