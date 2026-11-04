"""Static precommit edits; every anchor must match the frozen v24 exactly once."""
def install(source, replace):
    edits = [
        ('    return dict(paths=assignment,spawns=spawns,remaining=remaining,workers=count,fertilizer=fertilizer)\n',
         '    native=dict(paths=assignment,spawns=spawns,remaining=remaining,workers=count,fertilizer=fertilizer)\n    return _v25_assignment(obs,choices,spawns,remaining,count,fertilizer,native)\n'),
        ('    return plans,total_q,total_cost,all_units\n',
         '    return _v25_fertilizer_plans(obs,action,targets,stock,purchases,topup,(plans,total_q,total_cost,all_units))\n'),
        ('        if bonus*(float(prices[item])+5)*1.25>=float(prices[\'WHEAT\']):continue\n        if not _r86_next_feed(obs,positions[actor]):continue\n',
         '        native_skip=bonus*(float(prices[item])+5)*1.25<float(prices[\'WHEAT\'])\n        if not _r86_next_feed(obs,positions[actor]):continue\n        if not _v25_feed_skip(obs,actor,tile,bonus,native_skip):continue\n'),
        ('        pays_now = 3 * (p_c - _CA_DROP) - 20 > 4 * p_w - 10 + _CA_MARGIN\n',
         '        pays_now = 3 * (p_c - _CA_DROP) - 20 > 4 * p_w - 10 + _CA_MARGIN\n        v25_crop = _v25_crop_preference(observation, action, st, pays_now)\n        if v25_crop: pays_now = v25_crop == 2\n'),
        ('                if cu * (p_c - _CA_DROP) - 20 > wu * p_w - 10 + _CA_MARGIN:\n',
         '                if v25_crop == 2 or cu * (p_c - _CA_DROP) - 20 > wu * p_w - 10 + _CA_MARGIN:\n'),
        ('        _HD2_REPORT["hd2_decision"] = "keep@%d" % step\n',
         '        _HD2_REPORT["hd2_decision"] = "keep@%d" % step\n    st["mode"] = _v25_livestock_mode(obs, action, st, k, evs)\n'),
        ("                to=_y_target(o[1],shops,cfg)\n",
         "                to=_v25_shop_herd_target(obs,result,o,_y_target(o[1],shops,cfg))\n"),
        ('        if not st["decided"] and _CS_FROM <= step < _CS_TO:\n',
         '''        v25_swap, v25_plan = _v25_day6_swap(observation, action, st)
        if v25_swap:
            st["decided"] = True
            if v25_swap == 2:
                st["mode"], st["plan"] = "GOOSE", v25_plan
                for o in market:
                    if len(o) >= 3 and o[:2] == ["BUY_ANIMAL", "COW"]:
                        o[1] = "GOOSE"
                        _CS_REPORT["cs_rewrites"] += 1
        if not st["decided"] and _CS_FROM <= step < _CS_TO:
'''),
        ('            product, cap, first, interval = _CH_ANIMALS[tile["animal"]]\n',
         '''            v25_command = _v25_care_command(observation, i, pos, cmd, units, positions)
            if v25_command is not None:
                units[i] = v25_command
                changed |= v25_command != cmd
                if v25_command == ["HARVEST"]:
                    product = _CH_ANIMALS[tile["animal"]][0]
                    st["credit"][product] = st["credit"].get(product, 0) + max(0, int(tile.get("yield_units", 0)))
                continue
            product, cap, first, interval = _CH_ANIMALS[tile["animal"]]
'''),
        ('        return _v219_walk(pos,target) or command\n',
         '''        walk = _v219_walk(pos,target)
        if not walk and command == ['WATER'] and _v25_water_skip(obs, actor, view.tiles[target[1]][target[0]]):
            return ['PASS']
        return walk or command
'''),
    ]
    for old, new in edits:
        source = replace(source, old, new)
    return source
