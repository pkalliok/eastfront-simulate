
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from itertools import zip_longest
from math import log
from random import randint

trials = 1000

def d6(): return randint(1,6)

@dataclass(frozen=True)
class BattleTroops:
    triple_fire_cv: tuple[int] = ()
    double_fire_cv: tuple[int] = ()
    single_fire_cv: tuple[int] = ()

@dataclass(frozen=True)
class BattleStanding:
    attacker: BattleTroops = BattleTroops()
    defender: BattleTroops = BattleTroops()

def apply_damage(victim: BattleTroops, aggressor: BattleTroops):
    damage = sum(1 for _ in range(sum(aggressor.triple_fire_cv)) if d6() >= 4) \
           + sum(1 for _ in range(sum(aggressor.double_fire_cv)) if d6() >= 5) \
           + sum(1 for _ in range(sum(aggressor.single_fire_cv)) if d6() >= 6)
    if damage == 0: return victim
    result_triple = list(victim.triple_fire_cv)
    result_double = list(victim.double_fire_cv)
    result_single = list(victim.single_fire_cv)
    while damage > 0:
        max_triple = max(result_triple, default=0)
        max_double = max(result_double, default=0)
        max_single = max(result_single, default=0)
        if max_triple > max_double and max_triple > max_single:
            result_triple[result_triple.index(max_triple)] -= 1
        elif max_double > max_single:
            result_double[result_double.index(max_double)] -= 1
        elif max_single > 0:
            result_single[result_single.index(max_single)] -= 1
        else: break
        damage -= 1
    return BattleTroops(tuple(result_triple),
            tuple(result_double),
            tuple(result_single))

def battle_round_outcome(init: BattleStanding, air_strike: BattleTroops):
    firing_defenders = apply_damage(init.defender, air_strike)
    remaining_attackers = apply_damage(init.attacker, firing_defenders)
    remaining_defenders = apply_damage(firing_defenders, remaining_attackers)
    return BattleStanding(remaining_attackers, remaining_defenders)

def battle_round_outcome_distribution(
        init: BattleStanding, air_strike: BattleTroops
):
    result = defaultdict(int)
    for _ in range(trials):
        result[battle_round_outcome(init, air_strike)] += 1./trials
    return result

def order_by_likelihood(outcome_distribution):
    return sorted(((prob, standing)
            for standing, prob in outcome_distribution.items()),
            key = lambda x: -x[0])

def is_defeated(troops: BattleTroops):
    return sum(troops.triple_fire_cv \
            + troops.double_fire_cv \
            + troops.single_fire_cv) == 0

def outcome_sum(outcome1, outcome2, weight1, weight2):
    return [(weight1*victory1+weight2*victory2, weight1*defeat1+weight2*defeat2)
            for (victory1, defeat1), (victory2, defeat2)
            in zip_longest(outcome1, outcome2, fillvalue=(0.,0.))]

def recursive_regression(extended_outcome, recurse_probability):
    # approximate number of regressions to cover 99% of cases
    regression_length = round(-5. / log(recurse_probability))
    result = extended_outcome
    for _ in range(regression_length):
        result = outcome_sum(extended_outcome,
                [(0.,0.)] + result, 1., recurse_probability)
    return result

@cache
def extended_outcome_distribution(
        init: BattleStanding, air_strike: BattleTroops
):
    if is_defeated(init.attacker): return [(0.,1.)]
    if is_defeated(init.defender): return [(1.,0.)]
    one_round = battle_round_outcome_distribution(init, air_strike)
    recurse_probability = one_round.pop(init)
    regression_history, instant_win_prob, instant_defeat_prob = [], 0., 0.
    for next_standing, probability in one_round.items():
        if is_defeated(next_standing.defender):
            instant_win_prob += probability
        elif is_defeated(next_standing.attacker):
            instant_defeat_prob += probability
        else:
            regression_history = outcome_sum(
                extended_outcome_distribution(next_standing, air_strike),
                regression_history, probability, 1.)
    result_without_recursion = \
            [(instant_win_prob, instant_defeat_prob)] + regression_history
    return recursive_regression(result_without_recursion, recurse_probability)

def interpret_extended_outcome(extended_outcome):
    win_prob = sum(win for win, _ in extended_outcome)
    loss_prob = sum(loss for _, loss in extended_outcome)
    return {
        "win probability": win_prob, "loss probability": loss_prob,
        "expected time to win": 1 + sum((w / win_prob) * turn
            for turn, (w, _) in enumerate(extended_outcome)),
        "expected time to lose": 1 + sum((l / loss_prob) * turn
            for turn, (_, l) in enumerate(extended_outcome)),
    }

