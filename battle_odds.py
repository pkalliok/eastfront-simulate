
import re
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
    hits_to_damage: int = 1

@dataclass(frozen=True)
class BattleStanding:
    attacker: BattleTroops = BattleTroops()
    defender: BattleTroops = BattleTroops()

## helpers for writing BattleTroops and BattleStandings

troopspec_re = re.compile(r"(T[0-9]+)?(D[0-9]+)?(S[0-9]+)?(:[1-3])?")

def units(unitspec):
    if not unitspec: return ()
    return tuple(int(cv) for cv in unitspec[1:])

def troops(troopspec: str):
    """Make a BattleTroops out of string representation.
    For example `T43D1S4:2` means Triple Fire units of 4 and 3 CV,
    Double Fire of 1 CV, Single Fire of 2 CV, standing in a terrain that
    gives double defense."""
    parsed = troopspec_re.match(troopspec)
    if not parsed: raise Exception("bad troopspec: " + troopspec)
    tf, df, sf, defense = parsed.groups()
    return BattleTroops(
            units(tf), units(df), units(sf),
            int((defense or ":1")[1:]))

def standing(standspec: str):
    """Make a BattleStanding out of string representation.
    For example `D4/S3:1` means Double Fire unit of 4CV against Single
    Fire unit of 3CV in a field (no double defense)."""
    attack, defend = standspec.split("/")
    return BattleStanding(troops(attack), troops(defend))

# game logic

def apply_damage(
        victim: BattleTroops,
        aggressor: BattleTroops,
        carryover: int = 0
):
    """calculate the new situation of victim after damage is done by aggressor.
    Returns number of partial hits (carryover) and victim as new BattleTroops."""
    hits = carryover \
         + sum(1 for _ in range(sum(aggressor.triple_fire_cv)) if d6() >= 4) \
         + sum(1 for _ in range(sum(aggressor.double_fire_cv)) if d6() >= 5) \
         + sum(1 for _ in range(sum(aggressor.single_fire_cv)) if d6() >= 6)
    if hits == 0: return (0, victim)
    damage = hits // victim.hits_to_damage
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
    return (damage % victim.hits_to_damage, BattleTroops(
            tuple(result_triple),
            tuple(result_double),
            tuple(result_single),
            victim.hits_to_damage))

def battle_round_outcome(init: BattleStanding, air_strike: BattleTroops):
    """return new battle standing after one round of battle"""
    carryover, firing_defenders = apply_damage(init.defender, air_strike)
    _, remaining_attackers = apply_damage(init.attacker, firing_defenders)
    _, remaining_defenders = apply_damage(
            firing_defenders,
            remaining_attackers,
            carryover
    )
    return BattleStanding(remaining_attackers, remaining_defenders)

def battle_round_outcome_distribution(
        init: BattleStanding, air_strike: BattleTroops
):
    """Return a dictionary of probabilities of different outcomes from
    one round of battle.  Keys of the dictionary are the different
    outcomes (as BattleStanding objects), values are the probability of
    that outcome."""
    result = defaultdict(int)
    for _ in range(trials):
        result[battle_round_outcome(init, air_strike)] += 1./trials
    return result

def repeated_battle_distribution(
        nrounds: int, init: BattleStanding, air_strike: BattleTroops
):
    """Return a dictionary of probabilities of different outcomes from
    multiple consecutive rounds of battle (given by nrounds)."""
    if nrounds <= 0: return {init: 1.}
    result = defaultdict(int)
    for standing, probability in battle_round_outcome_distribution(
            init, air_strike).items():
        result_from_standing = repeated_battle_distribution(
                nrounds - 1, standing, air_strike)
        for end_standing, end_probability in result_from_standing.items():
            result[end_standing] += probability * end_probability
    return result

## helpers for watching outcome distributions

def order_by_likelihood(outcome_distribution):
    return sorted(((prob, standing)
            for standing, prob in outcome_distribution.items()),
            key = lambda x: -x[0])

def repeated_round_results(
        nrounds: int, init: BattleStanding, air_strike: BattleTroops
):
    return order_by_likelihood(
            repeated_battle_distribution(nrounds, init, air_strike))

## handlers for extended outcomes.

# extended outcomes are lists of (win_probablitity, loss_probability)
# pairs, where the place in list (0, 1, 2, ...) means how many rounds
# the battle has taken.  For instance, if the value at index 2 is (.8,
# .05), then the battle has 80% probability of ending in a win after two
# battle rounds, and 5% probablitity of ending in a loss after two
# rounds.  The probabilities of ending after other rounds are given at
# other indices in the list.

def is_defeated(troops: BattleTroops):
    return sum(troops.triple_fire_cv \
            + troops.double_fire_cv \
            + troops.single_fire_cv) == 0

def outcome_sum(outcome1, outcome2, weight1, weight2):
    """Takes two extended outcomes and returns their weigted sum."""
    return [(weight1*victory1+weight2*victory2, weight1*defeat1+weight2*defeat2)
            for (victory1, defeat1), (victory2, defeat2)
            in zip_longest(outcome1, outcome2, fillvalue=(0.,0.))]

def recursive_regression(extended_outcome, recurse_probability):
    """Returns an extended outcome based on its input and the fact that
    there is recurse_probability possibility that nothing happens in the
    battle on one round."""
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
    """Produces an extended outcome from when battle is started with
    init and continued until either attacker or defender dies."""
    if is_defeated(init.attacker): return [(0.,1.)]
    if is_defeated(init.defender): return [(1.,0.)]
    one_round = battle_round_outcome_distribution(init, air_strike)
    try:
        recurse_probability = one_round.pop(init)
    except KeyError:
        recurse_probability = 0.000000001
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

def results(init: BattleStanding, air_strike: BattleTroops):
    return interpret_extended_outcome(
            extended_outcome_distribution(init, air_strike))

