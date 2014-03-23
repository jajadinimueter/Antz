import random


__all__ = ('aslist', 'probability_choice')


def probability_choice(l):
    if not l:
        return None
    s = sum([b for (a,b) in l])
    if not s:
        return random.choice(l)[0]
    l2 = []
    # Make the probabilities add up to 1, preserving ratios
    for (a,b) in l:
        l2.append((a, b/s))
    random.shuffle(l2)
    r = random.random()
    for (a,b) in l2:
        if r < b:
            return a
        else:
            r -= b


def aslist(val):
    if val is None:
        return []
    if not isinstance(val, (set, list, tuple)):
        return [val]
    else:
        return val