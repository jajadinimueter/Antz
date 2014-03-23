__all__ = ('aslist', )


def aslist(val):
    if val is None:
        return []
    if not isinstance(val, (set, list, tuple)):
        return [val]
    else:
        return val