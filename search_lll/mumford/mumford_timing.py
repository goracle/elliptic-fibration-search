from collections import defaultdict


def mumford_timer_add(name, elapsed):
    _MUMFORD_TIMERS[name] += elapsed

# Timers
_MUMFORD_TIMERS = defaultdict(float)

def mumford_timers_reset():
    global _MUMFORD_TIMERS
    _MUMFORD_TIMERS.clear()

def mumford_timers_print():
    if not _MUMFORD_TIMERS:
        return
    print("\n[mumford detailed timers]")
    items = sorted(_MUMFORD_TIMERS.items(), key=lambda x: x[1], reverse=True)
    total = sum(t for _, t in items)
    for name, t in items:
        pct = 100.0 * t / total if total > 0 else 0.0
        print(f"  {name:40s}: {t:8.3f}s ({pct:5.1f}%)")
    print(f"  {'TOTAL':40s}: {total:8.3f}s")


def mumford_timer_get(name):
    """Get timer value."""
    return _MUMFORD_TIMERS.get(name, 0.0)

