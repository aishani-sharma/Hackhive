# Assumed, not measured: the share of brake efficiency each fault removes at full severity. The dashboard sliders
# let the user change these.
DEFAULT_LOSS_PER_SEVERITY = {"misfire": 0.25, "knock": 0.08, "valve_wear": 0.03}


def efficiency_loss(fault, severity, rates=None) -> float:
    """Fraction of efficiency lost: rate x severity, added over the faults present in a label like "misfire+knock"."""
    table = {**DEFAULT_LOSS_PER_SEVERITY, **(rates or {})}
    return sum(table[f] for f in fault.split("+") if f != "healthy") * severity


def degraded_efficiency(healthy_efficiency, fault, severity, rates=None) -> float:
    return healthy_efficiency * (1 - efficiency_loss(fault, severity, rates))
