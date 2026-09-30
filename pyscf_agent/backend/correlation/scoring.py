"""Shared heuristic strength categories, not convergence criteria."""


def correlation_strength_level(score: float) -> str:
    if score >= 0.67:
        return 'strong'
    if score >= 0.35:
        return 'moderate'
    return 'weak'
