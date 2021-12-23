from dataclasses import dataclass


@dataclass
class Transition:
    state: list
    action: float
    reward: float
    next_state: list
    done: bool
