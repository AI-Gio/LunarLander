from dataclasses import dataclass

@dataclass
class Transition:
    state: State
    action: float
    reward: float
    next_state= State
    done = bool