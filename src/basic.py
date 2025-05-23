import math
from typing import Union

from kaggle_environments.envs.kore_fleets.helpers import SPAWN_VALUES


def max_ships_to_spawn(turns_controlled: int) -> int:
    for idx, target in enumerate(SPAWN_VALUES):
        if turns_controlled < target:
            return idx + 1
    return len(SPAWN_VALUES) + 1


def max_flight_plan_len_for_ship_count(ship_count: int) -> int:
    return math.floor(2 * math.log(ship_count)) + 1


def min_ship_count_for_flight_plan_len(flight_plan_len: int) -> int:
    return math.ceil(math.exp((flight_plan_len - 1) / 2))


def collection_rate_for_ship_count(ship_count: int) -> float:
    return min(math.log(ship_count) / 20, 0.99)


def create_spawn_ships_command(num_ships: int) -> str:
    assert num_ships > 0, "num_ships must be a positive"
    return f"SPAWN_{num_ships}"


def create_launch_fleet_command(num_ships: int, plan: str) -> str:
    assert num_ships > 0, "num_ships must be a positive"
    return f"LAUNCH_{num_ships}_{plan}"


def next_step_kore(cell_kore: float, regen_rate: float, max_cell_kore: float) -> float:
    return (
        cell_kore
        if cell_kore >= max_cell_kore
        else round(cell_kore * (1 + regen_rate), 3)
    )


def amount_of_collected_kore(cell_kore, collection_rate):
    return round(cell_kore * min(collection_rate, 0.99), 3)


class cached_property:
    """
    python 3.9:
    >>> from functools import cached_property
    """

    def __init__(self, func):
        self.func = func
        self.key = "__" + func.__name__

    def __get__(self, instance, owner):
        try:
            return instance.__getattribute__(self.key)
        except AttributeError:
            value = self.func(instance)
            instance.__setattr__(self.key, value)
            return value


class cached_call:
    """
    may cause a memory leak, be careful
    """

    def __init__(self, func):
        self.func = func
        self.key = "__" + func.__name__

    def __get__(self, instance, owner):
        try:
            d = instance.__getattribute__(self.key)
        except AttributeError:
            d = {}
            instance.__setattr__(self.key, d)

        def func(*args):
            try:
                return d[args]
            except KeyError:
                value = self.func(instance, *args)
                d[args] = value
                return value

        return func


class Obj:
    def __init__(self, game_id: Union[str, int]):
        self._game_id = game_id

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._game_id})"

    @property
    def game_id(self):
        return self._game_id
