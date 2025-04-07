import functools
import math
import random
import numpy as np
from copy import copy
from typing import List, Union, Generator
from dataclasses import dataclass
from collections import defaultdict

from .basic import (
    max_flight_plan_len_for_ship_count,
    create_launch_fleet_command,
)
from .geometry import Point, Convert, ALL_ACTIONS
from .board import (
    Player,
    BoardRoute,
    PlanPath,
    PlanRoute,
    Shipyard,
    Fleet,
    DoNothing,
    Launch,
)
from .logger import logger


def get_encrypted_actions(agent: Player):
    shipyard_id_to_action = {}
    for sy in agent.shipyards:
        action = sy.action
        if not action or isinstance(action, DoNothing):
            continue

        initial_flight_plan = action.to_str()
        if isinstance(action, Launch):
            if action.is_convert_action():
                flight_plan = _encrypt_convert_action(action)
            else:
                flight_plan = _encrypt_route_action(action)
        else:
            flight_plan = action.to_str()

        if flight_plan != initial_flight_plan:
            logger.info(f"Encrypt action: {initial_flight_plan} -> {flight_plan}.")

        shipyard_id_to_action[sy.game_id] = flight_plan

    return shipyard_id_to_action


def _encrypt_route_action(
    action: Launch, encrypt_p: float = 0.5, fake_convert_p: float = 0
) -> str:
    max_str_action_len = max_flight_plan_len_for_ship_count(action.ship_count)

    original_plan = copy(action.route.plan)
    plan = copy(action.route.plan)

    if fake_convert_p and action.ship_count < 50:
        for _ in range(3):
            if random.random() > fake_convert_p:
                break

            if plan.command_length() >= max_str_action_len:
                break

            paths = plan.paths

            i = random.randint(1, min(4, len(paths)))
            paths = paths[:i] + [PlanPath(Convert, 0)] + paths[i:]
            plan = PlanRoute(paths)

    if plan.command_length() > max_str_action_len:
        plan = copy(original_plan)
    else:
        original_plan = copy(plan)

    for _ in range(3):
        if random.random() > encrypt_p:
            break

        if plan.command_length() >= max_str_action_len:
            break

        command = random.choice(list(ALL_ACTIONS))

        if command == Convert:
            num_steps = 0
        else:
            if plan.command_length() - max_str_action_len == 1:
                num_steps = 1
            else:
                num_steps = random.randint(1, 10)

        paths = plan.paths + [PlanPath(command, num_steps)]

        plan = PlanRoute(paths)

    if plan.command_length() > max_str_action_len:
        plan = copy(original_plan)

    return create_launch_fleet_command(action.ship_count, plan=plan.to_str())


def _encrypt_convert_action(action: Launch, encrypt_p: float = 0.5) -> str:
    max_str_action_len = max_flight_plan_len_for_ship_count(action.ship_count)

    original_plan = copy(action.route.plan)

    str_action = action.route.plan.to_str()
    while True:
        if random.random() > encrypt_p:
            break

        if len(str_action) >= max_str_action_len:
            break

        commands = [x.command for x in ALL_ACTIONS]

        str_action += random.choice(
            [
                *commands,
                *commands,
                *[str(i) for i in range(10)],
            ]
        )

    if len(str_action) > max_str_action_len:
        str_action = original_plan.to_str()

    return create_launch_fleet_command(action.ship_count, plan=str_action)


def is_invitable_victory(player: Player):
    if not player.opponents:
        return True

    board = player.board
    if board.steps_left > 100:
        return False

    board_kore = (
        sum(board.kore_at_point(x) for x in board)
        * (1 + board.regen_rate) ** board.steps_left
    )

    player_kore = player.kore + player.fleet_expected_kore()
    opponent_kore = max(x.kore + x.fleet_expected_kore() for x in player.opponents)
    return player_kore > opponent_kore + board_kore


@dataclass
class Interation:
    time: int
    object: Union[Fleet, Shipyard]
    hostile: bool
    is_direct_interaction: bool = True
    ship_count: int = None
    in_contact: bool = False

    def __post_init__(self):
        if self.ship_count is None:
            self.ship_count = self.object.ship_count

    @property
    def object_id(self):
        return self.object.game_id

    @property
    def player_id(self):
        return self.object.player_id

    @property
    def with_shipyard(self):
        return isinstance(self.object, Shipyard)


def find_route_interactions(
    agent: Player, route: BoardRoute
) -> Generator[Interation, None, None]:
    return find_points_interactions(agent, route.points[:-1])


def find_points_interactions(
    agent: Player,
    points: List[Point],
    fleet_interation=True,
    adjacent_interation=True,
    shipyard_interation=True,
) -> Generator[Interation, None, None]:
    agent_id = agent.game_id
    board = agent.board

    players_data = []
    for player_id, player in board.id_to_player.items():
        hostile = player_id != agent_id
        players_data.append((hostile, player.expected_positions))

    already_interacted = set()
    for time, point in enumerate(points, 1):
        for hostile, data in players_data:

            step_data = data[time]

            if point not in step_data:
                continue

            fleet, ad_fleets, shipyard = step_data[point]

            if not hostile:

                if (
                    fleet_interation
                    and fleet
                    and fleet.game_id not in already_interacted
                ):
                    yield Interation(
                        object=fleet,
                        time=time,
                        hostile=False,
                    )
                    already_interacted.add(fleet.game_id)

                if shipyard_interation and shipyard:
                    yield Interation(
                        object=shipyard,
                        time=time,
                        hostile=False,
                    )

            else:

                if (
                    fleet_interation
                    and fleet
                    and fleet.game_id not in already_interacted
                ):
                    yield Interation(
                        object=fleet,
                        time=time,
                        hostile=True,
                    )
                    already_interacted.add(fleet.game_id)

                if adjacent_interation:
                    for f, count, in_contact in ad_fleets:
                        if f.game_id not in already_interacted:
                            yield Interation(
                                object=f,
                                time=time,
                                hostile=True,
                                is_direct_interaction=False,
                                ship_count=count,
                                in_contact=in_contact,
                            )
                            already_interacted.add(f.game_id)

                if shipyard_interation and shipyard:
                    yield Interation(
                        object=shipyard,
                        time=time,
                        hostile=True,
                    )


def get_interaction_list_for_route(
    agent: Player, route: BoardRoute
) -> List[Interation]:
    return list(find_route_interactions(agent, route))


class Balancer:
    def __init__(self, balance=None):
        self.balance = balance or []

    def __repr__(self):
        return f"Balancer({self.balance})"

    # @property
    # def balance(self):
    #     balance = []
    #     for time, fleet in self.time_to_fleet.items():
    #         if len(balance) <= time:
    #             balance += [0 for _ in range(1 + time - len(balance))]
    #
    #         balance[time] += fleet.ship_count

    @classmethod
    def from_interactions(
        cls, agent: Player, interactions: List[Interation]
    ) -> "Balancer":
        b = Balancer()
        for interaction in sorted(interactions, key=lambda x: x.time):
            if interaction.player_id == agent.game_id:
                s = 1
            else:
                s = -1
            b.add_value(interaction.time, s * interaction.ship_count)
        return b

    def needs(self):
        min_b = 0
        b = 0
        time = None
        for t, x in enumerate(self.balance):
            b += x
            if b < 0:
                if time is None:
                    time = t
                if b < min_b:
                    min_b = b
        return time, -min_b

    def is_good(self):
        b = 0
        for x in self.balance:
            b += x
            if b < 0:
                return False
        return True

    def value(self):
        min_b = self.balance[0]
        b = 0
        for x in self.balance:
            b += x
            if b < min_b:
                min_b = b
        return min_b

    def add_value(self, time, value):
        if len(self.balance) <= time:
            self.balance += [0 for _ in range(1 + time - len(self.balance))]

        self.balance[time] += value


def estimate_point_values(agent: Player, as_array=False):
    agent_shipyards = {x.point: x for x in agent.shipyards}
    if agent.opponent:
        opponent_shipyards = {x.point: x for x in agent.opponent.shipyards}
    else:
        opponent_shipyards = {}

    board = agent.board

    agent_points = []
    op_points = []
    sy_to_points = defaultdict(list)
    for p, data in board.point_to_data.items():
        agent_sys, agent_distance = p.find_closest_points(agent_shipyards)
        op_sys, op_distance = p.find_closest_points(opponent_shipyards)

        if agent_distance is None and op_distance is None:
            continue

        if agent_distance is None:
            agent_distance = math.inf
        if op_distance is None:
            op_distance = math.inf

        if agent_distance > op_distance:
            op_points.append((p, data.kore, op_distance))
            for sy_point in op_sys:
                sy_to_points[opponent_shipyards[sy_point]].append((p, data.kore))
        elif agent_distance < op_distance:
            agent_points.append((p, data.kore, agent_distance))
            for sy_point in agent_sys:
                sy_to_points[agent_shipyards[sy_point]].append((p, data.kore))
        else:
            op_points.append((p, data.kore, op_distance))
            agent_points.append((p, data.kore, agent_distance))
            for sy_point in agent_sys:
                sy_to_points[agent_shipyards[sy_point]].append((p, data.kore))
            for sy_point in op_sys:
                sy_to_points[opponent_shipyards[sy_point]].append((p, data.kore))

    point_to_value = {}
    for p in board.point_to_data:
        flip_count = 0
        for p1, kore, distance in op_points:
            d = p.distance_from(p1)
            if 0 < d < distance:
                flip_count += kore / d

        for p1, kore, distance in agent_points:
            d = p.distance_from(p1)
            if 0 < d < distance:
                flip_count += kore / d * (distance - d) / distance

        point_to_value[p] = flip_count

    shipyard_to_value = {}
    for sy in board.shipyards:
        sy_points = sy_to_points[sy]
        flip_count = 0
        for p, kore in sy_points:
            d = p.distance_from(sy.point)
            if d > 0:
                flip_count += kore / d
        shipyard_to_value[sy] = flip_count

    if as_array:
        s = agent.board.field.size
        point_to_value_array = np.zeros((s, s), dtype=np.int32)
        for p, v in point_to_value.items():
            point_to_value_array[s - 1 - p.y, p.x] = v

        shipyard_to_value_array = np.zeros((s, s), dtype=np.int32)
        for sy, v in shipyard_to_value.items():
            shipyard_to_value_array[s - 1 - sy.point.y, sy.point.x] = v

        return point_to_value_array, shipyard_to_value_array

    return point_to_value, shipyard_to_value


def need_to_save_time(steps_left, remaining_time, max_steps=400, time_bank=60):
    if remaining_time < 5:
        return True
    if steps_left / max_steps > remaining_time / time_bank:
        return True
    return False


def action_check():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(agent: Player, *args, **kwargs):
            shipyard_with_actions = {sy.game_id for sy in agent.shipyards if sy.action}
            func(agent, *args, **kwargs)
            new_actions = any(
                sy.action and sy.game_id not in shipyard_with_actions
                for sy in agent.shipyards
            )
            return new_actions

        return wrapper

    return decorator
