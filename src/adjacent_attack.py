import math
import random
from typing import List
from collections import defaultdict
from dataclasses import dataclass

from .geometry import Point
from .board import Player, Launch, Fleet, BoardRoute, Shipyard
from .helpers import action_check, get_interaction_list_for_route
from .logger import logger


@dataclass
class AdjacentTarget:
    shipyard: Shipyard
    fleets: List[Fleet]
    hit_point: Point
    hit_time: int

    def __post_init__(self):
        final_fleets_state = [fleet.get_state(self.hit_time) for fleet in self.fleets]

        self.fleet_sizes = [x.ship_count for x in final_fleets_state]
        self.max_fleet_size = max(self.fleet_sizes)
        self.min_fleet_size = min(self.fleet_sizes)
        self.size_diff = self.max_fleet_size - self.min_fleet_size

        self.fleet_ship_count = sum(self.fleet_sizes)
        self.attack_distance = self.shipyard.point.distance_from(self.hit_point)
        self.launch_time = self.hit_time - self.attack_distance
        self.is_direct_attack = False

        self.score = self.score()

    def __repr__(self):
        return (
            f"AdjacentTarget(shipyard={self.shipyard.point.to_tuple()}, "
            f"fleets={[fleet.point.to_tuple() for fleet in self.fleets]} "
            f"hit_point={self.hit_point.to_tuple()}, hit_time={self.hit_time}, launch_time={self.launch_time}, "
            f"min_fleet_size={self.min_fleet_size})"
        )

    def score(self):
        if self.launch_time == 0 and self.shipyard.action:
            return -math.inf

        # if self.additional_ships_needed == 0:
        #     launch_score = 0
        # elif self.launch_time < 1:
        #     launch_score = -math.inf
        # else:
        #     launch_score = -self.additional_ships_needed / self.launch_time
        launch_score = 0

        score = -self.attack_distance + 2 * self.is_direct_attack + launch_score
        return score


@action_check()
def adjacent_attack(agent: Player, max_distance: int = 14, anti_siege=False):
    board = agent.board

    max_distance = min(board.steps_left, max_distance)

    if anti_siege:
        siege_fleets = []
        for sy in agent.shipyards:
            siege_fleets += [x.game_id for x in sy.incoming_hostile_fleets]
        if not siege_fleets:
            return

        attack_points = _find_adjacent_attack_points(agent, max_distance)
        if not attack_points:
            return

        tasks = _create_adjacent_attack_tasks(agent, attack_points)
        tasks = [x for x in tasks if x.launch_time == 0]

        tasks = [t for t in tasks if any(x.game_id in siege_fleets for x in t.fleets)]
        workers = {t.shipyard for t in tasks if t.launch_time == 0}
        for t in tasks:
            if t.launch_time > 0 and t.shipyard not in workers:
                logger.info(
                    f"Anti siege mode: preparing fleet at {t.shipyard.point} for task {t}"
                )
                t.shipyard.set_guard_ship_count(t.min_fleet_size)
        tasks = [x for x in tasks if x.launch_time == 0]

    else:
        attack_points = _find_adjacent_attack_points(agent, max_distance)
        if not attack_points:
            return

        tasks = _create_adjacent_attack_tasks(agent, attack_points)
        tasks = [x for x in tasks if x.launch_time == 0]

    # logger.debug(f"Adjacent attack tasks = {tasks}")

    fleets_to_be_attacked = set()
    for task in sorted(tasks, key=lambda x: x.size_diff):
        if task.shipyard.action:
            continue

        fleets = task.fleets
        if any(x.game_id in fleets_to_be_attacked for x in fleets):
            continue

        action = _do_adjacent_attack(task)
        if action:
            log_dict = dict(
                distance=task.attack_distance,
                num_ships=f"{action.ship_count}vs{'+'.join([str(x) for x in task.fleet_sizes])}",
            )
            if anti_siege:
                log_dict["anti_siege"] = True

            logger.info(
                f"Adjacent attack {task.shipyard.point}->{task.hit_point}: {log_dict}"
            )
            task.shipyard.action = action
            for fleet in task.fleets:
                fleets_to_be_attacked.add(fleet.game_id)


def _do_adjacent_attack(target: AdjacentTarget):
    shipyard = target.shipyard
    available_ship_count = shipyard.ship_count
    agent = shipyard.player
    start = shipyard.point
    hit_point = target.hit_point

    routes = []
    for p2t in start.plan_to(hit_point):
        if p2t.num_steps != target.attack_distance:
            continue

        route = BoardRoute(start, p2t)

        num_ships_to_launch = max(
            route.plan.min_fleet_size(),
            3,  # route_checker.safety_ship_count(route) + 1,
        )
        if available_ship_count < num_ships_to_launch:
            continue

        interaction_ship_count = check_adjacent_attack_interactions(
            agent, route, target, available_ship_count
        )
        if interaction_ship_count is None:
            continue

        # logger.info(f"{route} {interaction_ship_count} {num_ships_to_launch} {shipyard.available_ship_count}")

        if (
            interaction_ship_count < num_ships_to_launch
            or interaction_ship_count > available_ship_count
        ):
            continue

        num_ships_to_launch = interaction_ship_count
        expected_kore = route.expected_kore(shipyard.player, num_ships_to_launch)

        routes.append(
            {
                "route": route,
                "num_ships_to_launch": num_ships_to_launch,
                "expected_kore": expected_kore,
            }
        )

    if not routes:
        return

    min_kore = min(x["expected_kore"] for x in routes)

    route = random.choice([x for x in routes if x["expected_kore"] == min_kore])

    return Launch(route["num_ships_to_launch"], route["route"])


def check_adjacent_attack_interactions(agent, route, target, available_ship_count):
    target_fleet_ids = [x.game_id for x in target.fleets]
    target_distance = target.attack_distance

    interactions = get_interaction_list_for_route(agent, route)

    if any(x.with_shipyard for x in interactions):
        return

    if any(x.hostile and x.in_contact for x in interactions):
        return

    route_hit_times = [x.time for x in interactions if x.object_id in target_fleet_ids]
    if route_hit_times:
        return

    ship_count_to_out = {}
    for ship_count in range(1, available_ship_count + 1):
        out_ship_count = _simulate_adjacent_attack(interactions, ship_count)
        if out_ship_count is None:
            continue
        # if target.max_fleet_size <= out_ship_count <= 1.1 * target.max_fleet_size:
        #     min_ship_count = ship_count
        #     break
        if out_ship_count <= 1 * target.min_fleet_size:
            ship_count_to_out[ship_count] = out_ship_count

    # logger.info(f"{route}, {ship_count_to_out}")

    if not ship_count_to_out:
        return

    max_out = max(ship_count_to_out.values())

    return max([x for x, out in ship_count_to_out.items() if out == max_out])


def _simulate_adjacent_attack(interactions, ship_count: int):
    for interaction in interactions:
        if not interaction.hostile:
            if interaction.ship_count >= ship_count:
                return
            else:
                ship_count += interaction.ship_count
        else:
            if interaction.ship_count >= ship_count:
                return
            else:
                ship_count -= interaction.ship_count

                if ship_count <= 0:
                    return

    return ship_count


def _find_adjacent_attack_points(agent: Player, max_distance: int = 10):
    board = agent.board

    fleets_to_attack = []
    for f in board.id_to_fleet.values():
        if f.player_id == agent.game_id:
            continue

        final_state = f.final_state
        operations = final_state.operations
        if operations.direct_attack or operations.adjacent_attack:
            if operations.ships_hit < operations.ships_lost:
                continue

        fleets_to_attack.append(f)

    if not fleets_to_attack:
        return []

    shipyards_points = {x.point for x in board.shipyards}

    time = 0
    targets = []
    while time <= max_distance:
        time += 1

        point_to_fleet = {}
        for f in fleets_to_attack:
            try:
                state = f.states[time]
            except IndexError:
                continue
            if state.point in shipyards_points:
                continue
            point_to_fleet[state.point] = f

        if not point_to_fleet:
            break

        for point in board:
            if point in point_to_fleet or point in shipyards_points:
                continue

            adjacent_fleets = [
                point_to_fleet[x] for x in point.adjacent_points if x in point_to_fleet
            ]
            if len(adjacent_fleets) < 2:
                continue

            targets.append({"point": point, "time": time, "fleets": adjacent_fleets})

    return targets


def _create_adjacent_attack_tasks(agent: Player, target_points):
    tasks = []
    for target in target_points:
        hit_point = target["point"]
        hit_time = target["time"]
        fleets = target["fleets"]

        for sy in agent.shipyards:
            distance = hit_point.distance_from(sy.point)
            if distance > hit_time:
                continue

            task = AdjacentTarget(
                shipyard=sy, fleets=fleets, hit_point=hit_point, hit_time=hit_time
            )
            tasks.append(task)
    return tasks
