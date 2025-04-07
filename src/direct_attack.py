import math
import random
from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass

from .geometry import Point
from .board import Player, Shipyard, Launch, Fleet, BoardRoute
from .logger import logger
from .helpers import get_interaction_list_for_route, action_check


@dataclass
class FleetTarget:
    shipyard: Shipyard
    fleet: Fleet
    hit_point: Point
    hit_time: int

    def __post_init__(self):
        final_fleet_state = self.fleet.get_state(self.hit_time)

        self.fleet_ship_count = final_fleet_state.ship_count
        self.attack_distance = self.shipyard.point.distance_from(self.hit_point)
        self.launch_time = self.hit_time - self.attack_distance
        self.is_direct_attack = final_fleet_state.point == self.hit_point
        min_fleet_size_for_roundtrip = min(
            x.min_fleet_size()
            for x in self.shipyard.point.roundtrip_routes(self.hit_point)
        )

        self.min_fleet_size = max(
            min_fleet_size_for_roundtrip, self.fleet_ship_count + 1
        )

        if self.launch_time > 0:
            shipyard_ship_count = self.shipyard.get_state(self.launch_time).ship_count
        else:
            shipyard_ship_count = self.shipyard.available_ship_count

        self.additional_ships_needed = max(self.min_fleet_size - shipyard_ship_count, 0)

        self.score = self.score()

    def __repr__(self):
        return (
            f"FleetTarget(shipyard={self.shipyard.point.to_tuple()}, fleet={self.fleet.point.to_tuple()} "
            f"hit_point={self.hit_point.to_tuple()}, hit_time={self.hit_time}, launch_time={self.launch_time}, "
            f"min_fleet_size={self.min_fleet_size}, is_direct_attack={self.is_direct_attack})"
        )

    def score(self):

        d = 0
        if self.attack_distance < 6:
            if not self.is_direct_attack:
                d += 2

        else:
            if self.is_direct_attack:
                d += 2

        score = -self.attack_distance + d
        return score


@action_check()
def attack_enemy_fleets(agent: Player):
    targets = _find_target_fleets(agent)

    fleet_to_targets = defaultdict(list)
    for t in targets:
        fleet_to_targets[t.fleet].append(t)

    for fleet in sorted(fleet_to_targets, key=lambda f: -f.expected_value()):
        fleet_targets: List[FleetTarget] = fleet_to_targets[fleet]
        if not fleet_targets:
            continue

        fleet_targets = sorted(fleet_targets, key=lambda x: -x.score)

        # logger.debug(f"{fleet_targets[:3]}")

        for t in fleet_targets:
            if t.launch_time == 0:
                if t.shipyard.action:
                    continue
                action = _attack_enemy_fleet(t, agent.route_checker)
                if action:
                    t.shipyard.action = action
                    break
            else:
                t.shipyard.set_guard_ship_count(
                    min(t.shipyard.ship_count, int(t.min_fleet_size * 1.2))
                )
                t.shipyard.tasks.append(t)
                break


def _attack_enemy_fleet(target: FleetTarget, route_checker):
    shipyard = target.shipyard
    board = shipyard.board
    agent = shipyard.player
    start = shipyard.point
    hit_point = target.hit_point

    # logger.debug(f"{target}")

    p2t_plans = [
        p for p in start.plan_to(hit_point) if p.num_steps == target.attack_distance
    ]
    t2d_plans = []
    for sy in agent.shipyards:
        total_distance = target.attack_distance + sy.distance_from(hit_point)
        next_sy = board.get_shipyard(shipyard.game_id, total_distance)
        if next_sy and next_sy.player_id != sy.player_id:
            continue

        t2d_plans += target.hit_point.plan_to(sy.point)

    routes = []
    for p2t in p2t_plans:
        p2t_route = BoardRoute(start, p2t)

        for t2d in t2d_plans:
            t2d_route = BoardRoute(hit_point, t2d)

            route = p2t_route + t2d_route

            num_ships_to_launch = max(
                route.plan.min_fleet_size(),
                route_checker.safety_ship_count(route) + 1,
            )
            if shipyard.available_ship_count < num_ships_to_launch:
                continue

            interaction_ship_count = check_direct_attack_interactions(
                agent, route, target, shipyard.available_ship_count
            )
            if interaction_ship_count is None:
                continue

            if shipyard.available_ship_count < interaction_ship_count:
                continue

            num_ships_to_launch = min(
                shipyard.available_ship_count,
                max(
                    int(num_ships_to_launch * 1.2),
                    interaction_ship_count + 7,
                ),
            )
            if shipyard.available_ship_count < num_ships_to_launch:
                continue

            expected_kore = route.expected_kore(shipyard.player, num_ships_to_launch)
            routes.append(
                {
                    "route": route,
                    "ship_count": num_ships_to_launch,
                    "score": expected_kore
                    / math.sqrt(num_ships_to_launch)
                    / len(route.points),
                }
            )

    if not routes:
        return

    max_score = max(x["score"] for x in routes)
    route = random.choice([r for r in routes if r["score"] == max_score])
    logger.info(
        f"Attack enemy fleet {shipyard.point}->{target.hit_point}, distance={target.attack_distance}, "
        f"num_ships={route['ship_count']}vs{target.fleet_ship_count}"
    )
    return Launch(route["ship_count"], route["route"])


def check_direct_attack_interactions(agent, route, target, available_ship_count):
    target_fleet_id = target.fleet.game_id
    target_distance = target.attack_distance
    interactions = get_interaction_list_for_route(agent, route)
    if any(x.with_shipyard for x in interactions):
        return

    if any(x.hostile and x.in_contact for x in interactions):
        logger.debug(f"Ignore route {route} with interactions {interactions}")
        return

    # logger.debug(f"{route}, {interactions}")

    route_hit_times = [x.time for x in interactions if x.object_id == target_fleet_id]
    assert (
        len(route_hit_times) == 1
    ), f"route={route}, target={target_fleet_id}, interactions={interactions}"
    if route_hit_times[0] != target_distance:
        return

    min_ship_count = None
    for ship_count in range(1, available_ship_count + 1):
        out_ship_count = _simulate_direct_attack(interactions, ship_count)
        if out_ship_count is None:
            continue
        min_ship_count = ship_count
        break

    return min_ship_count


def _simulate_direct_attack(interactions, ship_count: int):
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


def _find_target_fleets(agent: Player) -> List[FleetTarget]:
    opponent = agent.opponent
    if not opponent:
        return []

    targets = []
    for fleet in opponent.fleets:
        final_state = fleet.final_state
        operations = final_state.operations
        if operations.direct_attack or operations.adjacent_attack:
            if operations.ships_hit < operations.ships_lost:
                continue

        for sy in agent.shipyards:
            time_to_points = _find_shipyard_to_fleet_time_points(fleet, sy)
            for time, points in time_to_points.items():
                for point in points:
                    targets.append(
                        FleetTarget(
                            fleet=fleet, shipyard=sy, hit_point=point, hit_time=time
                        )
                    )
    return targets


def _find_shipyard_to_fleet_time_points(
    fleet: Fleet, shipyard: Shipyard
) -> Dict[int, List[Point]]:
    sy_point = shipyard.point

    time_to_points = defaultdict(list)

    for time, state in enumerate(fleet.states):
        fleet_point = state.point
        shipyard_points = state.board.shipyard_positions
        if fleet_point in shipyard_points:
            continue

        if sy_point.distance_from(fleet_point) <= time:
            time_to_points[time].append(fleet_point)

        for adjacent_point in fleet_point.adjacent_points:
            if adjacent_point in shipyard_points:
                continue

            if sy_point.distance_from(adjacent_point) <= time:
                time_to_points[time].append(adjacent_point)

    return time_to_points
