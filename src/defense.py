import math
import random
from typing import List

from .board import Spawn, Player, Launch, Shipyard, BoardRoute, Fleet
from .helpers import find_route_interactions, action_check, Balancer
from .logger import logger


@action_check()
def defend_shipyards(agent: Player):
    board = agent.board

    sy_to_takeover_time = {}
    for sy in agent.shipyards:
        if sy.incoming_hostile_fleets:
            states = sy.states
            for time, state in enumerate(states):
                if state.player_id != agent.game_id:
                    sy_to_takeover_time[sy] = time
                    break
            if sy not in sy_to_takeover_time:
                sy_to_takeover_time[sy] = math.inf

    need_help_shipyards = []
    for sy in sorted(sy_to_takeover_time, key=lambda x: sy_to_takeover_time[x]):
        if sy.action:
            continue

        balance_value = sy.balancer.value()
        if balance_value > 0:
            sy.set_guard_ship_count(sy.ship_count - balance_value)
            continue

        # spawn as much as possible
        # num_ships_to_spawn = min(
        #     int(agent.available_kore() // board.spawn_cost), sy.max_ships_to_spawn
        # )

        # if num_ships_to_spawn and board.steps_left > 20:
        #     pass
        #     # logger.debug(f"Spawn ships to protect shipyard {sy.point}")
        #     # sy.action = Spawn(num_ships_to_spawn)
        # else:
        sy.set_guard_ship_count(sy.ship_count)

        need_help_shipyards.append(sy)
        # logger.debug(f"Need help: {need_help_shipyards}")

    if need_help_shipyards:
        agent_shipyards = [
            sy
            for sy in agent.shipyards
            if sy not in need_help_shipyards
            and sy.available_ship_count
            and not sy.action
        ]
        for sy in need_help_shipyards:
            incoming_hostile_time, _ = sy.balancer.needs()
            hostile_fleets = [f.game_id for f in sy.incoming_hostile_fleets]
            # logger.debug(f"incoming_hostile_time: {incoming_hostile_time}")
            if incoming_hostile_time is None:
                continue

            for other_sy in agent_shipyards:
                if other_sy.action:
                    continue

                distance = other_sy.distance_from(sy)
                # logger.debug(f"{other_sy}, distance = {distance}")

                if distance in (incoming_hostile_time, incoming_hostile_time - 1):

                    routes = _find_shipyard_to_shipyard_routes_for_defense(other_sy, sy)
                    # logger.debug(f"help routes: {routes}")

                    if not routes:
                        continue

                    route_to_num_reinforcement = {}
                    for route in routes:
                        min_ship_count = route.plan.min_fleet_size()
                        if other_sy.available_ship_count < min_ship_count:
                            continue
                        num_reinforcement = (
                            _check_shipyard_to_shipyard_route_for_defense(
                                agent,
                                route,
                                other_sy.available_ship_count,
                                target_shipyard_id=sy.game_id,
                                target_distance=distance,
                                hostile_fleets=hostile_fleets,
                            )
                        )
                        if num_reinforcement:
                            route_to_num_reinforcement[route] = num_reinforcement

                        # logger.debug(f"route_to_num_reinforcement = {route_to_num_reinforcement}")

                    if route_to_num_reinforcement:
                        max_reinforcement = max(route_to_num_reinforcement.values())
                        routes = [
                            x
                            for x, v in route_to_num_reinforcement.items()
                            if v == max_reinforcement
                        ]

                        num_ships = other_sy.available_ship_count
                        logger.info(
                            f"Send reinforcements {other_sy.point}->{sy.point}, "
                            f"start_ships={num_ships}, end_ships={max_reinforcement}, eta={distance}"
                        )
                        other_sy.action = Launch(
                            other_sy.available_ship_count, random.choice(routes)
                        )
                elif distance < incoming_hostile_time - 1:
                    other_sy.set_guard_ship_count(other_sy.ship_count)


def _find_shipyard_to_shipyard_routes_for_defense(
    sy1: Shipyard, sy2: Shipyard
) -> List[BoardRoute]:
    board = sy1.board
    min_distance = sy1.distance_from(sy2)

    routes = []
    for p in board:
        distance = sy1.distance_from(p) + p.distance_from(sy2.point)
        if distance != min_distance:
            continue

        s2p_plans = sy1.point.plan_to(p)
        p2e_plans = p.plan_to(sy2.point)

        for s2p in s2p_plans:
            for p2e in p2e_plans:
                plan = s2p + p2e

                route = BoardRoute(sy1.point, plan)

                routes.append(route)

    return routes


def _check_shipyard_to_shipyard_route_for_defense(
    agent, route, ship_count, target_shipyard_id, target_distance, hostile_fleets
):
    interactions = find_route_interactions(agent, route)
    for interaction in interactions:

        if interaction.with_shipyard:
            if interaction.hostile and interaction.object_id != target_shipyard_id:
                return
            continue

        if interaction.time >= target_distance:
            continue

        if not interaction.hostile:
            if interaction.ship_count >= ship_count:
                return
            else:
                ship_count += interaction.ship_count
        else:
            if interaction.in_contact:
                return

            if interaction.object_id in hostile_fleets:
                continue

            if interaction.ship_count >= ship_count:
                return
            else:
                ship_count -= interaction.ship_count

        if ship_count <= 0:
            return

    return ship_count


def set_shipyard_defense(agent: Player):
    board = agent.board
    shipyards = list(agent.shipyards)

    for sy in shipyards:
        balancer = Balancer()

        balancer.add_value(0, sy.ship_count)

        if isinstance(sy.action, Spawn):
            balancer.add_value(1, sy.action.ship_count)
        elif isinstance(sy.action, Launch):
            balancer.add_value(1, -sy.action.ship_count)

        for f in sy.incoming_hostile_fleets:
            balancer.add_value(f.board.step - board.step + 1, -f.ship_count)

        for f in sy.incoming_allied_fleets:
            balancer.add_value(f.board.step - board.step + 1, f.ship_count)

        sy.balancer = balancer

    # if agent.shipyard_count <= 1:
    #     return

    for sy in shipyards:
        window, _ = sy.balancer.needs()
        if window is not None:
            sy.force_self_route = True
            sy.window = window
            sy.greedy_mining = True
            # logger.debug(f"Force self route: shipyard {sy.point}, window={window}")
            continue

        # opposite_shipyards = sy.opposite_shipyards
        # if opposite_shipyards:
        #     power = sum(x.ship_count for x in opposite_shipyards)
        #     distance = sy.distance_from(opposite_shipyards[0])
        #     balancer = Balancer(copy(sy.balancer.balance))
        #
        #     balancer.add_value(distance, -power)
        #
        #     for support in shipyards:
        #         d = sy.distance_from(support)
        #         if 0 < d <= distance:
        #             balancer.add_value(d, support.ship_count)
        #
        #     for t in range(1, distance):
        #         balancer.add_value(t, sy.max_ships_to_spawn)
        #
        #     free_ships = balancer.value()
        #     if free_ships < 21:
        #         # logger.debug(f"Force self route: shipyard={sy.point}, window={distance}, free_ships={free_ships}")
        #         sy.force_self_route = True
        #         sy.window = distance
        #     # else:
        #         # logger.debug(f"Set shipyard defense: shipyard={sy.point}, free_ships={free_ships}")
        #         # sy.set_guard_ship_count(sy.ship_count - free_ships)


def protect_fleets(agent: Player):

    for f in agent.fleets:
        fleet = f.final_state
        operations = fleet.operations
        if operations.shipyard_attack:
            continue

        op_fleet_ids = []
        if operations.direct_attack:
            op_fleet_ids += operations.direct_attack
        if operations.adjacent_attack:
            op_fleet_ids += operations.adjacent_attack

        if len(op_fleet_ids) != 1:
            continue

        op_fleet_id = op_fleet_ids[0]
        op_fleet = f.board.get_fleet(op_fleet_id, f.eta - 1)
        if not op_fleet:
            continue

        logger.debug(
            f"Need protection for the fleet {f.point} at {fleet.point}, "
            f"powers={fleet.ship_count}vs{op_fleet.ship_count}"
        )

        num_ships_to_send = op_fleet.ship_count - fleet.ship_count + 1
        if num_ships_to_send <= 0:
            continue

        routes = _find_protect_fleet_routes(
            f,
            min_ships_to_send=num_ships_to_send,
            max_ships_to_send=fleet.ship_count - 1,
        )
        if not routes:
            continue

        routes = sorted(
            routes,
            key=lambda x: (
                len(x["route"].points),
                -x["expected_kore"] / x["num_ships"],
            ),
        )
        route = routes[0]

        shipyard = route["shipyard"]
        num_ships = route["num_ships"]
        route = route["route"]

        logger.info(
            f"Sending ships to protect fleet: {shipyard.point}->{f}, distance={len(route.points)}."
        )
        shipyard.action = Launch(num_ships, route)


def _find_protect_fleet_routes(
    fleet: Fleet, min_ships_to_send: int, max_ships_to_send: int
):
    time_to_points = {}
    for time, state in enumerate(fleet.states):
        time_to_points[time] = state.point

    board = fleet.board
    agent = fleet.player

    routes = []
    for shipyard in fleet.player.shipyards:
        if shipyard.action or shipyard.available_ship_count < min_ships_to_send:
            continue

        shipyard_position = shipyard.point

        for time, fleet_point in time_to_points.items():
            sy_distance = shipyard_position.distance_from(fleet_point)
            if sy_distance != time:
                continue

            for p in board:
                if (
                    shipyard_position.distance_from(p) + p.distance_from(fleet_point)
                    != sy_distance
                ):
                    continue

                for sy2p in shipyard_position.plan_to(p):
                    sy2p_route = BoardRoute(shipyard_position, sy2p)
                    for p2f in p.plan_to(fleet_point):
                        p2f_route = BoardRoute(p, p2f)

                        route = sy2p_route + p2f_route

                        if route.plan.min_fleet_size() > shipyard.available_ship_count:
                            continue

                        num_ships_to_out = check_protect_fleet_interactions(
                            agent,
                            route,
                            target_fleet_id=fleet.game_id,
                            available_ship_count=shipyard.available_ship_count,
                        )
                        if not num_ships_to_out:
                            continue

                        num_ships_to_out = {
                            c: out
                            for c, out in num_ships_to_out.items()
                            if min_ships_to_send <= out <= max_ships_to_send
                            and c >= route.plan.min_fleet_size()
                        }
                        if not num_ships_to_out:
                            continue

                        min_out = min(num_ships_to_out.values())
                        num_ships = min(
                            c for c, out in num_ships_to_out.items() if out == min_out
                        )
                        expected_kore = route.expected_kore(agent, num_ships)
                        routes.append(
                            {
                                "shipyard": shipyard,
                                "num_ships": num_ships,
                                "route": route,
                                "expected_kore": expected_kore,
                            }
                        )

    return routes


def check_protect_fleet_interactions(
    agent, route, target_fleet_id, available_ship_count
):
    interactions = list(find_route_interactions(agent, route))
    if any(x.with_shipyard for x in interactions):
        return

    if any(x.hostile and x.in_contact for x in interactions):
        logger.debug(f"Ignore route {route} with interactions {interactions}")
        return

    # logger.debug(f"{route}, {interactions}")

    num_ships_to_out = {}
    for ship_count in range(1, available_ship_count + 1):
        out_ship_count = _simulate_protect_fleet_attack(interactions, ship_count)
        if out_ship_count is None:
            continue
        num_ships_to_out[ship_count] = out_ship_count

    return num_ships_to_out


def _simulate_protect_fleet_attack(interactions, ship_count: int):
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
