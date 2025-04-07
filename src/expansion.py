import math
import random
from collections import defaultdict
import numpy as np

from .basic import min_ship_count_for_flight_plan_len
from .geometry import Point, Convert, PlanRoute, PlanPath
from .board import Player, BoardRoute, Launch, Shipyard
from .logger import logger
from .helpers import (
    get_interaction_list_for_route,
    find_points_interactions,
    action_check,
    estimate_point_values,
)


@action_check()
def expand(player: Player):
    board = player.board
    num_shipyards_to_create = need_more_shipyards(player)
    if not num_shipyards_to_create:
        return

    shipyard_to_points = find_best_position_for_shipyards(player)

    shipyard_count = 0
    for shipyard, score_points in shipyard_to_points.items():
        if shipyard_count >= num_shipyards_to_create:
            break

        if shipyard.action:
            continue

        for score, target in sorted(score_points, key=lambda x: -x[0])[:1]:

            routes = find_routes_to_expand(shipyard, target)

            if not routes:
                continue

            max_power = max(x["power"] for x in routes)
            route = random.choice([x for x in routes if x["power"] == max_power])
            num_ships_to_launch = route["num_ships_to_launch"]

            convert_route = BoardRoute(
                shipyard.point, route["route"].plan + PlanRoute([PlanPath(Convert)])
            )
            if num_ships_to_launch < convert_route.plan.min_fleet_size():
                continue

            convert_route_with_reverse = BoardRoute(
                shipyard.point,
                route["route"].plan
                + PlanRoute([PlanPath(Convert)])
                + route["route"].plan.reverse(),
            )
            if num_ships_to_launch >= convert_route_with_reverse.plan.min_fleet_size():
                convert_route = convert_route_with_reverse

            if shipyard.available_ship_count < num_ships_to_launch:
                continue

            shipyard.action = Launch(num_ships_to_launch, convert_route)
            logger.info(
                f"Expand {shipyard.point}->{target}, "
                f"num_ships_to_launch={num_ships_to_launch}, distance={shipyard.distance_from(target)}"
            )
            shipyard_count += 1
            break

    if shipyard_count < num_shipyards_to_create and shipyard_to_points:
        shipyards = sorted(
            shipyard_to_points,
            key=lambda sh: -max(x[0] for x in shipyard_to_points[sh]),
        )
        for sy in shipyards:
            if sy.action:
                continue
            logger.info(f"Shipyard {sy.point}: preparing a fleet to expand")
            sy.set_guard_ship_count(sy.ship_count)
            break


def find_routes_to_expand(shipyard: Shipyard, target: Point):
    target_distance = shipyard.distance_from(target)

    board = shipyard.board
    agent = shipyard.player
    route_checker = agent.route_checker

    # d = target.to_tuple() == (10, 7) and board.step == 0

    op_shipyards = list(agent.opponent.shipyards) if agent.opponent else []
    op_power = 0
    for op_sy in op_shipyards:
        op_distance = op_sy.point.distance_from(target)
        if op_distance < target_distance * 2:
            launch_time = 1 + max(0, target_distance * 2 - op_distance)
            # if d:
            #     print(op_sy, launch_time)
            future_op_sy = board.get_shipyard(op_sy.game_id, launch_time)
            if future_op_sy:
                op_power += future_op_sy.ship_count
            else:
                op_power += op_sy.ship_count

    if shipyard.available_ship_count - board.shipyard_cost < op_power:
        return []

    routes = []
    for p in board:

        distance = shipyard.distance_from(p) + p.distance_from(target)
        if distance > target_distance:
            continue

        for s2p in shipyard.point.plan_to(p):
            s2p_route = BoardRoute(shipyard.point, s2p)

            for p2t in p.plan_to(target):
                p2t_route = BoardRoute(p, p2t)

                route = s2p_route + p2t_route

                min_ships_to_launch = max(
                    board.shipyard_cost + route_checker.safety_ship_count(route) + 1,
                    min_ship_count_for_flight_plan_len(route.plan.command_length() + 1),
                )

                ship_count, power = check_expand_interactions(
                    agent, route, shipyard.available_ship_count
                )

                if (
                    ship_count is None
                    or power < board.shipyard_cost
                    or ship_count < min_ships_to_launch
                ):
                    continue

                expected_kore = route.expected_kore(agent, ship_count)
                routes.append(
                    {
                        "route": route,
                        "num_ships_to_launch": ship_count,
                        "score": expected_kore / math.sqrt(ship_count),
                        "power": power,
                    }
                )

    return routes


def check_expand_interactions(agent, route, available_ship_count):
    sy_point = route.end
    sy_eta = len(route.points) + 1
    sy_interactions = find_points_interactions(
        agent,
        [sy_point for _ in agent.board.simulations],
        adjacent_interation=False,
        shipyard_interation=False,
    )
    for interaction in sy_interactions:
        if (
            interaction.time >= sy_eta
            and interaction.hostile
            and interaction.is_direct_interaction
        ):
            return None, 0

    interactions = get_interaction_list_for_route(agent, route)
    if any(x.with_shipyard for x in interactions):
        return None, 0

    # logger.debug(f"{route}, {interactions}")

    ship_count_to_out = {}
    for ship_count in range(1, available_ship_count + 1):
        out_ship_count = _simulate_expand(interactions, ship_count)
        if out_ship_count is None:
            continue
        ship_count_to_out[ship_count] = out_ship_count

    if not ship_count_to_out:
        return None, 0

    max_out = max(ship_count_to_out.values())

    return max([x for x, out in ship_count_to_out.items() if out == max_out]), max_out


def _simulate_expand(interactions, ship_count: int):
    for interaction in interactions:
        if not interaction.hostile:
            if interaction.ship_count >= ship_count:
                return
            else:
                ship_count += interaction.ship_count
        else:
            if interaction.in_contact:
                return

            if interaction.ship_count >= ship_count:
                return
            else:
                ship_count -= interaction.ship_count

                if ship_count <= 0:
                    return

    return ship_count


def find_best_position_for_shipyards(player: Player):
    board = player.board
    point_to_value, _ = estimate_point_values(player)

    board_size = board.field.size
    shipyards = list(board.shipyards)
    for f in board.fleets:
        state = f.final_state
        if state and state.operations and state.operations.convert:
            shipyards.append(state)

    agent_shipyards = [x for x in shipyards if x.player_id == player.game_id]

    shipyard_to_scores = defaultdict(list)
    for p, data in board.point_to_data.items():
        if data.kore > 50:
            continue

        closed_shipyard = None
        min_distance = board_size
        for shipyard in shipyards:
            distance = shipyard.point.distance_from(p)
            if shipyard.player_id != player.game_id:
                distance -= 1

            if distance < min_distance:
                closed_shipyard = shipyard
                min_distance = distance

        if (
            not closed_shipyard
            or not isinstance(closed_shipyard, Shipyard)
            or closed_shipyard.player_id != player.game_id
            or min_distance < 4
            or min_distance > 6
        ):
            continue

        if (
            closed_shipyard.opposite_shipyards
            and closed_shipyard.distance_from(closed_shipyard.opposite_shipyards[0])
            <= 5
        ):
            continue

        # if len(agent_shipyards) > 1:
        #     distances = sorted([p.distance_from(sy.point) for sy in agent_shipyards])[
        #         :2
        #     ]
        #     if distances[-1] > 7:
        #         continue

        shipyard_to_scores[closed_shipyard].append((point_to_value[p], p))

    # shipyard_to_point = {}
    # for shipyard, scores in shipyard_to_scores.items():
    #     if (
    #         shipyard.opposite_shipyards
    #         and shipyard.distance_from(shipyard.opposite_shipyards[0]) <= 5
    #     ):
    #         continue
    #
    #     if scores:
    #         scores = sorted(scores, key=lambda x: x["score"])
    #         point = scores[-1]["point"]
    #         score = scores[-1]["score"]
    #         shipyard_to_point[shipyard] = point, score

    return shipyard_to_scores


def need_more_shipyards(player: Player) -> int:
    board = player.board

    if player.ship_count < 100:
        return 0

    fleet_distance = []
    for sy in player.shipyards:
        for f in sy.incoming_allied_fleets:
            f = board.get_fleet(f.game_id)
            if f:
                fleet_distance.append(f.eta)

    if not fleet_distance:
        return 0

    mean_fleet_distance = sum(fleet_distance) / len(fleet_distance)

    shipyard_production_capacity = sum(x.max_ships_to_spawn for x in player.shipyards)

    steps_left = board.steps_left
    if steps_left > 100:
        scale = 2
    elif steps_left > 50:
        scale = 3
    elif steps_left > 10:
        scale = 500
    else:
        scale = np.inf

    needed = player.kore > scale * shipyard_production_capacity * mean_fleet_distance
    if not needed:
        return 0

    current_shipyard_count = player.shipyard_count

    op_shipyard_positions = {
        x.point for x in board.shipyards if x.player_id != player.game_id
    }
    expected_shipyard_count = current_shipyard_count + sum(
        1
        for x in player.fleets
        if x.final_state.operations.convert or x.destination in op_shipyard_positions
    )

    opponent = player.opponent
    if opponent:
        opponent_shipyard_count = opponent.shipyard_count
        opponent_ship_count = opponent.ship_count
    else:
        opponent_shipyard_count = 0
        opponent_ship_count = 0

    if (
        expected_shipyard_count > opponent_shipyard_count
        and player.ship_count < opponent_ship_count
    ):
        return 0

    if (
        expected_shipyard_count >= 3 * opponent_shipyard_count
        and player.ship_count <= 3 * opponent_ship_count
    ):
        return 0

    if current_shipyard_count < 10:
        if expected_shipyard_count > current_shipyard_count:
            return 0
        else:
            return 1

    return max(0, 5 - (expected_shipyard_count - current_shipyard_count))
