from .board import Player, Launch, BoardRoute
from .helpers import action_check, get_interaction_list_for_route
from .logger import logger


@action_check()
def capture_shipyards(agent: Player, max_attack_distance=10):
    board = agent.board

    max_attack_distance = min(board.steps_left, max_attack_distance)

    capture_future_shipyards(agent, max_attack_distance)
    capture_existing_shipyards(agent, max_attack_distance)


def capture_future_shipyards(agent: Player, max_attack_distance=10):
    board = agent.board

    opponent = agent.opponent
    if not opponent:
        return

    agent_shipyards = list(agent.shipyards)
    op_shipyards = list(opponent.shipyards)

    for target in opponent.fleets:
        final_state = target.final_state
        if (
            not final_state
            or not final_state.operations
            or not final_state.operations.convert
        ):
            continue

        target_time = target.eta + 1
        target_point = final_state.point
        future_shipyard = board.get_shipyard(target.game_id, target.eta)

        min_distance = target_time
        workers = [
            x
            for x in agent_shipyards
            if min_distance <= x.distance_from(target_point) < min_distance + 3
        ]
        # logger.debug(f"workers {workers} min_distance={min_distance}, "
        #              f"distances = {[x.distance_from(target_point) for x in agent_shipyards]}")

        if workers:
            max_distance = max(x.distance_from(target) for x in workers)
        else:
            max_distance = min_distance + 3

        target_state = board.get_shipyard(target.game_id, max_distance)
        if target_state is None:
            target_state = future_shipyard

        target_spawn_capacity = max_distance - min_distance + 1

        nearby_shipyards = [
            sy for sy in op_shipyards if 0 < sy.distance_from(target) <= max_distance
        ]

        target_power = (
            target_state.ship_count
            + target_spawn_capacity
            + sum(sy.ship_count for sy in nearby_shipyards)
        )

        for sy in agent_shipyards:
            distance = sy.point.distance_from(target_point)
            if distance < target_time:
                logger.info(
                    f"Prepare future shipyard attack {sy.point}->{target.point}"
                )
                sy.set_guard_ship_count(target_power)

        estimated_target_power = max(2, int(target_power * 1.3))

        shipyard_to_routes = {}
        for sy in workers:
            if sy.action:
                continue

            routes = _find_attack_future_shipyard_routes(sy, target_state)
            if not routes:
                continue

            routes = sorted(routes, key=lambda x: (-x["power"], x["expected_kore"]))
            # logger.info(f"{target} {sy} {len(routes)}")
            shipyard_to_routes[sy] = routes[0]

        if not shipyard_to_routes:
            continue

        agent_power = sum(x["power"] for x in shipyard_to_routes.values())

        if agent_power > estimated_target_power:
            for sy, route in shipyard_to_routes.items():
                logger.info(
                    f"Attack future shipyard {sy.point}->{target.point}, powers {route['power']} vs {target_power}, "
                    f"num_ships_to_launch={route['num_ships_to_launch']}, distance={sy.distance_from(target)}"
                )
                sy.action = Launch(route["num_ships_to_launch"], route["route"])


def _find_attack_future_shipyard_routes(shipyard, target):
    start = shipyard.point
    end = target.point
    agent = shipyard.player
    # route_checker = agent.route_checker

    min_distance = start.distance_from(end)

    routes = []
    for p in shipyard.board:
        if start.distance_from(p) + p.distance_from(end) > min_distance:
            continue

        for p2t in start.plan_to(p):
            p2t_route = BoardRoute(start, p2t)

            for t2d in p.plan_to(end):
                t2d_route = BoardRoute(p, t2d)

                route = p2t_route + t2d_route
                if len(route) != min_distance:
                    continue

                min_ships_to_launch = max(
                    route.plan.min_fleet_size(),
                    3,  # route_checker.safety_ship_count(route) + 1,
                )
                if shipyard.available_ship_count < min_ships_to_launch:
                    continue

                interaction_ship_count, power = check_shipyard_attack_interactions(
                    agent,
                    route,
                    shipyard.available_ship_count,
                )
                if (
                    interaction_ship_count is None
                    or interaction_ship_count < min_ships_to_launch
                ):
                    continue

                num_ships_to_launch = interaction_ship_count
                expected_kore = route.expected_kore(
                    shipyard.player, num_ships_to_launch
                )

                routes.append(
                    {
                        "route": route,
                        "num_ships_to_launch": num_ships_to_launch,
                        "expected_kore": expected_kore,
                        "power": power,
                    }
                )

    return routes


def capture_existing_shipyards(agent: Player, max_attack_distance=10):
    board = agent.board

    opponent = agent.opponent
    if not opponent:
        return

    agent_shipyards = list(agent.shipyards)
    op_shipyards = list(opponent.shipyards)
    if not agent_shipyards or not op_shipyards:
        return

    agent_shipyards_positions = [x.point for x in agent_shipyards]
    for target in sorted(
        op_shipyards, key=lambda x: x.point.find_min_distance(agent_shipyards_positions)
    ):

        final_state = target.final_state
        if final_state.player_id != target.player_id:
            continue

        opposite_shipyards = target.opposite_shipyards
        if not opposite_shipyards:
            continue

        min_distance = target.distance_from(opposite_shipyards[0])
        if min_distance > max_attack_distance:
            continue

        workers = [
            x for x in agent_shipyards if x.distance_from(target) < min_distance + 3
        ]
        if not workers:
            continue

        max_distance = max(x.distance_from(target) for x in workers)

        try:
            target_state = target.states[max_distance]
        except IndexError:
            target_state = final_state

        target_spawn_capacity = min(
            int(target_state.player.kore // board.spawn_cost),
            target_state.max_ships_to_spawn * max_distance,
        )

        nearby_shipyards = [
            sy for sy in op_shipyards if 0 < sy.distance_from(target) <= max_distance
        ]

        target_power = (
            target_state.ship_count
            + target_spawn_capacity
            + sum(sy.ship_count for sy in nearby_shipyards)
        )

        estimated_target_power = max(2, int(target_power * 1.3))

        shipyard_to_routes = {}
        for sy in workers:
            if sy.action:
                continue

            # logger.info(f"check {target} {sy} ")

            routes = _find_attack_enemy_shipyard_routes(sy, target)
            if not routes:
                continue

            routes = sorted(routes, key=lambda x: (-x["power"], x["expected_kore"]))
            # logger.info(f"{target} {sy} {len(routes)}")
            shipyard_to_routes[sy] = routes[0]

        if not shipyard_to_routes:
            continue

        agent_power = sum(x["power"] for x in shipyard_to_routes.values())
        if agent_power > estimated_target_power:
            for sy, route in shipyard_to_routes.items():
                logger.info(
                    f"Attack shipyard {sy.point}->{target.point}, powers {route['power']} vs {target_power}, "
                    f"num_ships_to_launch={route['num_ships_to_launch']}, distance={sy.distance_from(target)}"
                )
                sy.action = Launch(route["num_ships_to_launch"], route["route"])


def _find_attack_enemy_shipyard_routes(shipyard, target):
    start = shipyard.point
    end = target.point
    agent = shipyard.player
    # route_checker = agent.route_checker

    min_distance = start.distance_from(end)

    routes = []
    for p in shipyard.board:
        if start.distance_from(p) + p.distance_from(end) > min_distance:
            continue

        for p2t in start.plan_to(p):
            p2t_route = BoardRoute(start, p2t)

            for t2d in p.plan_to(end):
                t2d_route = BoardRoute(p, t2d)

                route = p2t_route + t2d_route

                min_ships_to_launch = max(
                    route.plan.min_fleet_size(),
                    3,  # route_checker.safety_ship_count(route) + 1,
                )
                if shipyard.available_ship_count < min_ships_to_launch:
                    continue

                interaction_ship_count, power = check_shipyard_attack_interactions(
                    agent, route, shipyard.available_ship_count
                )
                if (
                    interaction_ship_count is None
                    or interaction_ship_count < min_ships_to_launch
                ):
                    continue

                num_ships_to_launch = interaction_ship_count
                expected_kore = route.expected_kore(
                    shipyard.player, num_ships_to_launch
                )

                routes.append(
                    {
                        "route": route,
                        "num_ships_to_launch": num_ships_to_launch,
                        "expected_kore": expected_kore,
                        "power": power,
                    }
                )

    return routes


def check_shipyard_attack_interactions(agent, route, available_ship_count):
    interactions = get_interaction_list_for_route(agent, route)
    if any(x.with_shipyard for x in interactions):
        return None, 0

    if any(x.hostile and x.in_contact for x in interactions):
        return None, 0

    ship_count_to_out = {}
    for ship_count in range(1, available_ship_count + 1):
        out_ship_count = _simulate_shipyard_attack(interactions, ship_count)
        if out_ship_count is None:
            continue
        ship_count_to_out[ship_count] = out_ship_count

    if not ship_count_to_out:
        return None, 0

    max_out = max(ship_count_to_out.values())

    return max([x for x, out in ship_count_to_out.items() if out == max_out]), max_out


def _simulate_shipyard_attack(interactions, ship_count):
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
