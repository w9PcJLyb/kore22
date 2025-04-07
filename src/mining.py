import math
import numpy as np
from typing import List, Tuple
from collections import defaultdict

from .geometry import Point, PlanRoute, PlanPath
from .board import Player, BoardRoute, Launch, Shipyard, Fleet
from .logger import logger
from .helpers import find_route_interactions, action_check


@action_check()
def mine(agent: Player, max_distance=10):
    board = agent.board
    opponent = agent.opponent
    if not opponent:
        return

    safety = False
    my_ship_count = agent.ship_count
    op_ship_count = opponent.ship_count
    if my_ship_count < 2 * op_ship_count:
        safety = True

    if getattr(agent, "saving_mode", False):
        max_distance = 5

    max_distance = min(int(board.steps_left // 2), max_distance)
    route_checker = agent.route_checker

    sy_to_routes = {}
    for sy in agent.shipyards:
        # logger.info(sy.__dict__)

        if sy.action or sy.force_self_route:
            continue

        free_ships = sy.available_ship_count

        if free_ships <= 2:
            continue

        fleet_eta = 0
        fleet_ship_count = 0
        incoming_allied_fleets = sy.incoming_allied_fleets
        if incoming_allied_fleets:
            fleets = []
            for f in incoming_allied_fleets:
                f = board.get_fleet(f.game_id)
                if f:
                    fleets.append(f)
            if fleets:
                closest_fleet = sorted(fleets, key=lambda x: x.eta)[0]
                fleet_eta = closest_fleet.eta
                fleet_ship_count = closest_fleet.final_state.ship_count

        # logger.debug(
        #     f"Mining: shipyard={sy.point}, fleet_eta={fleet_eta}, fleet_ship_count={fleet_ship_count}"
        # )

        routes = []
        for route, destination in find_shipyard_mining_routes(
            sy, safety=safety, max_distance=max_distance
        ):
            num_ships_to_launch = max(
                route.plan.min_fleet_size(), route_checker.safety_ship_count(route) + 1
            )

            if num_ships_to_launch > free_ships + fleet_ship_count:
                continue
            if num_ships_to_launch > free_ships:
                eta = fleet_eta
            else:
                eta = 0

            expected_kore = route.expected_kore(agent, num_ships_to_launch)
            if expected_kore < 20:
                continue

            # t = (len(route) + eta) ** -1
            # if shipyard_count == 1:
            #     t = t ** 0.5
            # elif shipyard_count == 2:
            #     t = t ** 0.6
            # else:
            #     t = t ** 1

            score = expected_kore / (len(route) + eta) / math.sqrt(num_ships_to_launch)

            if isinstance(destination, Shipyard) and destination.opposite_shipyards:
                score *= 1.2

            routes.append(
                {
                    "score": score,
                    "route": route,
                    "ship_count": num_ships_to_launch,
                    "expected_kore": expected_kore,
                    "fleet_eta": fleet_eta,
                    "destination": destination,
                }
            )

        if routes:
            sy_to_routes[sy] = routes

    if not sy_to_routes:
        return

    expected_kore_stat = []
    for routes in sy_to_routes.values():
        for r in routes:
            expected_kore_stat.append(r["expected_kore"] / r["ship_count"])

    if board.steps_left > 20:
        expected_kore_min = np.quantile(expected_kore_stat, q=0.6)
    else:
        expected_kore_min = 0

    for sy, routes in sy_to_routes.items():
        accepted_routes = [
            x
            for x in routes
            if x["expected_kore"] / x["ship_count"] > expected_kore_min
        ]
        # if board.step == 0:
        #     print(sy, len(accepted_routes))
        if not accepted_routes:
            logger.debug(f"{sy} has only low quality routes")
            _move_fleet(sy, routes)
            continue
        else:
            routes = accepted_routes

        free_ships = sy.available_ship_count

        routes = sorted(routes, key=lambda x: -x["score"])
        is_short = False
        if routes[0]["ship_count"] > free_ships:
            is_short = True
            routes = [x for x in routes if len(x["route"].points) <= x["fleet_eta"]]

        for r in routes[:5]:
            num_ships_to_launch = r["ship_count"]
            route = r["route"]
            if num_ships_to_launch > free_ships:
                break

            kw = dict(
                num_ships=num_ships_to_launch,
                time=len(route),
                expected_kore=round(r["expected_kore"]),
                complexity=len(route.paths),
            )
            if is_short:
                kw["is_short"] = is_short
            if isinstance(r["destination"], Fleet):
                kw["to_future_shipyard"] = True

            logger.debug(f"Mining: {sy.point}: {kw}")
            sy.action = Launch(num_ships_to_launch, route)
            break


@action_check()
def greedy_mine(agent: Player, max_distance=10):
    opponent = agent.opponent
    if not opponent:
        return

    if getattr(agent, "saving_mode", False):
        max_distance = 5

    route_checker = agent.route_checker

    for sy in agent.shipyards:
        # logger.info(sy.__dict__)

        if sy.action or not sy.force_self_route:
            continue

        destinations = [(sy, 0)]
        sy_max_distance = min(max_distance, sy.window // 2 - 1)

        routes = []
        for route, destination in find_shipyard_mining_routes(
            sy, safety=True, max_distance=sy_max_distance, destinations=destinations
        ):

            num_ships_to_launch = max(
                route.plan.min_fleet_size(), route_checker.safety_ship_count(route) + 1
            )
            if num_ships_to_launch > sy.ship_count:
                continue

            expected_kore = route.expected_kore(agent, num_ships_to_launch)
            if expected_kore < 10:
                continue

            score = expected_kore / len(route) / math.sqrt(num_ships_to_launch)

            routes.append(
                {
                    "score": score,
                    "route": route,
                    "ship_count": num_ships_to_launch,
                    "expected_kore": expected_kore,
                }
            )

        if not routes:
            continue

        routes = sorted(routes, key=lambda x: -x["score"])

        r = routes[0]

        num_ships_to_launch = r["ship_count"]
        route = r["route"]

        kw = dict(
            num_ships=num_ships_to_launch,
            time=len(route),
            expected_kore=round(r["expected_kore"]),
            complexity=len(route.paths),
        )

        logger.debug(f"Self Route Mining: {sy.point}: {kw}")
        sy.action = Launch(num_ships_to_launch, route)


def _move_fleet(sy: Shipyard, routes):
    if not routes:
        return

    destination_to_routes = defaultdict(list)
    for r in routes:
        destination_to_routes[r["destination"]].append(r)

    opponent = sy.player.opponent
    if not opponent:
        return
    op_shipyards = [x.point for x in opponent.shipyards]
    if not op_shipyards:
        return

    destination_to_op_distance = {}
    for d in destination_to_routes:
        destination_to_op_distance[d] = d.point.find_min_distance(op_shipyards)

    for d in sorted(destination_to_routes, key=lambda x: destination_to_op_distance[x]):
        if d == sy:
            continue
        routes = destination_to_routes[d]
        routes = sorted(routes, key=lambda x: len(x["route"].points))
        for r in routes:
            min_ships_to_launch = r["ship_count"]
            if sy.available_ship_count >= min_ships_to_launch:
                logger.debug(
                    f"Move: {sy.point} -> {d.point}, num_ships={sy.available_ship_count}, "
                    f"distance={len(r['route'].points)}"
                )
                sy.action = Launch(sy.available_ship_count, r["route"])
                return


def _should_to_mine_here(agent: Player, route: BoardRoute):
    board = agent.board

    if agent.kore / (agent.shipyard_count + 1) < 50:
        return True

    if board.steps_left < 50:
        return True

    kore_values = {p: board.kore_at_point(p) for p in route.points}
    kore_points = agent.kore_area
    if any(10 < kore < 400 and p in kore_points for p, kore in kore_values.items()):
        return False

    return True


def find_shipyard_mining_routes(
    sy: Shipyard, safety=True, max_distance: int = 15, destinations=None
) -> List[Tuple[BoardRoute, Shipyard]]:
    if max_distance < 1:
        return []

    departure = sy.point
    player = sy.player

    if destinations is None:
        destinations = set()
        for shipyard in player.shipyards:
            if shipyard.final_state.player_id == player.game_id:
                destinations.add((shipyard, 0))
        for fleet in player.fleets:
            final_state = fleet.final_state
            if (
                final_state
                and final_state.operations
                and final_state.operations.convert
            ):
                destinations.add((final_state, fleet.eta))

    if not destinations:
        return []

    end_points = {p for p, _ in destinations}
    point_to_kore = {p: d.kore for p, d in sy.board.point_to_data.items() if d.kore > 0}
    desired_points = sorted(point_to_kore, key=lambda p: -point_to_kore[p])[:4]

    routes = []
    bad_plans = []

    for c in sy.point.nearby_points(max_distance):
        if c == departure:
            continue

        d2p_routes = []
        for d2p in departure.plan_to(c):
            if any(d2p.startswith(x) for x in bad_plans):
                continue

            route = BoardRoute(departure, d2p)
            if not check_mining_interactions(
                agent=player,
                route=route,
                safety=safety,
                mh_time=len(route.points),
                destinations=end_points,
            ):
                bad_plans.append(d2p)
                continue

            d2p_routes.append(route)

            complex_plan = complicate_route(route, desired_points, sy.board.step)
            if complex_plan:
                complex_route = BoardRoute(departure, complex_plan)
                if not check_mining_interactions(
                    agent=player,
                    route=complex_route,
                    safety=safety,
                    mh_time=len(complex_route.points),
                    destinations=end_points,
                ):
                    continue
                d2p_routes.append(complex_route)
                # if sy.board.step == 0:
                #     print(d2p.paths, "===>", complex_plan.paths)
                #     print(route, "===>", complex_route, complex_route.points)

        if not d2p_routes:
            continue

        for destination, delay in sorted(
            destinations, key=lambda x: x[0].point.distance_from(c)
        )[:3]:
            if destination.point.distance_from(c) > max_distance:
                continue

            p2d_plans = c.plan_to(destination.point)

            for d2p_route in d2p_routes:
                mh_time = d2p_route.plan.num_steps
                for p2d in p2d_plans:
                    p2d_route = BoardRoute(c, p2d)

                    route = d2p_route + p2d_route

                    if route.plan.num_steps < delay:
                        continue

                    if not _should_to_mine_here(player, route):
                        continue

                    if not check_mining_interactions(
                        agent=player,
                        route=route,
                        safety=safety,
                        mh_time=mh_time,
                        destinations=end_points,
                    ):
                        continue

                    routes.append((route, destination))

    return routes


def check_mining_interactions(agent, route, safety, mh_time, destinations):
    interactions = find_route_interactions(agent, route)
    return _simulate_mining(interactions, safety, mh_time, route, destinations)


def _simulate_mining(interactions, safety, mh_time, route, destinations):
    for interaction in interactions:
        if interaction.with_shipyard:
            return False

        if interaction.hostile:

            if interaction.in_contact:
                logger.debug(f"Ignore route {route} with interaction {interaction}")
                return False

            if interaction.is_direct_interaction:
                if interaction.ship_count > 2:
                    return False
            else:
                if safety and interaction.ship_count > 2:
                    return False
        else:
            if interaction.ship_count > 2:
                destination = interaction.object.destination

                if not destination:
                    return False

                if destination.point not in destinations:
                    # logger.debug(f"{interaction.object.destination}, {destinations}")
                    return False

                if interaction.time <= mh_time:
                    return False

    return True


def complicate_route(route: BoardRoute, desired_points: List[Point], step):
    route_desired_times = [
        t for t, p in enumerate(route.points, 1) if p in desired_points
    ]
    if not route_desired_times:
        return

    cluster_time = [route_desired_times[0]]
    for t in route_desired_times[1:]:
        if t == cluster_time[-1] + 1:
            cluster_time.append(t)

    new_plan = []
    time = 0
    for path in route.plan.paths:
        if cluster_time[-1] > time + path.num_steps:
            new_plan.append(path)
            time += path.num_steps

        elif time < cluster_time[-1] <= time + path.num_steps:
            new_plan += [
                PlanPath(path.direction, cluster_time[-1] - time),
                PlanPath(path.direction, -len(cluster_time)),
                PlanPath(
                    path.direction,
                    len(cluster_time) + path.num_steps - (cluster_time[-1] - time),
                ),
            ]
            # print(
            #     path,
            #     PlanPath(path.direction, cluster_time[-1] - time),
            #     PlanPath(path.direction, -len(cluster_time)),
            #     PlanPath(
            #         path.direction,
            #         len(cluster_time) + path.num_steps - (cluster_time[-1] - time),
            #     ),
            # )
            time += path.num_steps + len(cluster_time) * 2

        else:
            new_plan.append(path)
            time += path.num_steps

    new_plan = PlanRoute(new_plan)

    return new_plan
