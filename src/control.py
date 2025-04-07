import random

from .board import Player, Spawn
from .helpers import is_invitable_victory


def _need_more_ships(agent: Player, ship_count: int):
    board = agent.board
    if board.steps_left < 10:
        return False
    if ship_count > _max_ships_to_control(agent):
        return False
    if board.steps_left < 50 and is_invitable_victory(agent):
        return False
    return True


def _max_ships_to_control(agent: Player):
    return max(100, 3 * sum(x.ship_count for x in agent.opponents))


def greedy_spawn(agent: Player):
    board = agent.board

    if not _need_more_ships(agent, agent.ship_count):
        return

    max_ship_count = _max_ships_to_control(agent)
    shipyards = list(agent.shipyards)
    shipyard_count = len(shipyards)
    ship_count = agent.ship_count

    random.shuffle(shipyards)
    # shipyards = sorted(shipyards, key=lambda x: -x.max_ships_to_spawn)

    for shipyard in shipyards:
        if shipyard.action:
            continue

        if shipyard.ship_count > 0.5 * ship_count / shipyard_count:
            continue

        num_ships_to_spawn = shipyard.max_ships_to_spawn

        if int(agent.available_kore() // board.spawn_cost) >= num_ships_to_spawn:
            shipyard.action = Spawn(num_ships_to_spawn)

        ship_count += num_ships_to_spawn
        if ship_count > max_ship_count:
            return


def spawn(agent: Player):
    board = agent.board

    if not _need_more_ships(agent, agent.ship_count):
        return

    ship_count = agent.ship_count
    max_ship_count = _max_ships_to_control(agent)

    shipyards = list(agent.shipyards)
    random.shuffle(shipyards)
    # shipyards = sorted(shipyards, key=lambda x: x.max_ships_to_spawn)
    for shipyard in shipyards:
        if shipyard.action:
            continue
        num_ships_to_spawn = min(
            int(agent.available_kore() // board.spawn_cost),
            shipyard.max_ships_to_spawn,
        )
        if num_ships_to_spawn:
            shipyard.action = Spawn(num_ships_to_spawn)
            ship_count += num_ships_to_spawn
            if ship_count > max_ship_count:
                return
