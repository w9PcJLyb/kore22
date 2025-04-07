from .board import Board
from .logger import logger, init_logger
from .offense import capture_shipyards
from .defense import defend_shipyards, set_shipyard_defense, protect_fleets
from .expansion import expand
from .mining import mine, greedy_mine
from .control import (
    spawn,
    greedy_spawn,
)
from .direct_attack import attack_enemy_fleets
from .adjacent_attack import adjacent_attack
from .helpers import get_encrypted_actions, need_to_save_time


def run_simulation(board, time_to_actions=None):
    time_to_actions = time_to_actions or {}

    boards = [board]
    time = 0
    while board.steps_left > 0 and board.id_to_fleet and time < 40:
        actions = time_to_actions.get(time)
        board = board.next(actions)
        boards.append(board)
        time += 1

    for i in range(len(boards)):
        boards[i].simulations = boards[i:]


def create_board(obs, conf, time_to_actions=None):
    board = Board.from_raw(obs, conf)
    # board.steps_left = 200

    run_simulation(board, time_to_actions)

    my_id = obs["player"]

    my_agent = board.get_player(my_id)
    saving_mode = need_to_save_time(
        steps_left=board.steps_left, remaining_time=obs["remainingOverageTime"]
    )
    if saving_mode:
        my_agent.__setattr__("saving_mode", saving_mode)

    return board, my_agent


def rerun_board(obs, conf, my_agent):
    # logger.debug("Rerun Board")
    actions = {}
    guard_ship_count = {}
    for sy in my_agent.shipyards:
        if sy.action:
            actions[sy.game_id] = sy.action
        if sy.guard_ship_count:
            guard_ship_count[sy.game_id] = sy.guard_ship_count

    new_board, new_agent = create_board(obs, conf, {0: actions})

    new_agent.kore = my_agent.kore
    for sy in new_agent.shipyards:
        if sy.game_id in actions:
            sy.action = actions[sy.game_id]
        if sy.game_id in guard_ship_count:
            sy.guard_ship_count = guard_ship_count[sy.game_id]

    set_shipyard_defense(new_agent)

    return new_board, new_agent


class AgentConfig:
    mining_distance = 25
    adjacent_attack_distance = 20
    capture_shipyards_distance = 20
    attack_enemy_fleets_distance = 20


def get_agent_config(a):
    conf = AgentConfig()

    shipyard_count = a.shipyard_count

    if shipyard_count > 3:
        conf.adjacent_attack_distance = 14
        conf.capture_shipyards_distance = 10
        conf.attack_enemy_fleets_distance = 20

    if shipyard_count <= 3:
        conf.mining_distance = 25
    elif shipyard_count < 10:
        conf.mining_distance = 21
    elif shipyard_count < 20:
        conf.mining_distance = 11
    else:
        conf.mining_distance = 7

    return conf


def agent(obs, conf):
    step = obs["step"]
    if step == 0:
        init_logger(logger)

    board, a = create_board(obs, conf)
    set_shipyard_defense(a)
    if not a.opponent:
        return {}

    remaining_time = obs["remainingOverageTime"]
    logger.info(
        f"<step_{step + 1}>, remaining {remaining_time:.1f}s, "
        f"powers: {a.ship_count}({a.shipyard_count})"
        f" vs {a.opponent.ship_count}({a.opponent.shipyard_count})"
    )
    if getattr(a, "saving_mode", False):
        logger.warning(
            f"Saving mode enabled: steps_left={board.steps_left}, remaining_time={remaining_time}"
        )

    # defend_shipyards(a)
    # capture_shipyards(a, shipyard_to_value)
    # expand(a, point_to_value)
    # adjacent_attack(a)
    # attack_enemy_fleets(a)
    # greedy_spawn(a, shipyard_to_value)
    # mine(a, shipyard_to_value)
    # spawn(a, shipyard_to_value)

    # logger.info(board.simulations)
    # for time, data in a.expected_positions.items():
    #     logger.info(time)
    #     logger.info(data)

    protect_fleets(a)
    agent_config = get_agent_config(a)

    if adjacent_attack(
        a, anti_siege=True, max_distance=agent_config.adjacent_attack_distance
    ):
        board, a = rerun_board(obs, conf, a)

    if defend_shipyards(a):
        board, a = rerun_board(obs, conf, a)

    if capture_shipyards(
        a, max_attack_distance=agent_config.capture_shipyards_distance
    ):
        board, a = rerun_board(obs, conf, a)

    if expand(a):
        board, a = rerun_board(obs, conf, a)

    if adjacent_attack(a, max_distance=agent_config.adjacent_attack_distance):
        board, a = rerun_board(obs, conf, a)

    if attack_enemy_fleets(a):
        board, a = rerun_board(obs, conf, a)

    if greedy_mine(a, max_distance=agent_config.mining_distance):
        board, a = rerun_board(obs, conf, a)

    greedy_spawn(a)

    if mine(a, max_distance=agent_config.mining_distance):
        pass
        # board, a = rerun_board(obs, conf, a)

    spawn(a)

    # actions = a.actions()
    actions = get_encrypted_actions(a)

    return actions
