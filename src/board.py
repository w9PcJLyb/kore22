import itertools
import math

import numpy as np
from typing import Set, Any, Dict, List, Union, Optional, Generator, Tuple
from collections import defaultdict
from dataclasses import field as dataclasses_field, dataclass
from kaggle_environments.envs.kore_fleets.helpers import Configuration

from .basic import (
    Obj,
    cached_call,
    next_step_kore,
    collection_rate_for_ship_count,
    max_ships_to_spawn,
    cached_property,
    amount_of_collected_kore,
    create_spawn_ships_command,
    create_launch_fleet_command,
    max_flight_plan_len_for_ship_count,
)
from .geometry import (
    Field,
    Point,
    North,
    South,
    Action,
    Convert,
    PlanPath,
    PlanRoute,
    GAME_ID_TO_ACTION,
    COMMAND_TO_ACTION,
)
from .logger import logger


class _ShipyardAction:
    def to_str(self):
        raise NotImplementedError

    def __repr__(self):
        return self.to_str()


class Spawn(_ShipyardAction):
    def __init__(self, ship_count: int):
        self.ship_count = ship_count

    def to_str(self):
        return create_spawn_ships_command(self.ship_count)


class Launch(_ShipyardAction):
    def __init__(self, ship_count: int, route: "BoardRoute"):
        self.ship_count = ship_count
        self.route = route
        if route.plan.min_fleet_size() > ship_count:
            logger.warning(f"Flight plan will be truncated: {route}")

    def to_str(self):
        return create_launch_fleet_command(self.ship_count, self.route.plan.to_str())

    def is_convert_action(self):
        plan = self.route.plan
        if plan.paths and plan.paths[-1].direction == Convert:
            return True
        return False


class DoNothing(_ShipyardAction):
    def __repr__(self):
        return "Do nothing"

    def to_str(self):
        raise NotImplementedError


class BoardPath:
    max_length = 32

    def __init__(self, start: "Point", plan: PlanPath):
        assert plan.num_steps > 0 or plan.direction == Convert

        self._plan = plan

        field = start.field
        x, y = start.x, start.y
        if np.isfinite(plan.num_steps):
            n = plan.num_steps + 1
        else:
            n = self.max_length
        action = plan.direction

        if plan.direction == Convert:
            self._track = []
            self._start = start
            self._end = start
            self._build_shipyard = True
            return

        if action in (North, South):
            track = field.get_column(x, start=y, size=n * action.dy)
        else:
            track = field.get_row(y, start=x, size=n * action.dx)

        self._track = track[1:]
        self._start = start
        self._end = track[-1]
        self._build_shipyard = False

    def __repr__(self):
        start, end = self.start, self.end
        return f"({start.x}, {start.y}) -> ({end.x}, {end.y})"

    def __len__(self):
        return len(self._track)

    @property
    def plan(self):
        return self._plan

    @property
    def points(self):
        return self._track

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end


class BoardRoute:
    def __init__(self, start: "Point", plan: Union["PlanRoute", List[BoardPath]]):
        if isinstance(plan, PlanRoute):
            paths = []
            for p in plan.paths:
                path = BoardPath(start, p)
                start = path.end
                paths.append(path)
        else:
            paths = plan
            plan = PlanRoute([x.plan for x in paths])
            assert start == paths[0].start
            for p1, p2 in zip(paths[:-1], paths[1:]):
                assert p1.end == p2.start

        self._plan = plan
        self._paths = paths
        self._start = paths[0].start
        self._end = paths[-1].end

    def __repr__(self):
        points = []
        for p in self._paths:
            points.append(p.start)
        points.append(self.end)
        return " -> ".join([f"({p.x}, {p.y})" for p in points])

    def __iter__(self) -> Generator["Point", None, None]:
        for p in self._paths:
            yield from p.points

    def __len__(self):
        return sum(len(x) for x in self._paths)

    def __add__(self, other: "BoardRoute"):
        return BoardRoute(self.start, plan=self.paths + other.paths)

    @cached_property
    def points(self) -> List["Point"]:
        points = []
        for p in self._paths:
            points += p.points
        return points

    @property
    def plan(self) -> PlanRoute:
        return self._plan

    def command(self) -> str:
        return self.plan.to_str()

    @property
    def paths(self) -> List[BoardPath]:
        return self._paths

    @property
    def start(self) -> "Point":
        return self._start

    @property
    def end(self) -> "Point":
        return self._end

    def command_length(self) -> int:
        return len(self.command())

    def last_action(self):
        return self.paths[-1].plan.direction

    def expected_kore(self, agent: "Player", ship_count: int):
        rate = collection_rate_for_ship_count(ship_count)
        if rate <= 0:
            return 0

        board = agent.board
        agent_id = agent.game_id

        route_points = self.points
        point_to_kore = {p: board.kore_at_point(p) for p in route_points}

        regen_rate = board.regen_rate
        max_cell_kore = board.max_cell_kore

        expected_kore = 0
        for time, point in enumerate(route_points, 1):
            collected = amount_of_collected_kore(point_to_kore[point], rate)
            expected_kore += collected
            point_to_kore[point] -= collected

            try:
                next_board = board.simulations[time]
            except IndexError:
                next_board = None

            fleet_points = set()
            if next_board:
                for p, kore in point_to_kore.items():
                    pd = next_board.point_data(p)

                    for fleet_id, kore_amount in pd.mined.items():
                        fleet = next_board.id_to_fleet[fleet_id]
                        collected = amount_of_collected_kore(
                            kore, fleet.collection_rate
                        )

                        point_to_kore[p] -= collected
                        fleet_points.add(p)

                        delta = kore_amount - collected
                        if delta > 0:
                            if fleet.player_id == agent_id:
                                expected_kore -= delta
                            else:
                                expected_kore += delta

                    for fleet_id, kore_amount in pd.dropped.items():
                        point_to_kore[p] += kore_amount

            for p, kore in point_to_kore.items():
                if p not in fleet_points:
                    point_to_kore[p] = next_step_kore(
                        point_to_kore[p], regen_rate, max_cell_kore
                    )

        return expected_kore


class PositionObj(Obj):
    def __init__(self, *args, point: Point, player_id: int, board: "Board", **kwargs):
        super().__init__(*args, **kwargs)
        self._point = point
        self.player_id = player_id
        self._board = board

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._game_id}, position={self.point}, player={self.player_id})"

    def dirs_to(self, obj: Union["PositionObj", Point]):
        if isinstance(obj, Point):
            return self._point.dirs_to(obj)
        return self._point.dirs_to(obj.point)

    def distance_from(self, obj: Union["PositionObj", Point]) -> int:
        if isinstance(obj, Point):
            return self._point.distance_from(obj)
        return self._point.distance_from(obj.point)

    @property
    def board(self) -> "Board":
        return self._board

    @property
    def point(self) -> Point:
        return self._point

    @property
    def player(self) -> "Player":
        return self.board.get_player(self.player_id)

    def find_closest_object(
        self, objects: List[Union[Point, "PositionObj"]]
    ) -> Optional[Union[Point, "PositionObj"]]:
        if not objects:
            return

        if len(objects) == 1:
            return objects[0]

        closest_object = objects[0]
        min_distance = self.distance_from(closest_object)
        for x in objects[1:]:
            distance = self.distance_from(x.point)
            if distance < min_distance:
                closest_object = x
                min_distance = distance

        return closest_object


class Shipyard(PositionObj):
    def __init__(self, *args, ship_count: int, turns_controlled: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.ship_count = ship_count
        self.turns_controlled = turns_controlled
        self.guard_ship_count = 0
        self.action: Optional[_ShipyardAction] = None
        self.tasks = []
        self.balancer = None
        self.force_self_route = False
        self.window = math.inf
        self.greedy_mining = False

    @property
    def max_ships_to_spawn(self) -> int:
        return max_ships_to_spawn(self.turns_controlled)

    @property
    def available_ship_count(self):
        return self.ship_count - self.guard_ship_count

    @cached_property
    def states(self):
        return self.board.get_shipyard(self.game_id, time=None)

    @property
    def final_state(self):
        return self.states[-1]

    def get_state(self, time):
        return self.board.get_shipyard(self.game_id, time)

    def set_guard_ship_count(self, ship_count):
        if ship_count <= self.guard_ship_count:
            return
        self.guard_ship_count = min(self.ship_count, ship_count)

    @cached_property
    def incoming_allied_fleets(self) -> List["Fleet"]:
        player_id = self.player_id
        shipyard_id = self.game_id
        used = []
        fleets = []
        for board in self._board.simulations:
            for game_id, fleet in board.id_to_fleet.items():
                if game_id in used or fleet.player_id != player_id:
                    continue
                destination = fleet.destination
                if destination and destination.game_id == shipyard_id:
                    used.append(game_id)
                    fleets.append(fleet.final_state)
        return fleets

    @cached_property
    def incoming_hostile_fleets(self) -> List["Fleet"]:
        player_id = self.player_id
        shipyard_id = self.game_id
        used = []
        fleets = []
        for board in self._board.simulations:
            for game_id, fleet in board.id_to_fleet.items():
                if game_id in used or fleet.player_id == player_id:
                    continue
                destination = fleet.destination
                if destination and destination.game_id == shipyard_id:
                    used.append(game_id)
                    fleets.append(fleet.final_state)
        return fleets

    @cached_property
    def nearest_hostile_shipyards(self) -> List["Shipyard"]:
        shipyard_to_distance = {}
        for op in self.player.opponents:
            for sh in op.shipyards:
                shipyard_to_distance[sh] = self.distance_from(sh)
        if not shipyard_to_distance:
            return []
        min_distance = min(shipyard_to_distance.values())
        return [sy for sy, d in shipyard_to_distance.items() if d == min_distance]

    @cached_property
    def opposite_shipyards(self) -> List["Shipyard"]:
        opposite_shipyards = []
        for op_sy in self.nearest_hostile_shipyards:
            if self in op_sy.nearest_hostile_shipyards:
                opposite_shipyards.append(op_sy)
        return opposite_shipyards


@dataclass()
class PointData:
    kore: float

    regenerate: float = 0
    mined: Dict[Any, float] = dataclasses_field(default_factory=dict)
    dropped: Dict[Any, float] = dataclasses_field(default_factory=dict)


@dataclass()
class FleetOperations:
    convert: bool = False
    merge: Optional["Fleet"] = None
    direct_attack: Optional[list] = dataclasses_field(default_factory=list)
    adjacent_attack: Optional[list] = dataclasses_field(default_factory=list)
    shipyard_attack: Optional[list] = dataclasses_field(default_factory=list)
    kore_mined: float = 0
    kore_stolen: float = 0
    ships_lost: int = 0
    ships_hit: int = 0


class Fleet(PositionObj):
    def __init__(
        self,
        *args,
        ship_count: int,
        kore: float,
        flight_plan: str,
        direction: Action,
        **kwargs,
    ):
        assert ship_count > 0
        assert kore >= 0

        super().__init__(*args, **kwargs)
        self.kore = kore
        self.flight_plan = flight_plan
        self.ship_count = ship_count
        self.direction = direction
        self.operations = FleetOperations()

    def __gt__(self, other):
        if self.ship_count != other.ship_count:
            return self.ship_count > other.ship_count
        if self.kore != other.kore:
            return self.kore > other.kore
        return self.direction.game_id > other.direction.game_id

    @property
    def collection_rate(self):
        return collection_rate_for_ship_count(self.ship_count)

    def cost(self):
        return self.board.spawn_cost * self.ship_count

    def value(self):
        return self.kore / self.cost()

    @cached_property
    def states(self):
        return self.board.get_fleet(self.game_id, time=None)

    def get_state(self, time):
        return self.board.get_fleet(self.game_id, time=time)

    @property
    def final_state(self):
        return self.states[-1]

    @property
    def eta(self):
        return len(self.states)

    @cached_property
    def destination(self):
        states = self.states
        operation = states[-1].operations

        if operation.merge:
            return self.board.get_shipyard(operation.merge)

        if operation.shipyard_attack:
            if len(operation.shipyard_attack) != 1:
                logger.error(
                    f"Can't attack {len(operation.shipyard_attack)} shipyards at once"
                )
                return
            shipyard_id = operation.shipyard_attack[0]
            shipyard = self.board.id_to_shipyard.get(shipyard_id)
            if not shipyard:
                shipyard = self.board.id_to_fleet.get(shipyard_id)
                if not shipyard:
                    logger.error(f"{self}: Can't find shipyard {shipyard_id}")
                    return
            return shipyard

    def expected_kore(self):
        return self.final_state.kore

    def expected_value(self):
        return self.expected_kore() / self.cost()


class RouteChecker:
    def __init__(self, agent: "Player"):
        self.agent = agent
        self.board = agent.board
        op_shipyards = {}
        for time, board in enumerate(self.board.simulations):
            for sy_id, sy in board.id_to_shipyard.items():
                if sy.player_id == agent.game_id:
                    continue

                if sy_id in op_shipyards:
                    continue

                op_shipyards[sy_id] = (sy, time)

        self.op_shipyards = list(op_shipyards.values())

    @cached_call
    def estimate_power(self, point: Point, time: int) -> int:
        board = self.board

        max_power = 0
        for sy, creation_time in self.op_shipyards:
            if creation_time > time:
                continue

            distance = point.distance_from(sy.point)
            if creation_time + distance - 1 > time:
                continue

            launch_time = time - distance + 1
            if launch_time < creation_time:
                continue

            launched_sy = board.get_shipyard(sy.game_id, launch_time)
            if not launched_sy:
                last_board = board.simulations[-1]
                launched_sy = last_board.get_shipyard(sy.game_id)

            power = launched_sy.ship_count + launched_sy.max_ships_to_spawn * (
                launch_time - creation_time
            )

            if power > max_power:
                max_power = power

        return max_power

    def safety_ship_count(self, route: BoardRoute):
        if len(route.points) < 2:
            return 0
        return max(
            self.estimate_power(point, time)
            for time, point in enumerate(route.points[:-1], 1)
        )


class Player(Obj):
    def __init__(self, *args, kore: float, board: "Board", **kwargs):
        super().__init__(*args, **kwargs)
        self.kore = kore
        self._board = board

    def fleet_kore(self):
        return sum(x.kore for x in self.fleets)

    def fleet_expected_kore(self):
        return sum(x.expected_kore() for x in self.fleets)

    def is_active(self):
        for _ in self.shipyards:
            return True

        for _ in self.fleets:
            return True

        return False

    @property
    def board(self):
        return self._board

    def _get_objects(self, name):
        player_id = self.game_id
        for x in self._board.__getattribute__(name).values():
            if x.player_id == player_id:
                yield x

    @property
    def fleets(self) -> Generator[Fleet, None, None]:
        return self._get_objects("id_to_fleet")

    @property
    def shipyards(self) -> Generator[Shipyard, None, None]:
        return self._get_objects("id_to_shipyard")

    @property
    def ship_count(self) -> int:
        return sum(x.ship_count for x in itertools.chain(self.fleets, self.shipyards))

    @property
    def shipyard_count(self) -> int:
        return sum(1 for _ in self.shipyards)

    @cached_property
    def opponents(self) -> List["Player"]:
        return [x for x in self.board.players if x != self]

    @cached_property
    def opponent(self) -> Optional["Player"]:
        if len(self.opponents) != 1:
            return
        return self.opponents[0]

    @cached_property
    def expected_positions(
        self,
    ) -> Dict[int, Dict[Point, Tuple[Fleet, List[Fleet], Shipyard]]]:
        """
        time -> point -> (fleet, [(fleet, ship_count, in_contact), ...], shipyard)
        """
        data = defaultdict(dict)
        player_id = self.game_id
        for time, board in enumerate(self._board.simulations):

            point_to_fleet = {}
            for fleet in board.id_to_fleet.values():
                if fleet.player_id == player_id:
                    point_to_fleet[fleet.point] = fleet

            point_to_shipyard = {}
            for shipyard in board.id_to_shipyard.values():
                if shipyard.player_id == player_id:
                    point_to_shipyard[shipyard.point] = shipyard

            adjacent_point_to_fleet = {}
            for point, dmf_list in board.point_to_fleet_dmg.items():
                dmf_list = [x for x in dmf_list if x[0].player_id == player_id]
                if dmf_list:
                    adjacent_point_to_fleet[point] = dmf_list

            d = {}
            for point in (
                set(point_to_fleet)
                | set(adjacent_point_to_fleet)
                | set(point_to_shipyard)
            ):
                fleet = point_to_fleet.get(point)
                adjacent_fleets = adjacent_point_to_fleet.get(point, [])
                shipyard = point_to_shipyard.get(point)
                d[point] = (fleet, adjacent_fleets, shipyard)

            data[time] = d

        return data

    def actions(self):
        if self.available_kore() < 0:
            logger.warning("Negative balance. Some ships will not spawn.")

        shipyard_id_to_action = {}
        for sy in self.shipyards:
            action = sy.action

            if not action or isinstance(action, DoNothing):
                continue

            shipyard_id_to_action[sy.game_id] = action.to_str()
        return shipyard_id_to_action

    def spawn_ship_count(self):
        return sum(
            x.action.ship_count for x in self.shipyards if isinstance(x.action, Spawn)
        )

    def need_kore_for_spawn(self):
        return self.board.spawn_cost * self.spawn_ship_count()

    def available_kore(self):
        return self.kore - self.need_kore_for_spawn()

    @cached_property
    def route_checker(self) -> RouteChecker:
        return RouteChecker(self)

    @cached_property
    def closed_area(self) -> List[Point]:
        board = self._board
        agent_id = self._game_id

        point_status = []
        for game_id, sy in board.id_to_shipyard.items():
            point_status.append((sy.point, sy.player_id == agent_id))

        kore_area = []
        for p in board:
            points = sorted(point_status, key=lambda x: p.distance_from(x[0]))
            statuses = [x[1] for x in points][:3]
            if all(statuses):
                kore_area.append(p)

        return kore_area

    @cached_property
    def kore_shipyards(self) -> List[Shipyard]:
        if self.shipyard_count < 4:
            return []

        opponent = self.opponent
        if not opponent:
            return list(self.shipyards)

        agent_points = [x.point for x in self.shipyards]
        opponent_points = [x.point for x in opponent.shipyards]
        if not opponent_points:
            return list(self.shipyards)

        kore_shipyards = []
        for p in agent_points:
            agent_distances = [p.distance_from(x) for x in agent_points]
            op_distance = p.find_min_distance(opponent_points)
            if op_distance > agent_distances[3]:
                kore_shipyards.append(p)

        kore_shipyards = [x for x in self.shipyards if x.point in kore_shipyards]
        return kore_shipyards

    @cached_property
    def kore_area(self) -> List[Point]:
        board = self._board
        agent_id = self._game_id
        kore_shipyards = self.kore_shipyards
        if not kore_shipyards:
            return []

        point_status = []
        for game_id, sy in board.id_to_shipyard.items():
            if sy.player_id == agent_id and sy not in kore_shipyards:
                continue

            point_status.append((sy.point, sy.player_id == agent_id))

        kore_area = []
        for p in board:
            direction_to_status = defaultdict(list)
            for point, status in point_status:
                distance = p.distance_from(point)
                if distance == 0:
                    continue
                plans = p.plan_to(point)
                for plan in plans:
                    command = plan.paths[0].direction
                    direction_to_status[command].append((distance, status))

            is_kore = True
            for direction, distance_status in direction_to_status.items():
                status = sorted(distance_status, key=lambda x: x[0])[0][1]
                if not status:
                    is_kore = False
                    break

            if is_kore:
                kore_area.append(p)

        return kore_area


_FIELD = None


class Board:
    def __init__(self, step, conf):
        conf = self.configuration = Configuration(conf)
        self.step = step
        self.shipyard_cost = conf.convert_cost
        self.spawn_cost = conf.spawn_cost
        self.regen_rate = conf.regen_rate
        self.max_cell_kore = conf.max_cell_kore
        self.steps_left = conf.episode_steps - step - 1

        global _FIELD
        if _FIELD is None or step == 0:
            _FIELD = Field(conf.size)
        else:
            assert _FIELD.size == conf.size

        self.field: Field = _FIELD

        self.point_to_data = {p: PointData(kore=0) for p in self.field}
        self.id_to_player: Dict[Any, Player] = {}
        self.id_to_fleet: Dict[Any, Fleet] = {}
        self.id_to_shipyard: Dict[Any, Shipyard] = {}
        self.point_to_fleet_dmg = {}

        self.simulations: List["Board"] = []

    @classmethod
    def from_raw(cls, obs, conf) -> "Board":
        conf = Configuration(conf)
        step = obs["step"]

        board = Board(step, conf)

        id_to_point = {x.game_id: x for x in board}

        for point_id, kore in enumerate(obs["kore"]):
            point = id_to_point[point_id]
            board.point_data(point).kore = kore

        players = []
        fleets = []
        shipyards = []

        for player_id, player_data in enumerate(obs["players"]):
            player_kore, player_shipyards, player_fleets = player_data
            player = Player(game_id=player_id, kore=player_kore, board=board)
            players.append(player)

            for fleet_id, fleet_data in player_fleets.items():
                point_id, kore, ship_count, direction, flight_plan = fleet_data
                position = id_to_point[point_id]
                direction = GAME_ID_TO_ACTION[direction]
                fleet = Fleet(
                    game_id=fleet_id,
                    point=position,
                    player_id=player_id,
                    ship_count=ship_count,
                    kore=kore,
                    board=board,
                    flight_plan=flight_plan,
                    direction=direction,
                )
                fleets.append(fleet)

            for shipyard_id, shipyard_data in player_shipyards.items():
                point_id, ship_count, turns_controlled = shipyard_data
                position = id_to_point[point_id]
                shipyard = Shipyard(
                    game_id=shipyard_id,
                    point=position,
                    player_id=player_id,
                    ship_count=ship_count,
                    turns_controlled=turns_controlled,
                    board=board,
                )
                shipyards.append(shipyard)

        for player in players:
            board.add_player(player)

        for fleet in fleets:
            board.add_fleet(fleet)

        for shipyard in shipyards:
            board.add_shipyard(shipyard)

        return board

    def to_raw(self):
        kore = []
        for p in sorted(self, key=lambda x: x.game_id):
            kore.append(self.kore_at_point(p))

        players = []
        for p in sorted(self.players, key=lambda x: x.game_id):
            fleets = {
                x.game_id: [
                    x.point.game_id,
                    x.kore,
                    x.ship_count,
                    x.direction.game_id,
                    x.flight_plan,
                ]
                for x in p.fleets
            }
            shipyards = {
                x.game_id: [x.point.game_id, x.ship_count, x.turns_controlled]
                for x in p.shipyards
            }
            players.append([p.kore, shipyards, fleets])

        data = []
        for i, p in enumerate(sorted(self.players, key=lambda x: x.game_id)):
            if i == 0:
                observation = {
                    "kore": kore,
                    "player": p.game_id,
                    "players": players,
                    "remainingOverageTime": 60,
                    "step": self.step,
                }
            else:
                observation = {
                    "player": p.game_id,
                    "remainingOverageTime": 60,
                }
            data.append(
                {
                    "action": {},
                    "info": {},
                    "observation": observation,
                    "reward": p.kore,
                    "status": "ACTIVE" if p.is_active() else "DONE",
                }
            )
        return data

    def __repr__(self):
        return f"Board(step={self.step})"

    def __contains__(self, item):
        if isinstance(item, Player):
            return item.game_id in self.id_to_player
        elif isinstance(item, Fleet):
            return item.game_id in self.id_to_fleet
        elif isinstance(item, Shipyard):
            return item.game_id in self.id_to_shipyard
        elif isinstance(item, Point):
            for p in self:
                if p.game_id == item.game_id:
                    return True
            return False
        raise ValueError

    def __getitem__(self, item):
        return self.field[item]

    def __iter__(self):
        return self.field.__iter__()

    def kore_at_point(self, point: Point) -> float:
        return self.point_to_data[point].kore

    def point_data(self, point: Point) -> PointData:
        return self.point_to_data[point]

    @property
    def players(self) -> List[Player]:
        return list(self.id_to_player.values())

    @property
    def fleets(self) -> List[Fleet]:
        return list(self.id_to_fleet.values())

    @property
    def shipyards(self) -> List[Shipyard]:
        return list(self.id_to_shipyard.values())

    def add_player(self, player: Player):
        self.id_to_player[player.game_id] = player

    def add_fleet(self, fleet: Fleet):
        self.id_to_fleet[fleet.game_id] = fleet

    def add_shipyard(self, shipyard):
        self.id_to_shipyard[shipyard.game_id] = shipyard

    def get_player(
        self, game_id, time: Optional[int] = 0
    ) -> Optional[Union[Player, List[Player]]]:
        return self.get_obj("id_to_player", game_id, time)

    def get_fleet(
        self, game_id, time: Optional[int] = 0
    ) -> Optional[Union[Fleet, List[Fleet]]]:
        return self.get_obj("id_to_fleet", game_id, time)

    def get_shipyard(
        self, game_id, time: Optional[int] = 0
    ) -> Optional[Union[Shipyard, List[Shipyard]]]:
        return self.get_obj("id_to_shipyard", game_id, time)

    def get_obj(self, field, game_id, time=0):
        if time == 0:
            return self.__getattribute__(field).get(game_id)
        elif time is not None:
            try:
                board = self.simulations[time]
            except IndexError:
                return
            return board.__getattribute__(field).get(game_id)
        else:
            out = []
            for board in self.simulations:
                obj = board.__getattribute__(field).get(game_id)
                if not obj:
                    break
                out.append(obj)
            return out

    def del_fleet(self, game_id):
        self.id_to_fleet.pop(game_id)

    def get_obj_at_point(self, point: Point) -> Optional[Union[Fleet, Shipyard]]:
        for x in itertools.chain(self.fleets, self.shipyards):
            if x.point == point:
                return x

    def next_step_kore(self, kore):
        return next_step_kore(kore, self.regen_rate, self.max_cell_kore)

    @staticmethod
    def get_largest_fleet(fleets):
        fleet = fleets[0]
        for f in fleets[1:]:
            if f > fleet:
                fleet = f
        return fleet

    @cached_property
    def shipyard_positions(self) -> Set[Point]:
        return {sy.point for sy in self.id_to_shipyard.values()}

    def _direct_fleet_to_fleet(self, next_board):
        point_to_fleets = defaultdict(list)
        for fleet in next_board.id_to_fleet.values():
            point_to_fleets[fleet.point].append(fleet)

        point_to_shipyard = {}
        for shipyard in next_board.id_to_shipyard.values():
            point_to_shipyard[shipyard.point] = shipyard

        for point, fleets in point_to_fleets.items():
            shipyard = point_to_shipyard.get(point)
            self._resolve_direct_collision(next_board, point, fleets, shipyard)

    def _resolve_direct_collision(self, next_board, point, fleets, shipyard=None):
        if len(fleets) < 2:
            return

        next_fleet = self.get_largest_fleet(fleets)
        dmg = max(f.ship_count for f in fleets if f != next_fleet)

        if dmg >= next_fleet.ship_count:
            point_data = next_board.point_data(point)
            for f in fleets:
                if shipyard:
                    shipyard.player.kore += f.kore
                else:
                    point_data.dropped[f.game_id] = f.kore
                    point_data.kore += f.kore
                next_board.del_fleet(f.game_id)

                fleet = self.id_to_fleet.get(f.game_id)
                if not fleet:
                    continue

                enemy = [x for x in fleets if x != f]
                fleet.operations.direct_attack += [x.game_id for x in enemy]
                fleet.operations.ships_hit += sum(x.ship_count for x in enemy)
                fleet.operations.ships_lost += f.ship_count
            return

        next_fleet.ship_count -= dmg
        next_fleet.kore = sum(f.kore for f in fleets)

        fleet = self.id_to_fleet.get(next_fleet.game_id)
        if fleet:
            enemy = [x for x in fleets if x != next_fleet]
            fleet.operations.direct_attack += [x.game_id for x in enemy]
            fleet.operations.ships_hit += sum(x.ship_count for x in enemy)
            fleet.operations.ships_lost += dmg
            fleet.operations.kore_stolen += sum(x.kore for x in enemy)

        other_fleets = [x for x in fleets if x != next_fleet]
        largest_other_fleet = self.get_largest_fleet(other_fleets)

        for f in other_fleets:
            next_board.del_fleet(f.game_id)
            fleet = self.id_to_fleet.get(f.game_id)
            if not fleet:
                continue
            enemy = [x for x in fleets if x != f]
            fleet.operations.direct_attack += [x.game_id for x in enemy]
            if f == largest_other_fleet:
                fleet.operations.ships_hit += f.ship_count
            fleet.operations.ships_lost += f.ship_count

    def _move_fleet(self, fleet: Fleet, next_board):
        flight_plan = fleet.flight_plan.lstrip("0")

        if (
            flight_plan
            and flight_plan.startswith(Convert.command)
            and fleet.ship_count < self.shipyard_cost
        ):
            flight_plan = flight_plan.lstrip(Convert.command)

        if flight_plan and flight_plan[0] == Convert.command:
            next_shipyard = Shipyard(
                game_id=fleet.game_id,
                point=fleet.point,
                player_id=fleet.player_id,
                ship_count=fleet.ship_count - self.shipyard_cost,
                turns_controlled=0,
                board=next_board,
            )
            fleet.operations.convert = True
            next_board.get_player(fleet.player_id).kore += fleet.kore
            next_board.add_shipyard(next_shipyard)
            return

        if not flight_plan:
            direction = fleet.direction
        elif flight_plan[0].isalpha():
            direction = COMMAND_TO_ACTION[flight_plan[0]]
            flight_plan = flight_plan[1:]
        else:
            direction = fleet.direction
            digits = ""
            for d in flight_plan:
                if d.isdigit():
                    digits += d
                else:
                    break
            flight_plan = flight_plan.lstrip(digits)
            digits = int(digits)
            if digits > 1:
                flight_plan = str(digits - 1) + flight_plan

        next_fleet = Fleet(
            game_id=fleet.game_id,
            point=fleet.point.apply(direction),
            player_id=fleet.player_id,
            ship_count=fleet.ship_count,
            kore=fleet.kore,
            board=next_board,
            flight_plan=flight_plan,
            direction=direction,
        )
        next_board.add_fleet(next_fleet)

    def _move_fleets(self, next_board):
        for agent in self.id_to_player.values():
            for fleet in agent.fleets:
                self._move_fleet(fleet, next_board)

    def _combine_fleets(self, next_board):
        for agent in next_board.id_to_player.values():
            point_to_fleets = defaultdict(list)
            for f in agent.fleets:
                point_to_fleets[f.point].append(f)

            for point_fleets in point_to_fleets.values():
                if len(point_fleets) < 2:
                    continue

                next_fleet = self.get_largest_fleet(point_fleets)

                kore = 0
                ship_count = 0
                for f in point_fleets:
                    if f == next_fleet:
                        continue
                    kore += f.kore
                    ship_count += f.ship_count
                    next_board.del_fleet(f.game_id)
                    fleet = self.id_to_fleet.get(f.game_id)
                    if fleet:
                        fleet.operations.merge = next_fleet.game_id

                next_fleet.kore += kore
                next_fleet.ship_count += ship_count

    def _fleet_to_shipyard(self, next_board):
        point_to_shipyard = {sy.point: sy for sy in next_board.id_to_shipyard.values()}
        for game_id, fleet in list(next_board.id_to_fleet.items()):
            if fleet.point not in point_to_shipyard:
                continue

            sy = point_to_shipyard[fleet.point]
            pf = self.id_to_fleet.get(game_id)

            if sy.player_id == fleet.player_id:
                sy.ship_count += fleet.ship_count
                sy.player.kore += fleet.kore
                next_board.del_fleet(game_id)
                if pf:
                    pf.operations.merge = sy.game_id
                continue

            if sy.ship_count >= fleet.ship_count:
                sy.ship_count -= fleet.ship_count
                sy.player.kore += fleet.kore
                next_board.del_fleet(game_id)
                if pf:
                    operations = pf.operations
                    operations.shipyard_attack.append(sy.game_id)
                    operations.ships_lost += fleet.ship_count
                    operations.ships_hit += fleet.ship_count
                    operations.kore_stolen = 0

            else:
                sy.ship_count = fleet.ship_count - sy.ship_count
                sy.player_id = fleet.player_id
                sy.turns_controlled = 1
                fleet.player.kore += fleet.kore
                next_board.del_fleet(game_id)
                if pf:
                    pf.operations.shipyard_attack.append(sy.game_id)
                    pf.operations.ships_lost += sy.ship_count
                    pf.operations.ships_hit += sy.ship_count

    def _adjacent_fleet_to_fleet(self, next_board):
        point_to_fleet = {}
        point_to_fleet_dmg_mem = next_board.point_to_fleet_dmg
        shipyard_points = {sy.point for sy in next_board.shipyards}
        for f in next_board.id_to_fleet.values():
            point_to_fleet[f.point] = f

            for point in f.point.adjacent_points:
                if point in shipyard_points:
                    continue

                if point not in point_to_fleet_dmg_mem:
                    point_to_fleet_dmg_mem[point] = [[f, f.ship_count, False]]
                else:
                    point_to_fleet_dmg_mem[point].append([f, f.ship_count, False])

        point_to_fleet_dmg = defaultdict(dict)
        fleet_to_list_of_points = defaultdict(list)
        for point, fleet in point_to_fleet.items():
            for p in point.adjacent_points:
                a_fleet = point_to_fleet.get(p)
                if a_fleet and a_fleet.player_id != fleet.player_id:
                    point_to_fleet_dmg[point][a_fleet] = a_fleet.ship_count
                    fleet_to_list_of_points[a_fleet.game_id].append(point)

        to_distribute = defaultdict(dict)
        for point, fleet in point_to_fleet.items():
            fleet_dmg = point_to_fleet_dmg.get(point)
            if fleet_dmg is None:
                continue

            total_dmg = sum(fleet_dmg.values())

            ships_lost = min(total_dmg, fleet.ship_count)

            pf = self.id_to_fleet.get(fleet.game_id)
            if pf:
                pf.operations.ships_lost += ships_lost

            for f, dmg in fleet_dmg.items():
                if ships_lost >= fleet.ship_count:
                    to_distribute[f][point] = fleet.kore / 2 * dmg / total_dmg

                pf = self.id_to_fleet.get(f.game_id)
                if pf:
                    pf.operations.adjacent_attack.append(fleet.game_id)
                    if ships_lost >= fleet.ship_count:
                        pf.operations.ships_hit += fleet.ship_count * dmg / total_dmg
                    else:
                        pf.operations.ships_hit += dmg

            if ships_lost >= fleet.ship_count:
                pd = next_board.point_data(point)
                pd.kore += fleet.kore / 2
                pd.dropped[fleet.game_id] = fleet.kore / 2
                next_board.del_fleet(fleet.game_id)
            else:
                fleet.ship_count -= ships_lost

        for fleet, point_to_kore in to_distribute.items():
            pf = self.id_to_fleet.get(fleet.game_id)

            if fleet.game_id in next_board.id_to_fleet:
                kore_captured = sum(point_to_kore.values())
                fleet.kore += kore_captured
                if pf:
                    pf.operations.kore_stolen += kore_captured
            else:
                for point, kore in point_to_kore.items():
                    f = point_to_fleet[point]
                    pd = next_board.point_data(point)
                    pd.kore += kore
                    pd.dropped[f.game_id] += kore

        for point, fd_list in point_to_fleet_dmg_mem.items():
            for fd in fd_list:
                point_list = fleet_to_list_of_points.get(fd[0].game_id)
                if not point_list:
                    continue
                if len(point_list) == 1 and point_list[0] == point:
                    continue
                fd[2] = True

    def _mine(self, next_board):
        for game_id, fleet in next_board.id_to_fleet.items():
            pf = self.id_to_fleet.get(game_id)
            pd = next_board.point_data(fleet.point)
            delta_kore = amount_of_collected_kore(pd.kore, fleet.collection_rate)
            pd.mined[game_id] = delta_kore
            pd.kore -= delta_kore
            fleet.kore += delta_kore
            if pf:
                pf.operations.kore_mined = delta_kore

    def _regenerate(self, next_board):
        fleet_points = {x.point for x in next_board.id_to_fleet.values()}
        shipyard_points = {x.point for x in next_board.id_to_shipyard.values()}
        regen_rate = self.regen_rate
        max_cell_kore = self.max_cell_kore
        for point, point_data in next_board.point_to_data.items():
            kore = point_data.kore
            if point in shipyard_points:
                next_kore = 0
            elif point in fleet_points:
                next_kore = kore
            else:
                next_kore = next_step_kore(kore, regen_rate, max_cell_kore)
                point_data.regenerate = next_kore - kore

            point_data.kore = next_kore

    def _apply_shipyard_actions(self, next_board, actions):
        for shipyard in self.id_to_shipyard.values():
            next_shipyard = Shipyard(
                game_id=shipyard.game_id,
                point=shipyard.point,
                player_id=shipyard.player_id,
                ship_count=shipyard.ship_count,
                turns_controlled=shipyard.turns_controlled + 1,
                board=next_board,
            )
            next_board.add_shipyard(next_shipyard)

            if not actions or next_shipyard.game_id not in actions:
                continue

            action = actions[next_shipyard.game_id]

            if isinstance(action, _ShipyardAction):
                action = action.to_str()

            if action.startswith("SPAWN_"):
                num_ships = int(action.lstrip("SPAWN_"))
                if (
                    1 <= num_ships <= shipyard.max_ships_to_spawn
                    and next_shipyard.player.kore >= num_ships * self.spawn_cost
                ):
                    next_shipyard.player.kore -= num_ships * self.spawn_cost
                    next_shipyard.ship_count += num_ships
            elif action.startswith("LAUNCH_"):
                num_ships, flight_plan = action.lstrip("LAUNCH_").split("_")
                num_ships = int(num_ships)
                flight_plan = flight_plan[
                    : max_flight_plan_len_for_ship_count(num_ships)
                ]
                if 1 <= num_ships <= next_shipyard.ship_count:
                    next_shipyard.ship_count -= num_ships
                    direction = COMMAND_TO_ACTION[flight_plan[0]]
                    fleet = Fleet(
                        game_id=f"{next_shipyard.game_id}-{self.step + 1}",
                        point=next_shipyard.point,
                        player_id=next_shipyard.player_id,
                        ship_count=num_ships,
                        kore=0,
                        board=self,
                        flight_plan=flight_plan,
                        direction=direction,
                    )
                    self._move_fleet(fleet, next_board)
            else:
                raise ValueError(f"Unknown action {action}")

    def next(self, actions=None) -> "Board":
        next_board = Board(self.step + 1, self.configuration)

        for point, point_data in self.point_to_data.items():
            next_board.point_to_data[point].kore = point_data.kore

        for agent in self.id_to_player.values():
            next_board.add_player(
                Player(game_id=agent.game_id, kore=agent.kore, board=next_board)
            )

        self._apply_shipyard_actions(next_board, actions)
        self._move_fleets(next_board)
        self._combine_fleets(next_board)
        self._direct_fleet_to_fleet(next_board)
        self._fleet_to_shipyard(next_board)
        self._adjacent_fleet_to_fleet(next_board)
        self._mine(next_board)
        self._regenerate(next_board)

        return next_board

    def to_array(self, points):
        if isinstance(points, dict):

            def get_value(_p):
                return points[_p]

        else:

            def get_value(_p):
                return 1

        s = self.field.size

        array = np.zeros((s, s), dtype=np.int32)
        for p in points:
            array[s - 1 - p.y, p.x] = get_value(p)
        return array
