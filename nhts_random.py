from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt

import logging as pylog
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
import random as rndm
import ast
import json

# local imports
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(__file__))
from human_agent.human_agent import Human, Human_Config
from human_agent import disease
from places import Location


agent_tuple_cache = {}
log = pylog.getLogger(__name__)

def restore_agent(agent_tuple: Tuple):
    """
    Deer tuple example: ((5, 0, 1), CFG_Dict_Object)
    (5, 0 ,1) == (AgentID, Type, Rank)

    Args:
        agent_tuple: tuple containing the data returned by save function.
        agent_tuple[0][0]: Agent ID
        agent_tuple[0][1]: Agent Type
        agent_tuple[0][2]: Rank
        agent_tuple[1]: CFG Dict
    """
    log.debug(f'Restoring agent: {agent_tuple}')
    uid = agent_tuple[0]

    if uid in agent_tuple_cache:
        # Agent exists in cache.
        agent = agent_tuple_cache[uid]
    else:
        # Cache this agent
        agent = Human(uid[0], uid[1], uid[2]) #(agent_id, rank)
        agent_cfg = agent_tuple[1]
        agent.set_cfg(agent_cfg)
        agent_tuple_cache[uid] = agent

    agent_cfg = agent_tuple[1]
    agent.set_cfg(agent_cfg)

    return agent

import resource as _resource
import socket as _socket
import time as _time

_STARTED = _time.perf_counter()
_STATES = ('Susceptible', 'Exposed', 'Infectious', 'Recovered')
_SEED = None


def seed_everything(params):
    """Seed the generators on every process from the run's base seed (`random.seed` in the config).
    Process r uses base * 1000 + r for Repast4Py's generator, NumPy's global generator and Python's random.
    The initial infections use their own fixed generator, default_rng(seed=42)."""
    global _SEED
    if 'random.seed' not in params:
        raise SystemExit("the config must set 'random.seed', the run's base seed")
    rank = MPI.COMM_WORLD.Get_rank()
    _SEED = int(params['random.seed']) * 1000 + rank
    random.init(_SEED)        # Repast4Py's generator: activity choices, Random Places' destinations
    np.random.seed(_SEED)     # NumPy's global generator: destinations, infection draws
    rndm.seed(_SEED)          # Python's random: dwell times, disease durations
    print(f'rank {rank} seed {_SEED}', flush=True)


def order_keys(ids, base_seed):
    """A random key per agent, fixed for the run and the same on every process (SplitMix64 of id and seed)."""
    mask = 0xFFFFFFFFFFFFFFFF
    x = np.asarray(ids, dtype=np.uint64) + np.uint64((int(base_seed) * 0x9E3779B97F4A7C15) & mask)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def write_log():
    """Whether to write the full agent log: yes unless the config sets write_agent_log: false."""
    return params.get('write_agent_log', True) not in (False, 'false', 'False', 0)


def out_dir():
    return os.path.dirname(params['agent_log_file']) or '.'


def end_of_tick(model_):
    """Called at the end of every step on every process. Rank 0 appends the agents in each state and the new
    infections to summary_ticks.csv. The agents infectious after the first tick are the initial cases."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    first = not hasattr(model_, '_initial_ids')
    tick = int(model_.runner.schedule.tick)
    counts = dict.fromkeys(_STATES, 0)
    infectious = []
    for h in model_.context.agents():
        counts[h.disease_state.value] += 1
        if first and h.disease_state is disease.Disease_State.INFECTIOUS:
            infectious.append(h.id)
    new_exposed = getattr(model_, '_new_exposed', 0)
    model_._new_exposed = 0
    rows = comm.gather((counts, new_exposed, infectious), root=0)
    if first:
        ids = sorted(i for r in rows for i in r[2]) if rank == 0 else None
        model_._initial_ids = set(comm.bcast(ids, root=0))
    if rank == 0:
        path = os.path.join(out_dir(), 'summary_ticks.csv')
        if first:
            with open(path, 'w') as f:
                f.write('tick,' + ','.join(_STATES) + ',new_exposed\n')
        total = {s: sum(r[0][s] for r in rows) for s in _STATES}
        with open(path, 'a') as f:
            f.write(f"{tick}," + ','.join(str(total[s]) for s in _STATES) + f",{sum(r[1] for r in rows)}\n")


def finish(model_):
    """At the end of the run, rank 0 writes initial_cases.csv (each initial case's counter, for R0) and
    run_info.json (seeds, hosts, peak memory and wall time per process, R0, log size)."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ids = getattr(model_, '_initial_ids', set())
    mine = [(h.id, h.infected_counter) for h in model_.context.agents() if h.id in ids]
    info = {'rank': rank, 'host': _socket.gethostname(), 'seed': _SEED,
            'peak_rss_mb': round(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            'wall_seconds': round(_time.perf_counter() - _STARTED, 1)}
    rows = comm.gather((mine, info), root=0)
    if rank == 0:
        out = out_dir()
        cases = sorted(c for r in rows for c in r[0])
        with open(os.path.join(out, 'initial_cases.csv'), 'w') as f:
            f.write('agent_id,n_infected\n' + ''.join(f'{a},{n}\n' for a, n in cases))
        fpath = getattr(model_.agent_logger, '_fpath', None)
        run_info = {'base_seed': int(params['random.seed']), 'processes': comm.Get_size(),
                    'stop_at': params['stop.at'], 'initial_cases': len(cases),
                    'R0': round(sum(n for _, n in cases) / len(cases), 4) if cases else None,
                    'agent_log': str(fpath) if (fpath and write_log()) else None,
                    'agent_log_bytes': os.path.getsize(fpath) if (fpath and write_log() and os.path.exists(fpath)) else 0,
                    'per_process': [r[1] for r in rows]}
        with open(os.path.join(out, 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=1)


class Model:
    """
    The Model class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behaviour of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """
    def __init__(self, comm: MPI.Intracomm, params: Dict):
        '''
        Initialise the Model Class
        '''
        log.info('Initialising Repast model...')
        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.end_sim)

        self.rank = comm.Get_rank()
        self.load_buildings_dict()
        self.load_dwell_time()
        self.load_restaurants()
        self.load_recreations()
        self.load_errands()
        self.load_hospitals()

        coords = [1577749, 44120, 1893069, 44888]    # covers every building
        self.building_ids = list(self.dbuildings.keys())

        # model parameters
        self.infection_chance = params['infection_chance']

        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(comm)

        box = space.BoundingBox(coords[0], coords[1],
                                coords[2], coords[3], 0, 0)
        #box = space.BoundingBox(1595925, 45180,
        #                        1952145, 27240, 0, 0)
        #box = space.BoundingBox(0, 100,
        #                        0, 100, 0, 0)
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple, buffer_size=10, comm=comm)
        self.context.add_projection(self.grid)

        #self.space = space.SharedCSpace('space', bounds=box, borders=space.BorderType.Sticky,
        #                                buffer_size=10, comm=comm,
        #                                tree_threshold=100)

        # Initialize human logging
        #self.agent_logger = logging.TabularLogger(comm, params['agent_log_file'],
        #                                          ['tick', 'agent_id', 'agent_uid_rank',
        #                                           'x', 'y','disease_state', 'location'])
        log_file = params['agent_log_file']+str(self.rank)+'.csv'
        self.agent_logger = logging.TabularLogger(comm, log_file,
                                                  ['tick', 'agent_id','disease_state',
                                                   'x', 'y','n_infected', 'agents_contacted', 'trip_day'])
        #self.agent_logger = logging.TabularLogger(comm, log_file,
        #                                          ['tick', 'agent_id','disease_state',
        #                                           'x', 'y','n_infected', 'trip_day'])
        #self.agent_logger = logging.TabularLogger(comm, log_file,
        #                                    ['tick', 'agent_id','disease_state'])

        # Initialise human agents
        n_agents = params['pop_size']
        world_size = comm.Get_size()
        humans_per_rank = int(n_agents / world_size)
        remaining_humans = n_agents % world_size
        if self.rank < remaining_humans:
            humans_per_rank += 1
        first_id = self.rank * humans_per_rank      # id of the first agent on this process
        if remaining_humans > 0:
            if self.rank >= remaining_humans:
                first_id += remaining_humans

        # REVISIT: n_agents should be no. of agents on this rank
        rng = np.random.default_rng(seed=42)
        init_infected = rng.choice(n_agents, int(n_agents * params['infectious_start_rate']))
        log.debug(f'init_infected: {init_infected}')
        init_infected = np.isin(np.arange(n_agents), init_infected)
        infectious_count = init_infected.sum()

        with open(params['input_agents']) as file:
            next(file)          # skip header
            for input_agent in range(0, first_id+humans_per_rank):
                if input_agent < first_id:          # skip input rows till agent id for this process
                    next(file)
                else:
                    input_line = file.readline()
                    human_id = int(input_line.split('[')[0].split(',')[0])
                    nhts_id = (int(input_line.split('[')[0].split(',')[1]))
                    agent_type = (int(input_line.split('[')[0].split(',')[2]))
                    census_tract = (int(input_line.split('[')[0].split(',')[3]))
                    trip_chain = ast.literal_eval("["+input_line.split('[')[1].split('"')[0])
                    # clustered spreading infectious agents, if disabled, uniformly spreading
                    # infect 100 in specific CT
                    if census_tract != 51059440800:
                        init_infected[human_id] = False
                        infectious_count = infectious_count-1
                    self.add_human_agent(human_id, nhts_id, trip_chain, census_tract,
                                            agent_type, init_infected[human_id])

        log.info(f"Creating {n_agents} human agents over {world_size} processes ({humans_per_rank} agents/process)...")
        log.info(f'initially infected: {infectious_count}')
        log.info(f'number of buildings: {len(self.building_ids)}')
        log.info(f'grid: {box}')

        self.tick_duration = 60               # minutes

        sim_start_str = '01/01/24 00:00:00'
        self.sim_start = datetime.strptime(sim_start_str, '%m/%d/%y %H:%M:%S')
        self.current_time = self.sim_start

    # creates buildings from file
    def load_buildings_dict(self):

        # --- Creating dictionary with buildings
        self.dbuildings = {}

        # --- Reading the input file
        file_path=params['input_buildings']
        dframe = pd.read_csv(file_path)
        #self.dbuildings = {dframe.iloc[i]['bldg_id']:
        #                        {"type": int(dframe.iloc[i]['bldg_type']),
        #                            "pos" : np.array([dframe.iloc[i]['x_centroid'], dframe.iloc[i]['y_centroid']],
        #                            dtype = int)}
        #                    for i in dframe.T}
        self.dbuildings = {dframe.iloc[i]['bldg_id']:
                        {"type": dframe.iloc[i]['bldg_type'],
                            "pos" : np.array([dframe.iloc[i]['x_centroid'], dframe.iloc[i]['y_centroid']],
                            dtype = int)}
                    for i in dframe.T}

    # creates dwell time from file
    def load_dwell_time(self):
        with open(params['input_dwell_time'], "r") as json_file:
            dwell_time = json.load(json_file)

        self.dwell_time = {int(k): v for k, v in dwell_time.items()}

    # read restaurants from file
    def load_restaurants(self):
        with open(params['input_restaurants'], "r") as json_file:
            restaurants = json.load(json_file)

        restaurants = {int(k): (np.array(v[0]), np.array(v[1])) for k, v in restaurants.items()}

        # get unique ids of restaurants
        restaurant_ids = []
        for val in restaurants.values():
            restaurant_ids.extend(val[0])
        self.restaurant_ids = np.unique(np.array(restaurant_ids))

    # read recreations from file
    def load_recreations(self):
        with open(params['input_recreations'], "r") as json_file:
            recreations = json.load(json_file)

        recreations = {int(k): (np.array(v[0]), np.array(v[1])) for k, v in recreations.items()}

        # get unique ids of recreations
        recreation_ids = []
        for val in recreations.values():
            recreation_ids.extend(val[0])
        self.recreation_ids = np.unique(np.array(recreation_ids))

    # read errands from file
    def load_errands(self):
        with open(params['input_errands'], "r") as json_file:
            errands = json.load(json_file)

        errands = {int(k): (np.array(v[0]), np.array(v[1])) for k, v in errands.items()}

        # get unique ids of errands
        errand_ids = []
        for val in errands.values():
            errand_ids.extend(val[0])
        self.errand_ids = np.unique(np.array(errand_ids))

    # read hospitals from file
    def load_hospitals(self):
        with open(params['input_hospitals'], "r") as json_file:
            hospitals = json.load(json_file)

        hospitals = {int(k): (np.array(v[0]), np.array(v[1])) for k, v in hospitals.items()}

        # get unique ids of hospitals
        hospital_ids = []
        for val in hospitals.values():
            hospital_ids.extend(val[0])
        self.hospital_ids = np.unique(np.array(hospital_ids))


    def end_sim(self):
        '''
        Clean up and log data.
        '''
        log.info(f'Cleaning up simulation...')
        self.end_start_timestamp = datetime.now()
        self.agent_logger.close()
        finish(self)

        return

    def add_human_agent(self, human_id, nhts_id, trip_chain, census_tract, agent_type, init_infected):
        log.debug(f'Testing input file!')
        '''
        Add agent to the sim located at Point(x,y) based on the home id
        '''
        home_id = trip_chain[-1]
        start_minutes = self.dwell_time[nhts_id][0]
        building_to_go = self.dbuildings[home_id]
        pt = dpt(building_to_go["pos"][0],
                building_to_go["pos"][1])

        # agent type: 1 - worker, 2 - student, 3 - homemaker
        # create and add human to the context
        #else:

        human = Human(human_id, agent_type, self.rank)
        human_cfg = Human_Config()
        human_cfg_object = human_cfg.rand_factory(pt.x, pt.y, home_id, nhts_id, trip_chain, census_tract,
                                                  start_minutes, init_infected, params)
        human.set_cfg(human_cfg_object)
        self.context.add(human)
        self.grid.move(human, pt)

    def nhts_random_schedule(self, human):
        if(human.trip_end_datetime <= self.current_time):
            next_trip = human.trip_chain[human.trip_num]
            log.debug(f"trip_end_time: {human.trip_end_datetime}")
            log.debug(f"current_time: {self.current_time}, going new place")
            if next_trip == 'restaurants':
                next_trip = np.random.choice(self.restaurant_ids)
            elif next_trip == 'recreation':
                next_trip = np.random.choice(self.recreation_ids)
            elif next_trip == 'errands':
                next_trip = np.random.choice(self.errand_ids)
            elif next_trip == 'hospitals':
                next_trip = np.random.choice(self.hospital_ids)
            #elif next_trip == human.home_id:
            #elif human.type == 1:
            #else:

            log.debug(f"human.id: {human.id}")
            log.debug(f"human.trip_chain: {human.trip_chain}")
            log.debug(f"human.nhts_id: {human.nhts_id}")
            log.debug(f"human.trip_num: {human.trip_num}")
            log.debug(f"human.dwell_time: {self.dwell_time[human.nhts_id]}")
            dwell_time = self.dwell_time[human.nhts_id][human.trip_num+1]
            human.trip_end_datetime = self.current_time + timedelta(minutes=dwell_time)
            log.debug(f"new trip_end_time: {human.trip_end_datetime}")
            self.go_to_place(human, next_trip)
            # finished 1-day activities
            if human.trip_num == 0:
                human.trip_day = human.trip_day + 1
            human.trip_num += 1
            if human.trip_num == len(human.trip_chain):
                human.trip_num = 0

    def go_to_place(self, human, place_id):
        next_bldg = self.dbuildings[place_id]
        x = next_bldg["pos"][0]
        y = next_bldg["pos"][1]
        human.move_coords(self, x, y)


    def disease_spread(self, human, infect_prob):
        # find and infect another susceptible agent at same place if human is infected
        pt = model.grid.get_location(human)
        n_at_place = model.grid.get_num_agents(pt)
        n_potential_agents = round(np.log2(n_at_place))

        # line up the agents present by a key fixed for the run (the same on every process); each agent
        # contacts the next n_potential_agents after itself in that line, wrapping round. Cached per place and tick.
        tick = self.runner.schedule.tick
        if getattr(self, '_order_tick', None) != tick:
            self._order_tick, self._order = tick, {}
        place = (pt.x, pt.y)
        if place not in self._order:
            present = list(self.grid.get_agents(pt))
            keys = order_keys([a.id for a in present], params['random.seed'])
            line = [present[i] for i in np.argsort(keys, kind='stable')]
            self._order[place] = (line, {a.uid: i for i, a in enumerate(line)})
        line, position = self._order[place]
        n = len(line)
        me = position[human.uid]

        for step_ahead in range(1, min(n_potential_agents, n - 1) + 1):
            other_agent = line[(me + step_ahead) % n]
            human.unique_contacts.append(other_agent.id)
            if human.disease_state is disease.Disease_State.INFECTIOUS:
                if infect_prob < self.infection_chance:
                    # let an infectious agent infect susceptible with a given probability
                    if other_agent.disease_state == disease.Disease_State.SUSCEPTIBLE:
                        disease.become_exposed(other_agent, params)
                        # count only actual new infections, during the first infectious period, for R0
                        if human.count_infected == True:
                            human.infected_counter += 1
                        self._new_exposed = getattr(self, '_new_exposed', 0) + 1

        # check disease state timers if agent is not susceptible
        # no timer for susceptible state, agent can be exposed at any moment
        if human.disease_state is not disease.Disease_State.SUSCEPTIBLE:
            disease.check_disease_state(human, params)
        if human.disease_state is disease.Disease_State.RECOVERED:
            human.count_infected = False


    def step(self):
        '''
        Increment the sim by one step.
        '''
        log.debug(f"  - Timestamp: {self.current_time.time()}")
        log.info(f"  - tick: {self.runner.schedule.tick}")
        self.current_time  = self.current_time + timedelta(minutes=self.tick_duration)

        for human in self.context.agents():
            log.debug(f'Working with agent {human.id}')
            human.timestamp = self.current_time
            self.nhts_random_schedule(human)


        self.context.synchronize(restore_agent)

        n_agents = 0
        all_agent_types = self.context.size()
        for agents in all_agent_types:
            n_agents += all_agent_types[agents]
        log.debug(f"n_agents on this process: {n_agents}")

        # generate n random numbers for infection chance
        rand_infect = np.random.rand(n_agents)

        for idx, human in enumerate(self.context.agents()):
            self.disease_spread(human, rand_infect[idx])

        self.log_human_agents()
        end_of_tick(self)

        return

    def log_human_agents(self):
        for human in self.context.agents():
            #self.agent_logger.log_row(tick, human.id, human.uid_rank, human.current_x, human.current_y,
            #                          human.disease_state, human.location)
            #self.agent_logger.log_row(self.runner.schedule.tick, human.id, human.disease_state,
            #                          human.current_x, human.current_y, human.infected_counter, human.trip_day)
            if write_log():
                self.agent_logger.log_row(self.runner.schedule.tick, human.id, human.disease_state,
                                      human.current_x, human.current_y, human.infected_counter, human.unique_contacts, human.trip_day)

            # reset counter every tick
            human.unique_contacts = []

        if write_log():
            self.agent_logger.write()

    def start(self):
        self.runner.execute()

def setup_logging(params):
    '''
    Setup the logging for both Repast4Py as well as a normal python logger
    '''

    # Setup a command line logger.
    loglevel = 'INFO'
    pylog.basicConfig(level=loglevel, format='%(asctime)s %(relativeCreated)6d %(threadName)s ::%(levelname)-8s %(message)s')
    log = pylog.getLogger()
    log.debug('Ready to log!')
    log.debug(f'Params: {params}')
    return log

def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()

if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    seed_everything(params)
    setup_logging(params)
    run(params)
    log.info('==END==')
