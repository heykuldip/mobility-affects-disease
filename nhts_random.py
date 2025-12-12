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

# model = None
agent_tuple_cache = {}
log = pylog.getLogger(__name__)

def restore_agent(agent_tuple: Tuple):
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

        coords = [1578015, 43715, 1893069, 44225]
        self.building_ids = list(self.dbuildings.keys())

        # model parameters
        self.infection_chance = params['infection_chance']
        
        # create the context to hold the agents and manage cross process
        # synchronization
        self.context = ctx.SharedContext(comm)

        box = space.BoundingBox(coords[0], coords[1],
                                coords[2], coords[3], 0, 0)
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple, buffer_size=10, comm=comm)
        self.context.add_projection(self.grid)

        log_file = params['agent_log_file']+'.csv'
        self.agent_logger = logging.TabularLogger(comm, log_file,
                                                  ['tick', 'agent_id','disease_state',
                                                   'x', 'y','n_infected', 'agents_contacted'])

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
    
        rng = np.random.default_rng(seed=42)
        init_infected = rng.choice(n_agents, int(n_agents * params['infectious_start_rate']))
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
                    if census_tract != 51059440800:
                        init_infected[human_id] = False
                        infectious_count = infectious_count-1
                    self.add_human_agent_v2(human_id, nhts_id, trip_chain, census_tract,
                                            agent_type, init_infected[human_id])
        
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

    def start(self):
        self.runner.execute()

    def end_sim(self):
        '''
        Clean up and log data.
        '''
        log.info(f'Cleaning up simulation...')
        self.end_start_timestamp = datetime.now()
        self.agent_logger.close()

        return
    
    def add_human_agent_v2(self, human_id, nhts_id, trip_chain, census_tract, agent_type, init_infected):
        '''
        Add agent to the sim located at Point(x,y) based on the home id
        '''
        #building_to_go = self.dframe[self.dframe['bldg_id'] == home_id]
        home_id = trip_chain[-1]
        start_minutes = self.dwell_time[nhts_id][0]
        building_to_go = self.dbuildings[home_id]
        pt = dpt(building_to_go["pos"][0],
                building_to_go["pos"][1])
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
            if next_trip == 'restaurants':
                next_trip = np.random.choice(self.restaurant_ids)
            elif next_trip == 'recreation':
                next_trip = np.random.choice(self.recreation_ids)
            elif next_trip == 'errands':
                next_trip = np.random.choice(self.errand_ids)
            elif next_trip == 'hospitals':
                next_trip = np.random.choice(self.hospital_ids) 

            dwell_time = self.dwell_time[human.nhts_id][human.trip_num+1]
            human.trip_end_datetime = self.current_time + timedelta(minutes=dwell_time)
            self.go_to_place(human, next_trip)
            human.trip_num += 1
            if human.trip_num == len(human.trip_chain):
                human.trip_num = 0                
    
    def go_to_place(self, human, place_id):
        next_bldg = self.dbuildings[place_id]
        x = next_bldg["pos"][0]
        y = next_bldg["pos"][1]
        human.move_coords(self, x, y)

    def disease_spread(self, human):
        # find and infect another susceptible agent at same place if human is infected
        pt = model.grid.get_location(human)
        n_at_place = model.grid.get_num_agents(pt)
        n_potential_agents = round(np.log2(n_at_place))

        agents_counter = 0
        for other_agent in self.grid.get_agents(pt):
            if agents_counter >= n_potential_agents:
                break
            if other_agent.id != human.id:
                human.unique_contacts.append(other_agent.id)
                agents_counter += 1
                if human.disease_state is disease.Disease_State.INFECTIOUS:
                    if np.random.rand() < self.infection_chance:
                        if human.count_infected == True:
                            human.infected_counter += 1
                        # let an infectious agent infect susceptible with a given probability
                        if other_agent.disease_state == disease.Disease_State.SUSCEPTIBLE:
                            disease.become_exposed(other_agent, params)

        # check disease state timers if agent is not susceptible
        if human.disease_state is not disease.Disease_State.SUSCEPTIBLE:
            disease.check_disease_state(human, params)
        if human.disease_state is disease.Disease_State.RECOVERED:
            human.count_infected = False

    def step(self):
        '''
        Increment the sim by one step.
        '''
        log.info(f"  - tick: {self.runner.schedule.tick}")
        self.current_time  = self.current_time + timedelta(minutes=self.tick_duration)

        for human in self.context.agents():
            human.timestamp = self.current_time                  
            self.nhts_random_schedule(human) 

        self.context.synchronize(restore_agent) 
        for human in self.context.agents():
            self.disease_spread(human)
        self.log_human_agents()

        return
    
    def log_human_agents(self):
        for human in self.context.agents():
            self.agent_logger.log_row(self.runner.schedule.tick, human.id, human.disease_state,
                                      human.current_x, human.current_y, human.infected_counter, human.unique_contacts)            
            human.unique_contacts = []
        self.agent_logger.write()

    def start(self):
        self.runner.execute()

def setup_logging(params):
    '''
    Setup the logging for both Repast4Py as well as a normal python logger
    '''
    loglevel = 'INFO'
    pylog.basicConfig(level=loglevel, format='%(asctime)s %(relativeCreated)6d %(threadName)s ::%(levelname)-8s %(message)s')
    log = pylog.getLogger()
    return log

def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start()

if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    setup_logging(params)
    run(params)
    log.info('==END==')