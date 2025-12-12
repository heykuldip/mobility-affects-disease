from repast4py import core, random
from repast4py.space import DiscretePoint as dpt
from typing import Tuple 
import numpy as np
from dataclasses import dataclass, field
import datetime
import random as rndm
import ast
import sys
import os

import logging as pylog # Repast logger is called as "logging"

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from . import disease

log = pylog.getLogger(__name__)

@dataclass
class Human_Config(object):
    '''
    Some basic configuration for a human agent. 
    '''  
    current_x: float = 0.0 
    current_y: float = 0.0
    home_id: float = 0.0 
    trip_chain: list = field(default_factory=list)
    trip_num: int = 0
    nhts_id: int = 0
    census_tract: int = 0
    infected_counter: int = 0
    count_infected: bool = True
    unique_contacts: list = field(default_factory=list)
    disease_state: disease.Disease_State = field(default = disease.Disease_State.SUSCEPTIBLE) 
    disease_end_datetime: datetime = datetime.datetime(2024, 1, 1)
    timestamp: datetime.datetime = datetime.datetime.fromisoformat('1970-01-01')
    trip_end_datetime: datetime = datetime.datetime(2024, 1, 1)

    @classmethod
    def rand_factory(cls, x, y, home_id, nhts_id, trip_chain, census_tract, start_minutes, init_infected, params):
        '''
        Returns a randomised human agent config
        '''
        initial_x = x
        initial_y = y
        trip_chain = trip_chain
        trip_num = 0
        nhts_id = nhts_id
        census_tract = census_tract
        infected_counter = 0
        count_infected = True

        home_id = home_id

        # initial disease state
        initial_timestamp = datetime.datetime.fromisoformat(params['start_time'])
        if init_infected:
            random_params = ast.literal_eval(params['infection_duration'])
            infection_days = rndm.uniform(random_params[0], random_params[1]) 
            disease_end_datetime = initial_timestamp + datetime.timedelta(days=infection_days)
            disease_state = disease.Disease_State.INFECTIOUS
        else: 
            disease_state = disease.Disease_State.SUSCEPTIBLE
            disease_end_datetime = initial_timestamp
        trip_end_datetime = initial_timestamp + datetime.timedelta(minutes=start_minutes) 
        return cls(
            current_x = initial_x,
            current_y = initial_y,
            home_id = home_id,
            trip_chain = trip_chain,
            trip_num = trip_num,
            nhts_id = nhts_id,
            census_tract = census_tract,
            disease_state = disease_state,
            disease_end_datetime = disease_end_datetime,
            timestamp = initial_timestamp,
            trip_end_datetime = trip_end_datetime,
            infected_counter = infected_counter,
            count_infected = count_infected,
            unique_contacts = list(),
)    

class Human(core.Agent):

    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, agent_type: int, rank: int):
        super().__init__(id=local_id, type=agent_type, rank=rank)

    def set_cfg(self, agent_cfg):
        '''
        Set/Get the config variables to/from a dict. This is to avoid the agent restore 
        function having a bunch of tuples.
        '''
        self.current_x = agent_cfg.current_x
        self.current_y = agent_cfg.current_y
        self.home_id = agent_cfg.home_id
        self.trip_chain = agent_cfg.trip_chain
        self.trip_num = agent_cfg.trip_num
        self.nhts_id = agent_cfg.nhts_id
        self.census_tract = agent_cfg.census_tract
        self.disease_state = agent_cfg.disease_state
        self.disease_end_datetime = agent_cfg.disease_end_datetime
        self.timestamp = agent_cfg.timestamp
        self.trip_end_datetime = agent_cfg.trip_end_datetime
        self.infected_counter = agent_cfg.infected_counter
        self.count_infected = agent_cfg.count_infected
        self.unique_contacts = agent_cfg.unique_contacts

    def get_cfg(self):
        agent_cfg = Human_Config()
        agent_cfg.current_x = self.current_x 
        agent_cfg.current_y = self.current_y 
        agent_cfg.home_id = self.home_id
        agent_cfg.trip_chain = self.trip_chain
        agent_cfg.trip_num = self.trip_num
        agent_cfg.nhts_id = self.nhts_id
        agent_cfg.census_tract = self.census_tract
        agent_cfg.disease_state = self.disease_state
        agent_cfg.disease_end_datetime = self.disease_end_datetime 
        agent_cfg.timestamp = self.timestamp
        agent_cfg.trip_end_datetime = self.trip_end_datetime
        agent_cfg.infected_counter = self.infected_counter
        agent_cfg.count_infected = self.count_infected
        agent_cfg.unique_contacts = self.unique_contacts
        return agent_cfg             

    def save(self) -> Tuple:
        """Saves the state of this agent as a Tuple.

        Returns:
            The saved state of this agent.
        """
        cfg = self.get_cfg()
        return (self.uid,
                cfg)

    def move_coords(self, model, x, y):
        pt = model.grid.move(self, dpt(x,y))
        self.current_x, self.current_y = pt.x, pt.y