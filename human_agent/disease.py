import random as rndm
from typing import Tuple 
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
import ast

import datetime
import numpy as np

import logging as pylog # Repast logger is called as "logging" 

log = pylog.getLogger(__name__)

'''
This hold the disease functions for the a S.I.R model.

Stuck in a state machine:
  - Susceptible
    - Move close to other infected
    - Become Infectious
    - Randomly disease_end_time to some point in future
  - Infectious 
    - Wait until current time > disease_end_time 
    - Become Recovered
    - Randomly disease_end_time to some point in future
  - Recovered 
    - Wait until current time > disease_end_time
    - Become Susceptible again
'''
 
class Disease_State(str, Enum):
  SUSCEPTIBLE = "Susceptible"
  EXPOSED = "Exposed"
  INFECTIOUS = "Infectious"
  RECOVERED = "Recovered"

def become_exposed(agent, params):
    '''
    What happens when the agent becomes exposed.
    '''
    random_params = ast.literal_eval(params['exposed_duration'])
    #exposed_days = rndm.gauss(random_params[0], random_params[1])
    exposed_days = rndm.uniform(random_params[0], random_params[1])

    agent.disease_end_datetime = agent.timestamp + datetime.timedelta(days=exposed_days)
    agent.disease_state = Disease_State.EXPOSED
    return agent

def become_infected(agent, params):
    '''
    What happens when the agent becomes infected.
    '''
    random_params = ast.literal_eval(params['infection_duration'])
    #infection_days = rndm.gauss(random_params[0], random_params[1])
    infection_days = rndm.uniform(random_params[0], random_params[1])

    agent.disease_end_datetime = agent.timestamp + datetime.timedelta(days=infection_days)
    agent.disease_state = Disease_State.INFECTIOUS
    return agent

def become_recovered(agent, params):
    '''
    What happens when the agent becomes recovered.
    '''
    log.debug(f'    - Recovered! {agent.id}')

    random_params = ast.literal_eval(params['immunity_duration'])
    #immunity_days = rndm.gauss(random_params[0], random_params[1])
    immunity_days = rndm.uniform(random_params[0], random_params[1])

    agent.disease_end_datetime = agent.timestamp + datetime.timedelta(days=immunity_days)
    agent.disease_state = Disease_State.RECOVERED
    
    return agent

def become_susceptible(agent, params):
    '''
    What happens when the agent becomes susceptible.
    ''' 
    log.debug(f'    - Susceptible! {agent.id}')

    agent.disease_end_datetime = agent.timestamp
    agent.disease_state = Disease_State.SUSCEPTIBLE
    return agent



def check_disease_state(agent, params):
    '''
    Check the timers, check how the agents disease state progresses 
    '''
    log.debug(f'Checking disease status for agent {agent.id}. Current state = {str(agent.disease_state)}')
    if agent.disease_end_datetime < agent.timestamp:
        # Only do disease checks if enough time has passed since last state change: 
        log.debug(f'enough time has passed!')        
        if agent.disease_state == Disease_State.EXPOSED:
            # change disease state from explosed to infected
            agent = become_infected(agent, params)
            return agent
        
        if agent.disease_state == Disease_State.INFECTIOUS:
            # Check timer to see if recovered
            log.debug(f'Agent {agent.id} Recovering. Current state = {str(agent.disease_state)}')
            agent = become_recovered(agent, params)
            return agent

        if agent.disease_state == Disease_State.RECOVERED:
            # Check timer to see if immunity has lapsed
            agent = become_susceptible(agent, params)
            return agent
    else:
        return agent        
