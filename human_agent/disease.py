import random as rndm
from typing import Tuple 
from enum import Enum
import ast

import datetime

import logging as pylog # Repast logger is called as "logging" 

log = pylog.getLogger(__name__)
 
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
    exposed_days = rndm.uniform(random_params[0], random_params[1])

    agent.disease_end_datetime = agent.timestamp + datetime.timedelta(days=exposed_days)
    agent.disease_state = Disease_State.EXPOSED
    return agent

def become_infected(agent, params):
    '''
    What happens when the agent becomes infected.
    '''
    random_params = ast.literal_eval(params['infection_duration'])
    infection_days = rndm.uniform(random_params[0], random_params[1])

    agent.disease_end_datetime = agent.timestamp + datetime.timedelta(days=infection_days)
    agent.disease_state = Disease_State.INFECTIOUS
    return agent

def become_recovered(agent, params):
    '''
    What happens when the agent becomes recovered.
    '''
    random_params = ast.literal_eval(params['immunity_duration'])
    immunity_days = rndm.uniform(random_params[0], random_params[1])

    agent.disease_end_datetime = agent.timestamp + datetime.timedelta(days=immunity_days)
    agent.disease_state = Disease_State.RECOVERED
    
    return agent

def become_susceptible(agent, params):
    '''
    What happens when the agent becomes susceptible.
    ''' 
    agent.disease_end_datetime = agent.timestamp
    agent.disease_state = Disease_State.SUSCEPTIBLE
    return agent


def check_infection_chance(agent, nearby_agents_list, params):
    '''
    Check if this agent has been infected by other agents.
    '''
    infection_chance = params['infection_chance']
    for other_agent in nearby_agents_list:
        if other_agent.disease_state == Disease_State.INFECTIOUS:
            if rndm.random() < infection_chance:
                return True
    return False

def check_disease_state(agent, params):
    '''
    Check the timers, check how the agents disease state progresses 
    '''
    if agent.disease_end_datetime < agent.timestamp:
        # Only do disease checks if enough time has passed since last state change: 
        if agent.disease_state == Disease_State.EXPOSED:
            # change disease state from explosed to infected
            agent = become_infected(agent, params)
            return agent
        
        if agent.disease_state == Disease_State.INFECTIOUS:
            # Check timer to see if recovered
            agent = become_recovered(agent, params)
            return agent

        if agent.disease_state == Disease_State.RECOVERED:
            # Check timer to see if immunity has lapsed
            agent = become_susceptible(agent, params)
            return agent
    else:
        return agent        