#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: valentinpiralkov
"""
from dice_game import DiceGame
game = DiceGame()
import numpy as np
from abc import ABC, abstractmethod


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game
    
    @abstractmethod
    def play(self, state):
        pass
def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()
    
    if(verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if(verbose): print(f"Starting dice: \n\t{state}\n")
    
    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1
        
        if(verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if(verbose and not game_over): print(f"Dice: \t\t{state}")

    if(verbose): print(f"\nFinal dice: {state}, score: {game.score}")
        
    return game.score

class MyAgent(DiceGameAgent):
    def __init__(self, game):
        """
        if your code does any pre-processing on the game, you can do it here
        
        e.g. you could do the value iteration algorithm here once, store the policy, 
        and then use it in the play method
        
        you can always access the game with self.game
        """
        # this calls the superclass constructor (does self.game = game)
        super().__init__(game) 
        
        # YOUR CODE HERE
        self.gamma = 0.9
        #assign initial state values to be 0
        self.values = {s: 0 for s in game.states}
        #assign initial optimal actions to be ()
        self.policy = {s: () for s in game.states}
        #calculate state values
        self.valueIteration()
        #extract policy
        self.policyIteration()
        
    def valueIteration(self):
        k = 0
        #iteration of the value calculation
        while k < 100:
            #stores the values of the last iteration
            oldValues = self.values.copy()
            #iteration through. the states of the game
            for state in self.game.states:
                #stores the values of the actions for a specific state
                actionValues = []
                #iteration through the actions
                for action in self.game.actions:
                    #store the values of the possible states after doing this current action
                    nextStateValues = []
                    #extracts the next state, reward, probability, and if the game is finished after doing the current action
                    states1, game_over, reward, probabilities = game.get_next_states(action, state)
                    #iterates over different next states and their probabilities
                    for state1, probability in zip(states1, probabilities):
                        #exclude the value of the next stte if the game is over
                        if game_over == True:
                            v = probability * (reward + self.gamma)
                        else:
                            #calculate the score of the next state -1 
                            score = self.calculateReward(state1, reward)
                            #calsulate the value of the next state
                            v = (probability * (score + self.gamma * oldValues[state1]))
                        #appent the value of the state
                        nextStateValues.append(v)
                    #combine all values of next states
                    actionValues.append(sum(nextStateValues))
                #choose the highest value for this state
                self.values[state] = max(actionValues)
            k+=1
            #break the loop if there is no significant change in the values
            if abs(sum(self.values.values()) - sum(oldValues.values())) < 310:
                break                                              

    #calculates the score of the next state minus the action reward
    def calculateReward(self, state, reward):
        states1, game_over, reward1, probabilities = game.get_next_states(self.game.actions[-1], state)
        score = reward1 + reward
        return score
    
    def policyIteration(self):
        k = 0
        #iteration of policy extraction
        while k < 10:
            #contains the values of the last iteration 
            oldPolicy = self.policy.copy()
            #true when there is no change in the policy
            converged = True
            for state in self.game.states:
                #contains values of different actions for the current state
                actionValues = {a: 0 for a in self.game.actions}
                for action in self.game.actions:
                    states1, game_over, reward, probabilities = self.game.get_next_states(action, state)
                    for state1, probability in zip(states1, probabilities):
                        if state1 == None:
                            v = (probability * (reward + self.gamma))
                        else:
                            score = self.calculateReward(state1, reward)
                            v = (probability * (reward + self.gamma * self.values[state1]))
                    actionValues[action] = v
                maxVal = 0
                optimalState = ()
                #iterate over the action values and choose the action with the highest value
                for key in actionValues:
                    if actionValues[key] > maxVal:
                        maxVal = actionValues[key]
                        optimalState = key
                #assign the optimal action to the agent policy
                self.policy[state] = optimalState
                #detect if there is change in the policy
                if optimalState != oldPolicy[state]:
                    converged = False
            k+=1
            #break if there is no change in the policy
            if converged == True :
                break   
                
    #return the optimal action from the policy for the given state 
    def play(self, state):
        """
        given a state, return the chosen action for this state
        at minimum you must support the basic rules: three six-sided fair dice
        
        if you want to support more rules, use the values inside self.game, e.g.
            the input state will be one of self.game.states
            you must return one of self.game.actions
        
        read the code in dicegame.py to learn more
        """
        # YOUR CODE HERE      
        return self.policy[state]
    
import time

total_score = 0
total_time = 0
n = 10

np.random.seed()

print("Testing basic rules.")
print()

game = DiceGame()

start_time = time.process_time()
test_agent = MyAgent(game)
total_time += time.process_time() - start_time

for i in range(n):
    start_time = time.process_time()
    score = play_game_with_agent(test_agent, game)
    total_time += time.process_time() - start_time

    print(f"Game {i} score: {score}")
    total_score += score

print()
print(f"Average score: {total_score/n}")
print(f"Total time: {total_time:.4f} seconds")

total_score = 0
total_time = 0
n = 10

print("Testing extended rules – two three-sided dice.")
print()

game = DiceGame(dice=2, sides=3)

start_time = time.process_time()
test_agent = MyAgent(game)
total_time += time.process_time() - start_time

for i in range(n):
    start_time = time.process_time()
    score = play_game_with_agent(test_agent, game)
    total_time += time.process_time() - start_time

    print(f"Game {i} score: {score}")
    total_score += score

print()
print(f"Average score: {total_score/n}")
print(f"Average time: {total_time/n:.5f} seconds")