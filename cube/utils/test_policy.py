import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cube.cube_env import Cube
from cube.policies.mlp import MLP

import argparse

def test_policy(env, obs_dim, act_dim, hid_dim, act=nn.Tanh, fpath=None):

    q = MLP(obs_dim, act_dim, hid_dim=hid_dim, act=nn.Tanh) # act=nn.Tanh)

    if fpath is not None:
        q.load_state_dict(torch.load(fpath))                                                 

    solves = 0
    max_moves = 1080
    trials = 100
    
    results = {"difficulty": [],\
            "solves": [],\
            "trials": [],\
            "solve_percent": [],\
            "solve_length": [],
            "max_steps": max_moves
                    }

    for difficulty in range(26):
        env.set_difficulty(difficulty)
        solves = 0
        for trial in range(trials):
            done = False
            cube_moves = 0
            obs = env.reset()
            total_moves = []

            while not done:

                obs = torch.Tensor(obs.ravel()).unsqueeze(0)
                q_values = q(obs)
                act = torch.argmax(q_values,dim=-1)
                # detach action to send it to the environment
                action = act.detach().numpy()[0]
                obs, reward, done, info = env.step(action)
                cube_moves += 1

                if done:
                    solves += 1
                elif cube_moves > max_moves:
                    #give up
                    done = True
            
            total_moves.append(cube_moves)
        if solves < 1:
            break

        print("{} solves of {} at difficulty {}, max moves {}".format(solves, trials, difficulty, max_moves))

        results["difficulty"].extend([env.difficulty])
        results["solves"].extend([solves])
        results["trials"].extend([trials])
        results["solve_percent"].extend([solves])
        results["solve_length"].append(total_moves)


    return results

if __name__ == "__main__":
    

    env = Cube()

    obs_dim = env.observation_space.call()
    obs_dim = obs_dim[0]*obs_dim[1]
    act_dim = env.action_dim #Cube.action_space.sample().shape
    hid_dim = [256,128,64]
    fpath = "q_weights_expmlp256x128x64.h5"

    results = test_policy(env, obs_dim, act_dim, hid_dim, act=nn.Tanh, fpath=fpath)

    np.save("./results/mlp256x128x64test_20000.npy",results)
