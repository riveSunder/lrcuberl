import numpy as np
import time
import random
import copy 

import torch 
import torch.nn as nn
import torch.nn.functional as F

import gym

from cube.cube_env import Cube, Cube2, Cube1
from cube.policies.lstm import LSTM
from cube.utils.test_policy import test_policy

import gc

import sys

class VPG(object):

    def __init__(self, env, policy, value_fn=None, obs_dim=72, act_dim=6, hid_dim=64, epochs=100, steps_per_epoch=1000, ablate_memory=False):

        # learning params
        self.epochs = epochs
        self.steps_per_epoch=steps_per_epoch
        self.lr = 3e-4
        self.device = torch.device("cpu")
        self.discount = 0.995
        self.difficulty = 1
        self.ablate_memory = ablate_memory

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env = env
        self.policy = policy
        self.value_fn = value_fn
        self.using_lstm = True


    def train(self, exp_name, start_epoch):

        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        total_steps = 0
        results = {}

        results["epoch_rewards"] = []
        results["epoch_loss"] = []
        results["epoch_solves"] = []
        results["epoch_attempts"] = [] 
        results["total_env_interacts"] = []

        for epoch in range(self.epochs):
            self.policy.h = self.policy.init_hidden()
            self.policy.cell = self.policy.init_cell()
            self.policy.zero_grad()

            observations, rewards, dones, actions, probabilities,\
                    solves, attempts, steps = self.get_trajectory() 
            total_steps += steps
            all_advs = self.compute_advantages(rewards, dones)

            surr_loss = self.compute_surr_loss(probabilities, actions, all_advs)
            surr_loss.backward()
            mean_step_rwd = torch.sum(rewards) /(dones.shape[0])
            print("------------------------------------------------------")
            print("| epoch {:.1f} | mean step. reward {:.2f} | loss {:.3e} |".format(\
                    epoch, mean_step_rwd, surr_loss))

            optimizer.step()

            results["epoch_rewards"].append(torch.sum(rewards))
            results["epoch_loss"].append(surr_loss.detach().numpy())
            results["epoch_solves"].append(solves)
            results["epoch_attempts"].append(attempts)
            results["total_env_interacts"].append(total_steps)
            
            del(observations)
            del(rewards)
            del(dones)
            del(actions)
            del(probabilities)
            

            if(1):
                if solves / attempts > 0.667:
                    print("incrementing difficulty from {} to {}".format(\
                            self.env.difficulty, self.env.difficulty+1))
                    self.env.difficulty += 1
                elif solves / attempts < .167:
                    self.env.difficulty = np.max([1,self.env.difficulty-1])

            if epoch % 50 == 0:
                torch.save(self.policy.state_dict(), \
                        "results/scrambles2x2/{}_lstm_weights.h5"\
                        .format(exp_name))
                np.save("results/scrambles2x2/{}.npy".format(exp_name),results)

    
    def get_trajectory(self):

        step = 0
        done = True

        # instantiate tensors for trajectory buffer
        observations = torch.Tensor()
        rewards = torch.Tensor()
        probabilities = torch.Tensor()
        actions = torch.Tensor()
        dones = torch.Tensor()
        
        attempts = 0
        solves = 0

        while step < self.steps_per_epoch:

            if self.ablate_memory:
                self.policy.h = self.policy.init_hidden()
                self.policy.cell = self.policy.init_cell()

            if done:
                self.policy.h = self.policy.init_hidden()
                self.policy.cell = self.policy.init_cell()
                obs = self.env.reset()
                attempts += 1
                obs = torch.Tensor(obs.ravel()).unsqueeze(0)
                done=False

            action, probs, h = self.policy.get_actions(obs)

            obs, reward, done, info = env.step(action.detach().numpy()[0,0]) 
            if done: solves += 1

            obs = torch.tensor(obs.ravel(), dtype=torch.float).unsqueeze(0)

            # concatenate trajectory 
            observations = torch.cat([observations, obs], dim=0)
            rewards = torch.cat([rewards, torch.tensor((reward,),\
                    dtype=torch.float)], dim=0)
            probabilities = torch.cat([probabilities, probs], dim=0)
            if len(action.shape) == 1: action = action.unsqueeze(0)

            actions = torch.cat([actions, torch.tensor(\
                    action.clone().detach(), dtype=torch.float)], dim=0)
            dones = torch.cat([dones, torch.tensor((done,),dtype=torch.float)],\
                    dim=0)
            step += 1

            if (1): 
                if step > self.env.difficulty*4: done = True

        return observations, rewards, dones, actions,\
                probabilities, solves, attempts, step

    def compute_advantages(self, all_rewards, all_dones, discount=0.9, baseline=None):
        all_advs = torch.zeros_like(all_rewards)
        all_advs[-1] = all_rewards[-1]

        for ii in range(all_rewards.shape[0]-2,-1,-1):
            all_advs[ii] = all_rewards[ii] \
                    + (1-all_dones[ii]) * discount * all_advs[ii+1]

        if baseline == None:
            mean_baseline = torch.mean(all_advs)

        all_advs = all_advs - mean_baseline

        return all_advs

    def compute_surr_loss(self, all_probs, all_acts, all_advs):
        
        all_acts = all_acts.long()
        one_hot_actions = torch.zeros((all_acts.shape[0], self.act_dim))
        one_hot_actions = torch.scatter(one_hot_actions,-1, \
                all_acts, 1.0)

        surr_loss = - torch.mean(one_hot_actions * torch.log(all_probs)\
                * all_advs.unsqueeze(1))

        return surr_loss
        
def test_policy(policy, env_maker, exp_name, ablate_memory=False):
    # test meta-learning capabilities with action choice swaps in the middle of test time

    max_steps = 5000
    env = env_maker(difficulty=3, use_target=True)
    done = True
    test_results = {}

    test_results["swaps"] = []
    test_results["rewards"] = []
    test_results["step"] = []
    test_results["obs"] = []
    test_results["next_obs"] = []
    test_results["action"] = []
    test_results["dones"] = []
    step = 0 
    total_steps = 0

    print("Testing policy")
    while (total_steps+step) < max_steps:
        if ablate_memory:
            policy.h = policy.init_hidden()
            policy.cell = policy.init_cell()
        if step % 500 == 0:
            for move in (0, env.action_dim-2, 2):
                env.swap_actions(move, move+1)
            test_results["swaps"].append(1)
            print("swapping, previous 500 steps includes {} solves"\
                    .format(np.sum(test_results["dones"][-50:])))
        else:
            test_results["swaps"].append(0)
        if done:
            obs = env.reset()
            obs = torch.tensor(obs.ravel(), dtype=torch.float).unsqueeze(0)

        action, probs, h = policy.get_actions(obs)
        old_obs = obs
        obs, reward, done, info = env.step(action.detach().numpy()[0,0]) 
        obs = torch.tensor(obs.ravel(), dtype=torch.float).unsqueeze(0)

        step += 1
        if step > 30:
            total_steps += step
            step = 0 
            done = True
            
        test_results["rewards"].append(reward)
        test_results["step"].append(step)
        test_results["obs"].append(old_obs)
        test_results["next_obs"].append(obs)    
        test_results["action"].append(action)
        test_results["dones"].append(done)

    np.save("results/scrambles2x2/{}_test_policy.npy".format(exp_name), test_results)

if __name__ == "__main__":
    
    for my_seed in [0,1,2]:
        torch.manual_seed(my_seed)
        np.random.seed(my_seed)
        for scramble_actions in [False, True]:
            for ablate_memory in [False, True]:
                env = Cube2(difficulty=1, use_target=True, scramble_actions=scramble_actions)
                
                act_dim = env.action_space.n
                obs_dim = env.observation_space.shape
                obs  = env.reset()
                obs_dim = obs.shape
                input_dim = obs_dim[0] * obs_dim[1]
                hid_dim = 512 

                policy = LSTM(input_dim, act_dim, hid_dim)

                vpg = VPG(env, policy, obs_dim=input_dim, act_dim=act_dim,\
                        epochs=3000, steps_per_epoch=1000, \
                        ablate_memory=ablate_memory)
                exp_name = "scrambles2x2{}_memory{}_seed{}"\
                        .format(scramble_actions, not(ablate_memory), my_seed)
                vpg.train(exp_name, 0) 

                test_policy(vpg.policy, Cube2, exp_name, ablate_memory=ablate_memory)
