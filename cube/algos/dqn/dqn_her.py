import numpy as np
import time
import random
import copy 

import torch 
import torch.nn as nn
import torch.nn.functional as F

import gym

from cube.cube_env import Cube, Cube2
from cube.policies.mlp import MLP
from cube.utils.test_policy import test_policy

import gc

import sys

class DQN(object):
    def __init__(self, env, obs_dim=2, act_dim=12, hid_dim=[64,64], epochs=100):
        # dimensions
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # environment 
        self.env = env
        
        # hyperparameters
        self.min_eps = torch.Tensor(np.array(0.02))
        self.eps = torch.Tensor(np.array(0.9))
        self.eps_decay = torch.Tensor(np.array(0.95))
        self.lr = 1e-3
        self.batch_size = 256
        self.steps_per_epoch = 5200 
        self.update_qt = 50 #100
        self.epochs = epochs
        self.device = torch.device("cpu")
        self.discount = 0.995
        self.difficulty = 1
        self.hid_dim = hid_dim
        
        # action-value networks
        self.q = MLP(obs_dim, act_dim, hid_dim=hid_dim, act=nn.Tanh)# act=nn.Tanh)
        try:
            fpath = "q_weights_exp{}.h5".format(exp_name) 
            print("should now restoring weights from: ", fpath) 
            self.q.load_state_dict(torch.load(fpath))
            print("restoring weights from: ", fpath) 
        except:
            pass
        self.qt = MLP(obs_dim, act_dim, hid_dim=hid_dim, act=nn.Tanh) # act=nn.Tanh)
        
        self.qt.load_state_dict(copy.deepcopy(self.q.state_dict()))

        self.q = self.q.to(self.device)
        self.qt = self.qt.to(self.device)
        for param in self.qt.parameters():
            param.requires_grad = False

    def train(self, exp_name, start_epoch):
        
        # initialize optimizer
        optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        self.rewards = []
        self.losses = []

        for epoch in range(start_epoch, start_epoch + self.epochs):
            # get episodes
            l_obs, l_act, l_rew, l_next_obs, l_done = self.get_episodes()
            # update q
            loss_mean = 0.
            batches = 0
            for batch_start in range(0,len(l_obs)-self.batch_size,\
                    self.batch_size):

                ll_obs = l_obs[batch_start:batch_start+self.batch_size]
                ll_act = l_act[batch_start:batch_start+self.batch_size]
                ll_rew = l_rew[batch_start:batch_start+self.batch_size]
                ll_next_obs = l_next_obs[batch_start:batch_start+self.batch_size]
                ll_done = l_done[batch_start:batch_start+self.batch_size]

                self.q.zero_grad()

                loss = self.compute_q_loss(ll_obs, ll_act, ll_rew, \
                        ll_next_obs, ll_done)
                loss.backward()
                optimizer.step()

                batches += 1.0
                loss_mean += loss
            self.rewards.append((torch.sum(l_rew)/\
                    (torch.Tensor(np.array(1.)) + torch.sum(l_done)))\
                    .detach().cpu().numpy())
            self.losses.append(loss_mean.detach()/batches)
            # attenuate epsilon
            self.eps = torch.max(self.min_eps, self.eps*self.eps_decay)

            print("epoch {} mean episode rewards: {}, and q loss {}".format(\
                    epoch, self.rewards[-1], self.losses[-1]))
            print("            current epsilon: {}".format(self.eps))
                    
            # maybe update qt
            if epoch % self.update_qt == 0:
                eval_trials = 100
                eval_r, eval_solves, eval_steps = self.evaluate(\
                        trials=eval_trials)

                print("evaluation at epoch {} ".format(epoch))
                print("mean r: {}, {} solves in {} steps over {} trials"\
                        .format(\
                        np.mean(eval_r), eval_solves, \
                        np.sum(eval_steps), eval_trials))
                if eval_solves/ eval_trials < 0.15 and self.difficulty >= 1:
                    self.difficulty -= 1
                    print("decrementing difficulty to ".format(\
                            self.difficulty))
                while eval_solves/eval_trials >= 0.75: 
                    self.difficulty += 1
                    print("incrementing difficulty to {}"\
                            .format(self.difficulty))
                    eval_r, eval_solves, eval_steps = self.evaluate(\
                            trials=eval_trials)

                    print("evaluation at epoch {} ".format(epoch))
                    print("mean r: {}, {} solves in {} steps over {} trials"\
                        .format(\
                        np.mean(eval_r), eval_solves, \
                        np.sum(eval_steps), eval_trials))

                self.qt.load_state_dict(copy.deepcopy(self.q.state_dict()))
                for param in self.qt.parameters():
                    param.requires_grad = False
            if epoch % 100 == 0:
                torch.save(self.q.state_dict(), "results/q_weights_exp{}_start{}_pt2.h5"\
                        .format(exp_name, start_epoch))

                
                np.save("./results/exp{}_rewards_start{}.npy"\
                        .format(exp_name, start_epoch), np.array(self.rewards))

        fpath = "q_weights_exp{}.h5".format(exp_name)
        print("saving weights to ", fpath)
        torch.save(self.q.state_dict(),fpath)

        results = test_policy(self.env, self.obs_dim, self.act_dim,\
            self.hid_dim, fpath=fpath)
        
        np.save("./results/{}/test_{}_epoch{}.npy".format(exp_name,exp_name,\
                epoch),results)


    def evaluate(self, trials, difficulty=None):
    
        total_rewards = []
        total_moves = []
        solves = 0
        max_moves = 540 
        render = False #True
        for trial in range(trials):
            done = False
            cube_moves = 0
            obs = self.env.reset(difficulty=self.difficulty)
            total_reward = 0.0
            while not done:
                if render: 
                    self.env.render()
                    time.sleep(0.05)
                obs = torch.Tensor(obs.ravel()).unsqueeze(0)
                q_values = self.q(obs)
                act = torch.argmax(q_values,dim=-1)
                # detach action to send it to the environment
                action = act.detach().numpy()[0]
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                cube_moves += 1

                if done:
                    solves += 1
                elif cube_moves > max_moves:
                    #give up
                    done = True
            total_moves.append(cube_moves)
            total_rewards.append(total_reward)


        return total_rewards, solves, total_moves

    def compute_q_loss(self, l_obs, l_act, l_rew, l_next_obs, l_done,\
            double=True):

        with torch.no_grad():
            qt = self.qt.forward(l_next_obs)
            if double:
                qtq = self.q.forward(l_next_obs)
                qt_max = torch.gather(qt, -1,\
                        torch.argmax(qtq, dim=-1).unsqueeze(-1))
            else:
                qt_max = torch.gather(qt, -1, \
                        torch.argmax(qt, dim=-1).unsqueeze(-1))

            yj = l_rew + ((1-l_done) * self.discount * qt_max)

        l_act = l_act.long()
        q_av = self.q.forward(l_obs)
        q_act = torch.gather(q_av, -1, l_act)

        loss =  torch.mean(torch.pow(yj-q_act,2))

        return loss

    def get_episodes(self,steps=None):
        
        if steps == None:
            steps = self.steps_per_epoch

        l_obs = torch.Tensor()
        l_rew = torch.Tensor()
        l_act = torch.Tensor()
        l_next_obs = torch.Tensor()
        l_done = torch.Tensor()

        max_moves = int(self.difficulty*3)

        done = True
        with torch.no_grad():
            for step in range(steps):
                if done:
                    cube_moves = 0
                    obs = self.env.reset(difficulty=self.difficulty)
                    obs = torch.Tensor(obs.ravel()).unsqueeze(0)
                    done = False

                
                if torch.rand(1) < self.eps:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.q(obs)
                    act = torch.argmax(q_values,dim=-1)
                    # detach action to send it to the environment
                    action = act.detach().numpy()[0]

                prev_obs = obs
                obs, reward, done, info = self.env.step(action)
                obs = torch.Tensor(obs.ravel()).unsqueeze(0)

                cube_moves += 1

                # concatenate data from current step to buffers
                l_obs = torch.cat([l_obs, prev_obs], dim=0)
                l_rew = torch.cat([l_rew, torch.Tensor(np.array(1.* reward))\
                        .reshape(1,1)], dim=0)
                l_act = torch.cat([l_act, torch.Tensor(np.array([action]))\
                        .reshape(1,1)], dim=0)
                l_done = torch.cat([l_done, torch.Tensor(np.array(1.0*done))\
                        .reshape(1,1)], dim=0)
                l_next_obs = torch.cat([l_next_obs, obs], dim=0)

                if cube_moves > max_moves and not done:
                    l_done[-1] = 1.0 
                    # implement HER
                    # Whatever the current state is, that's what we meant to do.
                    # Copy the experience
                    temp_l_obs = l_obs[-cube_moves:].clone().detach()
                    temp_l_rew = l_rew[-cube_moves:].clone().detach()
                    temp_l_act = l_act[-cube_moves:].clone().detach()
                    temp_l_done = l_done[-cube_moves:].clone().detach()
                    temp_l_next_obs = l_next_obs[-cube_moves:].clone().detach()
    
                    # Apply rose-tinted glasses
                    target_half = int(obs.shape[-1]/2)
                    rose_target = obs[:, target_half:]
                    for replay_step in range(0,cube_moves):
                        temp_l_next_obs[replay_step, target_half:] = rose_target
                        temp_l_obs[replay_step, target_half:] = rose_target

                    temp_l_done[-1] = 1.0
                    temp_l_rew[-1] = 26. 
                    
                    done = True

                    l_obs = torch.cat([l_obs, temp_l_obs], dim=0)
                    l_rew = torch.cat([l_rew, temp_l_rew], dim=0)
                    l_act = torch.cat([l_act, temp_l_act], dim=0)
                    l_done = torch.cat([l_done, temp_l_done], dim=0)
                    l_next_obs = torch.cat([l_next_obs, temp_l_next_obs], dim=0)
                
        return l_obs, l_act, l_rew, l_next_obs, l_done

if __name__ == "__main__":

    
    args = sys.argv[1:]
    print(args)
    time.sleep(3)
    epochs = 1
    start_epoch = 0
    exp_name = "default"
    for cc in range(len(args)):
        if args[cc] == "--epochs":
            epochs = int(args[cc+1])
        elif args[cc] == "--exp_name":
            exp_name = str(args[cc+1])
        elif args[cc] == "--start":
            start_epoch = int(args[cc+1])

    env = Cube2(use_target=True)
    #env = gym.make("CartPole-v0")

    torch.set_num_threads(1)

    obs_dim = env.observation_space.call()
    obs_dim = obs_dim[0]*obs_dim[1]
    act_dim = env.action_dim #Cube.action_space.sample().shape
    #obs_dim = 4
    #act_dim = 2
    dqn = DQN(env, obs_dim=obs_dim, act_dim=act_dim, hid_dim=[64,64,64], epochs=epochs)

    dqn.train(exp_name, start_epoch)
