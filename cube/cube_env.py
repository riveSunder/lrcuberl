import numpy as np
import time
from gym import spaces

class Cube1():
    def __init__(self, difficulty=4, obs_mode="mlp", use_target=False, scramble_actions=False):
        self.action_dim = 4

        if use_target:
            self.obs_dim = (6, 12)
        else:
            self.obs_dim = (6, 6)

        self.difficulty = difficulty
        self.scramble_actions = scramble_actions
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(0, 1, shape=self.obs_dim, dtype=np.int16)

        self.obs_mode = obs_mode 
        self.use_target = use_target


        self.action_dict = {0: self.U, 1: self.u,
                2: self.L, 3: self.l,
                4: self.F, 5: self.f,
                6: self.R, 7: self.r,
                8: self.B, 9: self.b,
                10: self.D, 11: self.d}

        _ = self.reset()

    def reset(self, difficulty=None):

        self.moves = 0

        if difficulty is not None:
            self.difficulty = difficulty

        self.target = np.zeros((1,1,6))
        for face in range(6):
            self.target[..., face] = face
            
        self.cube = np.zeros((1,1,6))
        for face in range(6):
            self.cube[...,face] = face

        # Scramble cube
        for cc in range(self.difficulty):
            self.step(self.get_random_action())

        if self.scramble_actions:
            for aa in range(2):
                swap_move = 2 * np.random.randint(0,int(self.action_dim/2))
                self.swap_actions(swap_move, swap_move+1)

        return self.categorical_cube()

    def swap_actions(self, action_a, action_b):
       

        move_a = self.action_dict[action_a]
        move_b = self.action_dict[action_b]

        self.action_dict[action_b] = move_a
        self.action_dict[action_a] = move_b


    def step(self, action):
        """
        available actions are F, R, L, U, D, B 
        and their reverses
        """

        self.action_dict[action]()

        info = {}
        done = self.is_solved()

        self.moves += 1
        if done:
            reward = 26.0 
        else:
            reward = 0.0

        if self.obs_mode == "mlp":
            observation = self.categorical_cube()

        return observation, reward, done, info

    def set_difficulty(self, difficulty, verbose=True):
        if difficulty == self.difficulty:
            pass
        else:
            if verbose: print("changing difficulty from {} to {}".format(self.difficulty, difficulty))
            self.difficulty = difficulty

    def categorical_cube(self):

        categorical_cube = np.zeros((6,6))

        flat_cube = np.copy(self.cube.ravel())

        for idx in range(len(flat_cube)):
            categorical_cube[idx,int(flat_cube[idx])] = 1.

        if self.use_target:
            categorical_target = np.zeros((6,6))
            flat_target = np.copy(self.target.ravel())

            # convert to one-hot embedding
            for idx in range(len(flat_target)):
                categorical_target[idx, int(flat_target[idx])] = 1.

            categorical_cube = np.append(categorical_cube,\
                    categorical_target,\
                    axis=0)
        
        return categorical_cube

    def is_solved(self):

        for face in range(6):
            for ii in range(self.cube.shape[0]):
                for jj in range(self.cube.shape[1]):
                    if self.cube[ii,jj,face] != self.target[ii,jj,face]:
                        return False

        return True

    def get_random_action(self):
        return np.random.randint(self.action_dim)
        
    
    def display_cube(self, target=False):
    
        cube = self.target if target else self.cube

        scoot = 7
        print('\n')
        print(' ' * scoot + str(cube[:,:,0]) + ' ' * scoot) # Up
        
        print(str(cube[:,:,1])+\
                str(cube[:,:,2])+\
                str(cube[:,:,3])+\
                str(cube[:,:,4]))

        print(' ' * scoot + str(cube[:,:,5]) + ' ' * scoot) # Up

    def F(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,2] = self.cube[:,:,2]
        new_cube[:,:,0] = self.cube[:,:,1]
        new_cube[:,:,3] = self.cube[:,:,0] 
        new_cube[:,:,5] = self.cube[:,:,3] 
        new_cube[:,:,1] = self.cube[:,:,5]

        self.cube = new_cube

    def f(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,2] = self.cube[:,:,2]
        new_cube[:,:,1] = self.cube[:,:,0]
        new_cube[:,:,0] = self.cube[:,:,3] 
        new_cube[:,:,3] = self.cube[:,:,5] 
        new_cube[:,:,5] = self.cube[:,:,1]
    
        self.cube = new_cube

    def U(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,0] = self.cube[:,:,0]

        new_cube[:,:,1] = self.cube[:,:,2]
        new_cube[:,:,2] = self.cube[:,:,3] 
        new_cube[:,:,3] = self.cube[:,:,4] 
        new_cube[:,:,4] = self.cube[:,:,1]

        self.cube = new_cube

    def u(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,0] = self.cube[:,:,0]

        new_cube[:,:,2] = self.cube[:,:,1]
        new_cube[:,:,3] = self.cube[:,:,2] 
        new_cube[:,:,4] = self.cube[:,:,3] 
        new_cube[:,:,1] = self.cube[:,:,4]

        self.cube = new_cube

    def D(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,5] = self.cube[:,:,5]

        new_cube[:,:,1] = self.cube[:,:,2]
        new_cube[:,:,2] = self.cube[:,:,3] 
        new_cube[:,:,3] = self.cube[:,:,4] 
        new_cube[:,:,4] = self.cube[:,:,1]

        self.cube = new_cube

    def d(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,5] = self.cube[:,:,5]

        new_cube[:,:,2] = self.cube[:,:,1]
        new_cube[:,:,3] = self.cube[:,:,2] 
        new_cube[:,:,4] = self.cube[:,:,3] 
        new_cube[:,:,1] = self.cube[:,:,4]

        self.cube = new_cube

    def B(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,4] = self.cube[:,:,4]

        new_cube[:,:,3] = self.cube[:,:,5]
        new_cube[:,:,5] = self.cube[:,:,1] 
        new_cube[:,:,1] = self.cube[:,:,0] 
        new_cube[:,:,0] = self.cube[:,:,3]

        self.cube = new_cube

    def b(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,4] = self.cube[:,:,4]

        new_cube[:,:,5] = self.cube[:,:,3]
        new_cube[:,:,1] = self.cube[:,:,5] 
        new_cube[:,:,0] = self.cube[:,:,1] 
        new_cube[:,:,3] = self.cube[:,:,0]

        self.cube = new_cube

    def L(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,1] = self.cube[:,:,1]

        new_cube[:,:,5] = self.cube[:,:,2]
        new_cube[:,:,2] = self.cube[:,:,0] 
        new_cube[:,:,0] = self.cube[:,:,4] 
        new_cube[:,:,4] = self.cube[:,:,5]

        self.cube = new_cube

    def l(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,1] = self.cube[:,:,1]

        new_cube[:,:,2] = self.cube[:,:,5]
        new_cube[:,:,0] = self.cube[:,:,2] 
        new_cube[:,:,4] = self.cube[:,:,0] 
        new_cube[:,:,5] = self.cube[:,:,4]

        self.cube = new_cube

    def R(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,3] = self.cube[:,:,3]

        new_cube[:,:,2] = self.cube[:,:,5]
        new_cube[:,:,0] = self.cube[:,:,2] 
        new_cube[:,:,4] = self.cube[:,:,0] 
        new_cube[:,:,5] = self.cube[:,:,4]

        self.cube = new_cube

    def r(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,3] = self.cube[:,:,3]

        new_cube[:,:,5] = self.cube[:,:,2]
        new_cube[:,:,2] = self.cube[:,:,0] 
        new_cube[:,:,0] = self.cube[:,:,4] 
        new_cube[:,:,4] = self.cube[:,:,5]

        self.cube = new_cube

class Cube2():
    class ActionSpace():
        def __init__(self, act_dim):
            self.act_dim = act_dim
        def sample(self):
            return np.random.randint(self.act_dim)
                
    class ObservationSpace():
        def __init__(self, obs_dim):
            self.obs_dim = obs_dim
        def call(self):
            return self.obs_dim


    def __init__(self, difficulty=10, obs_mode="mlp", use_target=False):
        self.action_dim = 12
        if use_target:
            self.obs_dim = (24, 12)
        else:
            self.obs_dim = (24, 6)

        self.difficulty = difficulty
        self.action_space = self.ActionSpace(self.action_dim)
        self.observation_space = self.ObservationSpace(self.obs_dim)

        self.obs_mode = obs_mode 
        self.use_target = use_target

        _ = self.reset()


    def reset(self, difficulty=None):

        self.cube = np.zeros((2,2,6))
        self.moves = 0

        if difficulty is not None:
            self.difficulty = difficulty

        if self.use_target:
            self.target = np.zeros((2,2,6))
            for face in range(6):
                self.target[..., face] = face
            
        for face in range(6):
            self.cube[...,face] = face

        # Scramble cube
        for cc in range(self.difficulty):
            self.step(self.get_random_action())

        return self.categorical_cube()

    def step(self, action):
        """
        available actions are F, R, L, U, D, B 
        and their reverses
        """
        if action == 0:
            self.U()
        elif action == 1:
            self.L()
        elif action == 2:
            self.F()
        elif action == 3:
            self.R()
        elif action == 4:
            self.B()
        elif action == 5:
            self.D()
        elif action == 6:
            self.u()
        elif action == 7:
            self.l()
        elif action == 8:
            self.f()
        elif action == 9:
            self.r()
        elif action == 10:
            self.b()
        elif action == 11:
            self.d()
        
        info = {}
        done = self.is_solved()

        self.moves += 1
        if done:
            reward = 26.0 
        else:
            reward = 0.0

        if self.obs_mode == "mlp":
            observation = self.categorical_cube()

        return observation, reward, done, info

    def set_difficulty(self, difficulty, verbose=True):
        if difficulty == self.difficulty:
            pass
        else:
            if verbose: print("changing difficulty from {} to {}".format(self.difficulty, difficulty))
            self.difficulty = difficulty

    def categorical_cube(self):

        categorical_cube = np.zeros((24,6))

        flat_cube = np.copy(self.cube.ravel())

        for idx in range(len(flat_cube)):
            categorical_cube[idx,int(flat_cube[idx])] = 1.

        if self.use_target:
            categorical_target = np.zeros((24,6))
            flat_target = np.copy(self.target.ravel())

            # convert to one-hot embedding
            for idx in range(len(flat_target)):
                categorical_target[idx, int(flat_target[idx])] = 1.

            categorical_cube = np.append(categorical_cube,\
                    categorical_target,\
                    axis=0)
        
        return categorical_cube

    def is_solved(self):

        for face in range(6):
            solve = self.cube[1,1,face] * np.ones_like(self.cube[:,:,face])
            if np.max(np.abs(solve - self.cube[:,:,face])) > 0:
                return False
            

        return True

    def get_random_action(self):
        return np.random.randint(self.action_dim)
        
    
    def display_cube(self, target=False):
    
        cube = self.target if target else self.cube

        scoot = 7
        print('\n')
        for row in range(2):
            print(' ' * scoot + str(cube[row,:,0]) + ' ' * scoot) # Up
        
        for long_row in range(2):
            print(str(cube[long_row,:,1])+\
            str(cube[long_row,:,2])+\
            str(cube[long_row,:,3])+\
            str(cube[long_row,:,4]))

        for row in range(2):
            print(' ' * scoot + str(cube[row,:,5]) + ' ' * scoot) # Up


    def F(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,2] = self.cube[:,:,2].T

        for col in [0,1]:
            new_cube[:,col,2] = self.cube[:,:,2].T[:,1-col]

        temp = self.cube[:,1,1].tolist()
        temp.reverse()
        new_cube[1,:,0] = np.array(temp)

        temp = self.cube[1,:,0].tolist()
        new_cube[:,0,3] = np.array(temp)

        temp = self.cube[:,0,3].tolist()
        temp.reverse()
        new_cube[0,:,5] = np.array(temp)
        
        temp = self.cube[0,:,5].tolist()
        new_cube[:,1,1] = np.array(temp)
    
        self.cube = new_cube

    def f(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,2] = self.cube[:,:,2].T
        for row in [0,1]:
            new_cube[row,:,2] = self.cube[:,:,2].T[1-row,:]

        temp = self.cube[1,:,0].tolist()
        temp.reverse()
        new_cube[:,1,1] = np.array(temp)

        temp = self.cube[:,0,3].tolist()
        new_cube[1,:,0] = np.array(temp)

        temp = self.cube[0,:,5].tolist()
        temp.reverse()
        new_cube[:,0,3] = np.array(temp)
        
        temp = self.cube[:,1,1].tolist()
        new_cube[0,:,5] = np.array(temp)
    
        self.cube = new_cube

    def B(self):

        new_cube = np.copy(self.cube)

        new_cube[:,:,4] = self.cube[:,:,4].T
        for col in [0,1]:
            new_cube[:,col,4] = self.cube[:,:,4].T[:,1-col]

        temp = self.cube[1,:,5].tolist()
        temp.reverse()
        new_cube[:,1,3] = np.array(temp)

        temp = self.cube[:,1,3].tolist()
        new_cube[0,:,0] = np.array(temp)
         
        temp = self.cube[0,:,0].tolist()
        temp.reverse()
        new_cube[:,0,1] = np.array(temp)
    
        temp = self.cube[:,0,1].tolist()
        new_cube[1,:,5] = np.array(temp)

        self.cube = new_cube

    def b(self):

        new_cube = np.copy(self.cube)

        new_cube[:,:,4] = self.cube[:,:,4].T
        for row in [0,1]:
            new_cube[row,:,4] = self.cube[:,:,4].T[1-row,:]

        temp = self.cube[:,1,3].tolist()
        temp.reverse()
        new_cube[1,:,5] = np.array(temp)

        temp = self.cube[0,:,0].tolist()
        new_cube[:,1,3] = np.array(temp)
         
        temp = self.cube[:,0,1].tolist()
        temp.reverse()
        new_cube[0,:,0] = np.array(temp)
    
        temp = self.cube[1,:,5].tolist()
        new_cube[:,0,1] = np.array(temp)

        self.cube = new_cube

    def D(self):

        new_cube = np.copy(self.cube)

        new_cube[:,:,5] = self.cube[:,:,5].T
        for col in [0,1]:
            new_cube[:,col,5] = self.cube[:,:,5].T[:,1-col]

        temp = self.cube[1,:,2].tolist()
        new_cube[1,:,3] = np.array(temp)

        temp = self.cube[1,:,3].tolist()
        new_cube[1,:,4] = np.array(temp)
         
        temp = self.cube[1,:,4].tolist()
        temp.reverse()
        new_cube[1,:,1] = np.array(temp)
    
        temp = self.cube[1,:,1].tolist()
        new_cube[1,:,2] = np.array(temp)

        self.cube = new_cube

    def d(self):

        new_cube = np.copy(self.cube)

        new_cube[:,:,5] = self.cube[:,:,5].T
        for row in [0,1]:
            new_cube[row,:,5] = self.cube[:,:,5].T[1-row,:]

        temp = self.cube[1,:,3].tolist()
        new_cube[1,:,2] = np.array(temp)

        temp = self.cube[1,:,4].tolist()
        new_cube[1,:,3] = np.array(temp)
         
        temp = self.cube[1,:,1].tolist()
        temp.reverse()
        new_cube[1,:,4] = np.array(temp)
    
        temp = self.cube[1,:,2].tolist()
        new_cube[1,:,1] = np.array(temp)

        self.cube = new_cube

    def U(self):
        new_cube = np.copy(self.cube)

        my_face = 0
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for col in [0,1]:
            new_cube[:,col,my_face] = self.cube[:,:,my_face].T[:,1-col]

        temp = self.cube[0,:,2].tolist()
        new_cube[0,:,1] = np.array(temp)

        temp = self.cube[0,:,3].tolist()
        new_cube[0,:,2] = np.array(temp)
         
        temp = self.cube[0,:,4].tolist()
        new_cube[0,:,3] = np.array(temp)
    
        temp = self.cube[0,:,1].tolist()
        temp.reverse()
        new_cube[0,:,4] = np.array(temp)

        self.cube = new_cube

    def u(self):
        new_cube = np.copy(self.cube)

        my_face = 0
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for row in [0,1]:
            new_cube[row,:,my_face] = self.cube[:,:,my_face].T[1-row,:]

        temp = self.cube[0,:,1].tolist()
        new_cube[0,:,2] = np.array(temp)

        temp = self.cube[0,:,2].tolist()
        new_cube[0,:,3] = np.array(temp)
         
        temp = self.cube[0,:,3].tolist()
        new_cube[0,:,4] = np.array(temp)
    
        temp = self.cube[0,:,4].tolist()
        temp.reverse()
        new_cube[0,:,1] = np.array(temp)

        self.cube = new_cube

    def R(self):
        new_cube = np.copy(self.cube)

        my_face = 3
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for col in [0,1]:
            new_cube[:,col,my_face] = self.cube[:,:,my_face].T[:,1-col]

        temp = self.cube[:,1,2].tolist()
        new_cube[:,1,0] = np.array(temp)

        temp = self.cube[:,1,5].tolist()
        new_cube[:,1,2] = np.array(temp)
         
        temp = self.cube[:,1,0].tolist()
        temp.reverse()
        new_cube[:,0,4] = np.array(temp)
    
        temp = self.cube[:,0,4].tolist()
        temp.reverse()
        new_cube[:,1,5] = np.array(temp)

        self.cube = new_cube

    def r(self):
        new_cube = np.copy(self.cube)

        my_face = 3
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for row in [0,1]:
            new_cube[row,:,my_face] = self.cube[:,:,my_face].T[1-row,:]

        temp = self.cube[:,1,0].tolist()
        new_cube[:,1,2] = np.array(temp)

        temp = self.cube[:,1,2].tolist()
        new_cube[:,1,5] = np.array(temp)
         
        temp = self.cube[:,0,4].tolist()
        temp.reverse()
        new_cube[:,1,0] = np.array(temp)
    
        temp = self.cube[:,1,5].tolist()
        temp.reverse()
        new_cube[:,0,4] = np.array(temp)

        self.cube = new_cube

    def L(self):
        new_cube = np.copy(self.cube)

        my_face = 1
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for col in [0,1]:
            new_cube[:,col,my_face] = self.cube[:,:,my_face].T[:,1-col]

        temp = self.cube[:,0,2].tolist()
        new_cube[:,0,5] = np.array(temp)

        temp = self.cube[:,1,4].tolist()
        temp.reverse()
        new_cube[:,0,0] = np.array(temp)
         
        temp = self.cube[:,0,0].tolist()
        new_cube[:,0,2] = np.array(temp)
    
        temp = self.cube[:,0,5].tolist()
        temp.reverse()
        new_cube[:,1,4] = np.array(temp)

        self.cube = new_cube

    def l(self):
        new_cube = np.copy(self.cube)

        my_face = 1
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for row in [0,1]:
            new_cube[row,:,my_face] = self.cube[:,:,my_face].T[1-row,:]

        temp = self.cube[:,0,5].tolist()
        new_cube[:,0,2] = np.array(temp)

        temp = self.cube[:,0,0].tolist()
        temp.reverse()
        new_cube[:,1,4] = np.array(temp)
         
        temp = self.cube[:,0,2].tolist()
        new_cube[:,0,0] = np.array(temp)
    
        temp = self.cube[:,1,4].tolist()
        temp.reverse()
        new_cube[:,0,5] = np.array(temp)

        self.cube = new_cube


class Cube():
    class ActionSpace():
        def __init__(self, act_dim):
            self.act_dim = act_dim
        def sample(self):
            return np.random.randint(self.act_dim)
                
    class ObservationSpace():
        def __init__(self, obs_dim):
            self.obs_dim = obs_dim
        def call(self):
            return self.obs_dim
        
    def __init__(self, difficulty=10, obs_mode="mlp"):

        self.cube = np.zeros((3,3,6))
        self.action_dim = 12
        self.obs_dim = (54, 6)

        self.difficulty = difficulty
        self.action_space = self.ActionSpace(self.action_dim)
        self.observation_space = self.ObservationSpace(self.obs_dim)

        self.obs_mode = obs_mode 
        _ = self.reset()


    def reset(self, difficulty=None):

        self.moves = 0

        if difficulty is not None:
            self.difficulty = difficulty

        for face in range(6):
            self.cube[...,face] = face

        # Scramble cube
        for cc in range(self.difficulty):
            self.step(self.get_random_action())

        return self.categorical_cube()

    def set_difficulty(self, difficulty, verbose=True):
        if difficulty == self.difficulty:
            pass
        else:
            if verbose: print("changing difficulty from {} to {}".format(self.difficulty, difficulty))
            self.difficulty = difficulty

    def step(self, action):
        """
        available actions are F, R, L, U, D, B 
        and their reverses
        """
        if action == 0:
            self.U()
        elif action == 1:
            self.L()
        elif action == 2:
            self.F()
        elif action == 3:
            self.R()
        elif action == 4:
            self.B()
        elif action == 5:
            self.D()
        elif action == 6:
            self.u()
        elif action == 7:
            self.l()
        elif action == 8:
            self.f()
        elif action == 9:
            self.r()
        elif action == 10:
            self.b()
        elif action == 11:
            self.d()
        
        info = {}
        done = self.is_solved()

        self.moves += 1
        if done:
            reward = 26.0 + np.max([(26.0 - self.moves),0.0])
        else:
            reward = 0.0

        if self.obs_mode == "mlp":
            observation = self.categorical_cube()

        return observation, reward, done, info

    def categorical_cube(self):

        categorical_cube = np.zeros((54,6))

        flat_cube = np.copy(self.cube.ravel())

        for idx in range(len(flat_cube)):
            categorical_cube[idx,int(flat_cube[idx])] = 1.

        return categorical_cube

    def is_solved(self):

        for face in range(6):
            solve = self.cube[1,1,face] * np.ones_like(self.cube[:,:,face])
            if np.max(np.abs(solve - self.cube[:,:,face])) > 0:
                return False
            

        return True

    def get_random_action(self):
        return np.random.randint(self.action_dim)
        
    
    def display_cube(self):
    
        scoot = 10
        print('\n')
        for row in range(3):
            print(' ' * scoot + str(self.cube[row,:,0]) + ' ' * scoot) # Up
        
        for long_row in range(3):
            print(str(self.cube[long_row,:,1])+\
            str(self.cube[long_row,:,2])+\
            str(self.cube[long_row,:,3])+\
            str(self.cube[long_row,:,4]))

        for row in range(3):
            print(' ' * scoot + str(self.cube[row,:,5]) + ' ' * scoot) # Up

    def render(self):
        self.display_cube()
        time.sleep(0.01)

    def F(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,2] = self.cube[:,:,2].T
        for col in [0,2]:
            new_cube[:,col,2] = self.cube[:,:,2].T[:,2-col]

        temp = self.cube[:,2,1].tolist()
        temp.reverse()
        new_cube[2,:,0] = np.array(temp)

        temp = self.cube[2,:,0].tolist()
        new_cube[:,0,3] = np.array(temp)

        temp = self.cube[:,0,3].tolist()
        temp.reverse()
        new_cube[0,:,5] = np.array(temp)
        
        temp = self.cube[0,:,5].tolist()
        new_cube[:,2,1] = np.array(temp)
    
        self.cube = new_cube

    def f(self):
        new_cube = np.copy(self.cube)

        new_cube[:,:,2] = self.cube[:,:,2].T
        for row in [0,2]:
            new_cube[row,:,2] = self.cube[:,:,2].T[2-row,:]

        temp = self.cube[2,:,0].tolist()
        temp.reverse()
        new_cube[:,2,1] = np.array(temp)

        temp = self.cube[:,0,3].tolist()
        new_cube[2,:,0] = np.array(temp)

        temp = self.cube[0,:,5].tolist()
        temp.reverse()
        new_cube[:,0,3] = np.array(temp)
        
        temp = self.cube[:,2,1].tolist()
        new_cube[0,:,5] = np.array(temp)
    
        self.cube = new_cube

    def B(self):

        new_cube = np.copy(self.cube)

        new_cube[:,:,4] = self.cube[:,:,4].T
        for col in [0,2]:
            new_cube[:,col,4] = self.cube[:,:,4].T[:,2-col]

        temp = self.cube[2,:,5].tolist()
        temp.reverse()
        new_cube[:,2,3] = np.array(temp)

        temp = self.cube[:,2,3].tolist()
        new_cube[0,:,0] = np.array(temp)
         
        temp = self.cube[0,:,0].tolist()
        temp.reverse()
        new_cube[:,0,1] = np.array(temp)
    
        temp = self.cube[:,0,1].tolist()
        new_cube[2,:,5] = np.array(temp)

        self.cube = new_cube

    def b(self):

        new_cube = np.copy(self.cube)

        new_cube[:,:,4] = self.cube[:,:,4].T
        for row in [0,2]:
            new_cube[row,:,4] = self.cube[:,:,4].T[2-row,:]

        temp = self.cube[:,2,3].tolist()
        temp.reverse()
        new_cube[2,:,5] = np.array(temp)

        temp = self.cube[0,:,0].tolist()
        new_cube[:,2,3] = np.array(temp)
         
        temp = self.cube[:,0,1].tolist()
        temp.reverse()
        new_cube[0,:,0] = np.array(temp)
    
        temp = self.cube[2,:,5].tolist()
        new_cube[:,0,1] = np.array(temp)

        self.cube = new_cube

    def D(self):

        new_cube = np.copy(self.cube)

        new_cube[:,:,5] = self.cube[:,:,5].T
        for col in [0,2]:
            new_cube[:,col,5] = self.cube[:,:,5].T[:,2-col]

        temp = self.cube[2,:,2].tolist()
        new_cube[2,:,3] = np.array(temp)

        temp = self.cube[2,:,3].tolist()
        new_cube[2,:,4] = np.array(temp)
         
        temp = self.cube[2,:,4].tolist()
        temp.reverse()
        new_cube[2,:,1] = np.array(temp)
    
        temp = self.cube[2,:,1].tolist()
        new_cube[2,:,2] = np.array(temp)

        self.cube = new_cube

    def d(self):

        new_cube = np.copy(self.cube)

        new_cube[:,:,5] = self.cube[:,:,5].T
        for row in [0,2]:
            new_cube[row,:,5] = self.cube[:,:,5].T[2-row,:]

        temp = self.cube[2,:,3].tolist()
        new_cube[2,:,2] = np.array(temp)

        temp = self.cube[2,:,4].tolist()
        new_cube[2,:,3] = np.array(temp)
         
        temp = self.cube[2,:,1].tolist()
        temp.reverse()
        new_cube[2,:,4] = np.array(temp)
    
        temp = self.cube[2,:,2].tolist()
        new_cube[2,:,1] = np.array(temp)

        self.cube = new_cube

    def U(self):
        new_cube = np.copy(self.cube)

        my_face = 0
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for col in [0,2]:
            new_cube[:,col,my_face] = self.cube[:,:,my_face].T[:,2-col]

        temp = self.cube[0,:,2].tolist()
        new_cube[0,:,1] = np.array(temp)

        temp = self.cube[0,:,3].tolist()
        new_cube[0,:,2] = np.array(temp)
         
        temp = self.cube[0,:,4].tolist()
        new_cube[0,:,3] = np.array(temp)
    
        temp = self.cube[0,:,1].tolist()
        temp.reverse()
        new_cube[0,:,4] = np.array(temp)

        self.cube = new_cube

    def u(self):
        new_cube = np.copy(self.cube)

        my_face = 0
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for row in [0,2]:
            new_cube[row,:,my_face] = self.cube[:,:,my_face].T[2-row,:]

        temp = self.cube[0,:,1].tolist()
        new_cube[0,:,2] = np.array(temp)

        temp = self.cube[0,:,2].tolist()
        new_cube[0,:,3] = np.array(temp)
         
        temp = self.cube[0,:,3].tolist()
        new_cube[0,:,4] = np.array(temp)
    
        temp = self.cube[0,:,4].tolist()
        temp.reverse()
        new_cube[0,:,1] = np.array(temp)

        self.cube = new_cube

    def R(self):
        new_cube = np.copy(self.cube)

        my_face = 3
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for col in [0,2]:
            new_cube[:,col,my_face] = self.cube[:,:,my_face].T[:,2-col]

        temp = self.cube[:,2,2].tolist()
        new_cube[:,2,0] = np.array(temp)

        temp = self.cube[:,2,5].tolist()
        new_cube[:,2,2] = np.array(temp)
         
        temp = self.cube[:,2,0].tolist()
        temp.reverse()
        new_cube[:,0,4] = np.array(temp)
    
        temp = self.cube[:,0,4].tolist()
        temp.reverse()
        new_cube[:,2,5] = np.array(temp)

        self.cube = new_cube

    def r(self):
        new_cube = np.copy(self.cube)

        my_face = 3
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for row in [0,2]:
            new_cube[row,:,my_face] = self.cube[:,:,my_face].T[2-row,:]

        temp = self.cube[:,2,0].tolist()
        new_cube[:,2,2] = np.array(temp)

        temp = self.cube[:,2,2].tolist()
        new_cube[:,2,5] = np.array(temp)
         
        temp = self.cube[:,0,4].tolist()
        temp.reverse()
        new_cube[:,2,0] = np.array(temp)
    
        temp = self.cube[:,2,5].tolist()
        temp.reverse()
        new_cube[:,0,4] = np.array(temp)

        self.cube = new_cube

    def L(self):
        new_cube = np.copy(self.cube)

        my_face = 1
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for col in [0,2]:
            new_cube[:,col,my_face] = self.cube[:,:,my_face].T[:,2-col]

        temp = self.cube[:,0,2].tolist()
        new_cube[:,0,5] = np.array(temp)

        temp = self.cube[:,2,4].tolist()
        temp.reverse()
        new_cube[:,0,0] = np.array(temp)
         
        temp = self.cube[:,0,0].tolist()
        new_cube[:,0,2] = np.array(temp)
    
        temp = self.cube[:,0,5].tolist()
        temp.reverse()
        new_cube[:,2,4] = np.array(temp)

        self.cube = new_cube

    def l(self):
        new_cube = np.copy(self.cube)

        my_face = 1
        new_cube[:,:,my_face] = self.cube[:,:,my_face].T
        for row in [0,2]:
            new_cube[row,:,my_face] = self.cube[:,:,my_face].T[2-row,:]

        temp = self.cube[:,0,5].tolist()
        new_cube[:,0,2] = np.array(temp)

        temp = self.cube[:,0,0].tolist()
        temp.reverse()
        new_cube[:,2,4] = np.array(temp)
         
        temp = self.cube[:,0,2].tolist()
        new_cube[:,0,0] = np.array(temp)
    
        temp = self.cube[:,2,4].tolist()
        temp.reverse()
        new_cube[:,0,5] = np.array(temp)

        self.cube = new_cube

if __name__ == "__main__":

    env = Cube1(difficulty = 4, use_target=True, scramble_actions=True)
    obs = env.reset()

    import pdb; pdb.set_trace()
    done = False
    steps = []
    for cc in range(1000):
        step = 0
        done = False
        while not done:
            _ = env.reset()
            obs, reward, done, info = env.step(env.action_space.sample()) 
            step += 1
        print("guessed solving move once in {}".format(step))
        steps.append(step)

    avg_step = np.mean(steps)
    std_step = np.std(steps)
    print("done: {}, reward: {}, avg trials before solve {}+/-{} std dev. ".format(done,reward, avg_step, std_step))
    env.display_cube()
