import numpy as np
import time

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
        
    def __init__(self, difficulty=10):

        self.cube = np.zeros((3,3,6))
        self.action_dim = 12
        self.obs_dim = (54, 6)

        self.difficulty = difficulty
        self.action_space = self.ActionSpace(self.action_dim)
        self.observation_space = self.ObservationSpace(self.obs_dim)

        _ = self.reset()


    def reset(self, difficulty=None):

        if difficulty is not None:
            self.difficulty = difficulty

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

        if done:
            reward = 32
        else:
            reward = -1

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

    env = Cube(difficulty = 1)
    obs = env.reset()


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
