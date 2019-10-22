import numpy as np

class Cube():
    def __init__(self, difficulty=10):

        self.cube = np.zeros((3,3,6))
        self.action_dim = 6
        self.difficulty = difficulty
        self.reset()


    def reset(self):

        for face in range(6):
            self.cube[...,face] = face

        # Scramble cube
        for cc in range(self.difficulty):
            self.step(self.get_random_action())

    def step(self, action):
        """
        available actions are F, R, L, U, D, B 
        and their reverses
        """
        if action == 0:
            self.F()
        elif action == 1:
            self.R()
        elif action == 2:
            self.D()
        elif action == 3:
            self.L()
        elif action == 4:
            self.U()
        elif action == 5:
            self.B()

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

if __name__ == "__main__":
    cube = Cube()

    cube.reset()
    cube.display_cube()
    cube.F()
    print("F: ") 
    cube.display_cube()
    cube.B()
    print("B: ") 
    cube.display_cube()
    cube.U()
    print("U: ") 
    cube.display_cube()
    cube.R()
    print("R: ") 
    cube.display_cube()
    cube.L()
    print("L: ") 
    cube.display_cube()

