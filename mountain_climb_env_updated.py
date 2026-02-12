import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import creature
import genome
import math
import prepare_shapes
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from creature import Motor
from typing import Tuple

p.connect(p.GUI)
step_num = 0
dist_moved = 0
vel = 0
new_pos = 0
reward = 0
truncated = False
class MountainClimbEnv(gym.Env):
    def __init__(self):
        gene_count = np.random.randint(3, 13)  # Upper limit is exclusive, so use 13 to include 12
        print (f"Gene Count: {gene_count}")
        # generate a random creature
        self.cr = creature.Creature(gene_count)
        # save it to XML
        with open('test.urdf', 'w') as f:
            f.write(self.cr.to_xml())
            # load it into the sim
        self.rob1 = p.loadURDF('test.urdf', (0, 7, 7))

        self.num_joints = p.getNumJoints(self.rob1)
        print (f"Number of Joints: {self.num_joints}")
        self.start_pos, self.orn = p.getBasePositionAndOrientation(self.rob1)

        self.plot_x = []
        self.plot_y = []
        self.target_pose = [0,0,0]


        self.observation_space = Box(
            low=np.array([-10.0, -10.0], dtype=np.float32),
            high=np.array([10.0, 10.0], dtype=np.float32),
            shape = (2,),
            dtype=np.float32
        )

        self.action_space = Box(
            low=np.array([-1.0] * self.num_joints, dtype=np.float32),  # Example: actions in range [-1, 1] for each joint
            high=np.array([1.0] * self.num_joints, dtype=np.float32),
            shape=(self.num_joints,),
            dtype=np.float32
        )

    def step(self, action):
        global dist_moved
        global step_num
        global vel
        global new_pos
        global reward
        global truncated
        step_num += 1
        p.stepSimulation()
        p.setRealTimeSimulation(1)
        time.sleep(1.0/240)
        if step_num % 1000 == 0:
            #print ("Getting New Motors")
            motors = self.cr.get_motors()
            assert len(motors) == p.getNumJoints(self.rob1), "Something went wrong"
            for jid in range(p.getNumJoints(self.rob1)):
                mode = p.VELOCITY_CONTROL
                vel = action[jid]
                #print (f"Velocity Received: {vel}")
                p.setJointMotorControl2(self.rob1,
                            jid,
                            controlMode=mode,
                            targetVelocity=vel)
            self.new_pos, self.orn = p.getBasePositionAndOrientation(self.rob1)

            dist_moved = np.linalg.norm(np.asarray(self.new_pos) - np.asarray(self.target_pose))
        reward = dist_moved
        self.new_pos, self.orn = p.getBasePositionAndOrientation(self.rob1)
        obs = [self.new_pos[0], self.new_pos[1]]
        if (step_num % 10000 == 0):
            truncated = True
        terminated = False
        info = {}
        return obs, reward, terminated, truncated, info
        #print(f"Distance Travelled: {dist_moved}")
        #print (new_pos)

    def reset(self):
        p.resetBasePositionAndOrientation(self.rob1, self.start_pos, self.orn)
        info = {}
        return self.start_pos, info
    def close(self):
        p.removeBody(self.rob1)

def make_mountain(num_rocks=100, max_size=0.25, arena_size=10, mountain_height=5):
    def gaussian(x, y, sigma=arena_size/4):
        """Return the height of the mountain at position (x, y) using a Gaussian function."""
        return mountain_height * math.exp(-((x**2 + y**2) / (2 * sigma**2)))

    for _ in range(num_rocks):
        x = random.uniform(-1 * arena_size/2, arena_size/2)
        y = random.uniform(-1 * arena_size/2, arena_size/2)
        z = gaussian(x, y)  # Height determined by the Gaussian function

        # Adjust the size of the rocks based on height. Higher rocks (closer to the peak) will be smaller.
        size_factor = 1 - (z / mountain_height)
        size = random.uniform(0.1, max_size) * size_factor

        orientation = p.getQuaternionFromEuler([random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14)])
        rock_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size])
        rock_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0.5, 0.5, 0.5, 1])
        rock_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rock_shape, baseVisualShapeIndex=rock_visual, basePosition=[x, y, z], baseOrientation=orientation)



def make_rocks(num_rocks=100, max_size=0.25, arena_size=10):
    for _ in range(num_rocks):
        x = random.uniform(-1 * arena_size/2, arena_size/2)
        y = random.uniform(-1 * arena_size/2, arena_size/2)
        z = 0.5  # Adjust based on your needs
        size = random.uniform(0.1,max_size)
        orientation = p.getQuaternionFromEuler([random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14)])
        rock_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size])
        rock_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0.5, 0.5, 0.5, 1])
        rock_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=rock_shape, baseVisualShapeIndex=rock_visual, basePosition=[x, y, z], baseOrientation=orientation)


def make_arena(arena_size=10, wall_height=1):
    wall_thickness = 0.5
    floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness])
    floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness], rgbaColor=[1, 1, 0, 1])
    floor_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, -wall_thickness])

    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray walls

    # Create four walls
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, arena_size/2, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -arena_size/2, wall_height/2])

    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray walls

    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[arena_size/2, 0, wall_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[-arena_size/2, 0, wall_height/2])

def step():
    return


p.setGravity(0, 0, -10)

arena_size = 20
make_arena(arena_size=arena_size)

#make_rocks(arena_size=arena_size)
mountain_position = (0, 0, -3)  # Adjust as needed
mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
p.setAdditionalSearchPath('shapes/')
# mountain = p.loadURDF("mountain.urdf", mountain_position, mountain_orientation, useFixedBase=1)
# mountain = p.loadURDF("mountain_with_cubes.urdf", mountain_position, mountain_orientation, useFixedBase=1)
create_mountain = prepare_shapes.generate_gaussian_pyramid4("./shapes/gaussian_pyramid.obj")

mountain = p.loadURDF("gaussian_pyramid.urdf", mountain_position, mountain_orientation, useFixedBase=1)

#print (f"Environment Observation Space: {env.observation_space.sample()}")
#print (f"Environment Action Space: {env.action_space.sample()}")

def make_env():
    return MountainClimbEnv()

env = make_env()
truncated = False
while True:
    if truncated:
        env.close()
        env = make_env()  # Assign the new environment to 'env'
        truncated = False
        print (f"Episode Reward: {reward}")
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())


