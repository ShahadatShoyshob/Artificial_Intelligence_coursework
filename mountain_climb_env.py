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

class MountainClimbEnv(gym.Env):
    def __init__(self):
        # For each new environment, establish a fresh PyBullet connection
        # This ensures URDF files are not cached
        self.physics_client = p.connect(p.GUI)
        p.setGravity(0, 0, -10)
        self._setup_world()

        # Reset step counter and other variables
        self.step_num = 0
        self.dist_moved = 0
        self.vel = 0
        self.new_pos = 0
        self.reward = 0

        self.prev_z = 0.0  # Track previous height


        # Remove old robot if it exists
        if hasattr(self, 'rob1'):
            p.removeBody(self.rob1)

        # Generate a random creature
        gene_count = np.random.randint(3, 13)
        print(f"Gene Count: {gene_count}")
        self.cr = creature.Creature(gene_count)

        # Debug: Check how many motors the creature actually has
        motors = self.cr.get_motors()
        print(f"Creature motors count: {len(motors)}")

        # Save it to XML with a unique filename to avoid caching issues
        import time
        timestamp = str(int(time.time() * 1000000))  # microsecond timestamp
        urdf_filename = f'test_{timestamp}.urdf'

        urdf_content = self.cr.to_xml()
        with open(urdf_filename, 'w') as f:
            f.write(urdf_content)

        # Debug: Print URDF content to see what's actually being generated
        print("URDF Joint count:", urdf_content.count('<joint'))
        print("URDF Link count:", urdf_content.count('<link'))

        # Load it into the sim with the unique filename
        self.rob1 = p.loadURDF(urdf_filename, (0, 7, 7))

        # Store filename for cleanup
        self.urdf_filename = urdf_filename

        self.num_joints = p.getNumJoints(self.rob1)
        print(f"Number of Joints in PyBullet: {self.num_joints}")

        # Verify joint information
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.rob1, i)
            print(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")

        self.start_pos, self.orn = p.getBasePositionAndOrientation(self.rob1)

        self.plot_x = []
        self.plot_y = []
        self.target_pose = [0, 0, 0]

        # Define observation space
        self.observation_space = Box(
            low=np.array([-10.0, -10.0], dtype=np.float32),
            high=np.array([10.0, 10.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # Define action space based on current creature's joints
        self.action_space = Box(
            low=np.array([-1.0] * self.num_joints, dtype=np.float32),
            high=np.array([1.0] * self.num_joints, dtype=np.float32),
            shape=(self.num_joints,),
            dtype=np.float32
        )

    def _setup_world(self):
        """Setup the arena and mountain"""
        arena_size = 20
        self.make_arena(arena_size=arena_size)

        # Create mountain
        mountain_position = (0, 0, -3)
        mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
        p.setAdditionalSearchPath('shapes/')
        create_mountain = prepare_shapes.generate_gaussian_pyramid4("./shapes/gaussian_pyramid.obj")
        self.mountain = p.loadURDF("gaussian_pyramid.urdf", mountain_position, mountain_orientation, useFixedBase=1)

    def step(self, action):
        self.step_num += 1
        p.stepSimulation()
        p.setRealTimeSimulation(1)
        time.sleep(1.0/240)
        # Update position
        self.new_pos, self.orn = p.getBasePositionAndOrientation(self.rob1)

        # Get current z position (height)
        current_z = self.new_pos[2]

        # Reward is height gain since last step
        self.reward = current_z - self.prev_z

        # Optionally clip small negative rewards to avoid penalizing slight drops
        if self.reward < -0.01:
            self.reward = -0.01  # Allow falling penalty if needed
        elif self.reward < 0:
            self.reward = 0  # No reward if slightly descending

        # Save for next step
        self.prev_z = current_z

        # Prepare observation
        obs = [self.new_pos[0], self.new_pos[1]]


        if self.step_num % 1000 == 0:
            motors = self.cr.get_motors()

            # Debug print to see what's happening
            print(f"Motors count: {len(motors)}, Joints count: {p.getNumJoints(self.rob1)}")

            # Only control joints that correspond to motors
            active_joints = min(len(motors), p.getNumJoints(self.rob1))

            for jid in range(active_joints):
                mode = p.VELOCITY_CONTROL
                vel = action[jid] if jid < len(action) else 0.0
                p.setJointMotorControl2(self.rob1,
                                      jid,
                                      controlMode=mode,
                                      targetVelocity=vel)

            # Handle remaining joints (set to zero velocity)
            for jid in range(active_joints, p.getNumJoints(self.rob1)):
                p.setJointMotorControl2(self.rob1,
                                      jid,
                                      controlMode=p.VELOCITY_CONTROL,
                                      targetVelocity=0.0)

            self.new_pos, self.orn = p.getBasePositionAndOrientation(self.rob1)
            self.dist_moved = np.linalg.norm(np.asarray(self.new_pos) - np.asarray(self.target_pose))

        self.reward = self.dist_moved
        self.new_pos, self.orn = p.getBasePositionAndOrientation(self.rob1)
        obs = [self.new_pos[0], self.new_pos[1]]

        truncated = (self.step_num % 10000 == 0)
        terminated = False
        info = {}

        return obs, self.reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.prev_z = self.start_pos[2]


        # Reset robot position
        p.resetBasePositionAndOrientation(self.rob1, self.start_pos, self.orn)

        # Reset internal variables
        self.step_num = 0
        self.dist_moved = 0
        self.vel = 0
        self.new_pos = 0
        self.reward = 0

        # Return initial observation
        obs = [self.start_pos[0], self.start_pos[1]]
        info = {}
        return obs, info

    def close(self):
        """Clean up the environment"""
        if hasattr(self, 'rob1'):
            p.removeBody(self.rob1)

        # Clean up the URDF file
        if hasattr(self, 'urdf_filename'):
            import os
            try:
                os.remove(self.urdf_filename)
            except FileNotFoundError:
                pass

        # Disconnect this instance's physics client
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)

    def make_arena(self, arena_size=10, wall_height=1):
        """Create the arena walls and floor"""
        wall_thickness = 0.5
        floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness])
        floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness], rgbaColor=[1, 1, 0, 1])
        floor_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, -wall_thickness])

        wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2])
        wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size/2, wall_thickness/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])

        # Create four walls
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, arena_size/2, wall_height/2])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -arena_size/2, wall_height/2])

        wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2])
        wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness/2, arena_size/2, wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])

        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[arena_size/2, 0, wall_height/2])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[-arena_size/2, 0, wall_height/2])


def main():
    """Main training loop"""
    env = None
    episode_count = 0

    try:
        while True:
            # Create new environment for each episode
            if env is not None:
                env.close()

            env = MountainClimbEnv()
            episode_count += 1
            print(f"Starting Episode {episode_count}")

            obs, info = env.reset()
            episode_reward = 0
            step_count = 0

            while True:
                # Sample action from action space
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                step_count += 1

                if terminated or truncated:
                    print(f"Episode {episode_count} finished after {step_count} steps")
                    print(f"Episode Reward: {episode_reward}")
                    break

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
