import mengine as m
import numpy as np
import os
from thrower_setup import ThrowerSetup
import pybullet as p
import pinocchio as pin

from tqdm import tqdm

import pickle
import matplotlib.pyplot as plt

import torch
from utils import circular_trajectory, vel_ballistic_model
from dataclasses import dataclass, field

from model import TorquePredictionModel, pad_sequences

import time

GRAVITY = 9.81
MAX_ATTEMPTS = 1_000

@dataclass
class DataStream:
    sensor_stream_motion: list[tuple] = field(default_factory=list)
    joint_stream_motion: list[tuple] = field(default_factory=list)
    sensor_static: tuple = None
    distance_desired: float = None
    residual_vel: float = None

class Thrower:
    def __init__(self, env: m.Env, id: int, train: bool = True, render = True):
        """
        Make sure id is different between thrower
        """
        self.train = train
        self.ts = ThrowerSetup(env, render = render)
        self.id = id
        self.datastreams = []
        self.render = render

    def run_sequence(self, distance_desired: float = None):
        self.ts.robot.set_gripper_position([1] * 2, set_instantly=True)
        #diff_grasps = np.linspace(-0.04, 0.035, len(self.ts.array_objects))[::-1]
        diff_grasps = np.random.uniform(-0.04, 0.035, len(self.ts.array_objects))

        for i in range(1):#len(self.ts.array_objects)):
            s = time.time()

            pos, _ = list(self.ts.array_objects[i].get_base_pos_orient())
            diff_grasp = diff_grasps[i]

            pos[0] += diff_grasp
            self.pick_object(pos)

            # Collect of info
            data = self.collect_data_stream()
            if not self.train and distance_desired:
                data.distance_desired = distance_desired

            vel_control = self.vel_controller(self.train, data)

            residual_vel, distance_reached = self.throw_object(self.ts.array_objects[i], vel=vel_control)
            
            if self.train:
                data.distance_desired = distance_reached
                data.residual_vel = residual_vel
                self.datastreams.append(data)

        return distance_reached
            #print("Time processing per object:", time.time() - s)
                
    def store_data_stream(self):
        train_saving_dir = "train"
        if not os.path.exists(train_saving_dir):
            os.makedirs(train_saving_dir)

        fp = os.path.join("train", f"datastreams_{self.id}.pkl")
        with open(fp, 'wb') as f:
            pickle.dump(self.datastreams, f)

    def collect_data_stream(self) -> DataStream:
        robot = self.ts.robot
        data = DataStream()
        joints_data = [4, 5, 6]

        # Create initial stable joint config
        stable_joint_config = list(self.ts.start_joints)
        stable_joint_config[self.ts.thrower_link] += 0.5 # Literal to have a range
        self.moveto(joint_angles=stable_joint_config)

        # Static measurement
        data.sensor_static = robot.get_motor_joint_states(joints_data)[2]
        # print("Controllable:", joints_data) [0, 1, 2, 3, 4, 5, 6]

        # Motion measurement
        robot.control(self.ts.start_joints)
        attempts = 0
        while np.linalg.norm(robot.get_joint_angles(robot.controllable_joints) - self.ts.start_joints) > 0.05:
            attempts += 1
            if attempts > MAX_ATTEMPTS:
                raise Exception("Something happened in collecting data motion. Exiting.")
            states = robot.get_motor_joint_states(joints_data)
            print(states[2])
            data.sensor_stream_motion.append(states[2])
            data.joint_stream_motion.append(states[1])
            m.step_simulation(realtime=self.render)
        
        return data

    def vel_controller(self, train: bool, data: DataStream) -> float:
        vel = np.random.uniform(1.0, 5.2) # Physically possible
        if not train and data.distance_desired:
            # Ballistic model
            estimated_pose_released = np.array([-0.123, 0.,  1.11 ])
            estimated_pose_obj_landed = np.array([estimated_pose_released[0] - data.distance_desired,
                                                  0.0,
                                                  self.ts.overtable_pose[2] + 0.2])

            delta_x = abs(estimated_pose_obj_landed[0] - estimated_pose_released[0])
            delta_z = estimated_pose_obj_landed[2] - estimated_pose_released[2]
            angle_to_base = np.pi + np.deg2rad(-120)

            vel_ballistic = vel_ballistic_model(delta_x, delta_z, angle_to_base)

            # residual velocity
            model = TorquePredictionModel(dim_model=3, num_heads=1, num_layers=1, dim_feedforward=64, fixed_vec_dim=4, output_dim=1)
            model.load_state_dict(torch.load('best_model.pth', weights_only=True))

            model.eval()

            torque_seq, _ = pad_sequences([torch.tensor(data.sensor_stream_motion)])
            joint_seq, _ = pad_sequences([torch.tensor(data.joint_stream_motion)])
            fixed_vec = torch.tensor([list(data.sensor_static) + [data.distance_desired]], dtype=torch.float32)
            
            model_output = model(torque_seq, fixed_vec, joint_seq)

            estimated_residual_vel = model_output.item()

            if not self.train:
                m.Shape(m.Sphere(radius=0.01), static=True, collision=False, position=estimated_pose_obj_landed, rgba=[0, 1, 0, 1])
                estimated_pose_obj_landed[2] -= 0.2
                obj = m.Shape(m.Mesh(filename='./objects/basket/meshes/model.obj', scale=[0.9, 0.9, 0.9]), static=True, mass=0.250, position=estimated_pose_obj_landed, orientation=m.get_quaternion([0,0,np.pi]), rgba=None, visual=True, collision=False)
                obj.set_whole_body_frictions(lateral_friction=0, spinning_friction=0, rolling_friction=0)

            vel = vel_ballistic + estimated_residual_vel

        # acc = np.array([0.0, 0.0, -9.81])
        # v = np.array([-vel_ballistic * np.cos(np.deg2rad(-120) + np.pi), 0., vel_ballistic * np.sin(np.deg2rad(-120) + np.pi)])
        # for t in np.linspace(0.0,0.7,100):
        #     pose = estimated_pose_released + v * t + 0.5 * acc * t**2
        #     m.Shape(m.Sphere(radius=0.01), static=True, collision=False,
        #          position=pose, rgba=[1, 0, 0, 1])
        
        # v = np.array([-vel * np.cos(np.deg2rad(-120) + np.pi), 0., vel * np.sin(np.deg2rad(-120) + np.pi)])
        # for t in np.linspace(0.0,0.7,100):
        #     pose = estimated_pose_released + v * t + 0.5 * acc * t**2
        #     m.Shape(m.Sphere(radius=0.01), static=True, collision=False,
        #          position=pose, rgba=[0, 1, 0, 1])

        # m.step_simulation(1000, realtime=True)
        return vel

    def throw_object(self, obj, vel:float = 5.0, angle_release_deg: float = -120):
        robot = self.ts.robot

        if obj.get_base_pos_orient()[0][2] < self.ts.overtable_pose[2] + 0.10:
            p.removeBody(obj.body)
            print("Warning: Objects not picked.")
            return (np.nan, np.nan)

        # Move to pose
        j_config = list(self.ts.start_joints)
        j_config[self.ts.thrower_link] -= np.deg2rad(10)
        self.moveto(joint_angles=j_config)#(joint_angles=self.ts.start_joints)

        # Throwing control
        vel_joint_des = np.zeros_like(robot.controllable_joints, dtype=float)
        vel_joint_des[self.ts.thrower_link] = vel / self.ts.thrower_radius # omega
        # print("vel:", vel)
        robot.control(vel_joint_des, velocity_control=True)

        angle_release = np.deg2rad(angle_release_deg)
        attempts = 0
        while robot.get_joint_angles(robot.controllable_joints)[self.ts.thrower_link] < angle_release:
            attempts += 1
            # print(robot.get_joint_velocities(robot.controllable_joints)[self.ts.thrower_link], vel / self.ts.thrower_radius)
            if attempts > MAX_ATTEMPTS:
                raise Exception("Something happened when throwing. Exiting")
            m.step_simulation(1, realtime=self.render)

        # actual_release_vel = np.linalg.norm(obj.get_link_velocity(obj.base))
        # print(actual_release_vel)
        
        # release
        self.ts.robot.set_gripper_position([1] * 2, set_instantly=True)
        pose_released = robot.get_link_pos_orient(robot.end_effector)[0] # [x, y, z]
        robot.control(np.zeros_like(robot.controllable_joints, dtype=float), velocity_control=True)

        angle_released = robot.get_joint_angles(robot.controllable_joints)[self.ts.thrower_link]
        if np.rad2deg(abs(angle_released - angle_release)) > 4: # 5 degree to rad
            print(f"WARNING: Object thrown at different angle: {np.rad2deg(abs(angle_released - angle_release))}")

        # Release object calculation
        attempts = 0
        while obj.get_base_pos_orient()[0][2] > self.ts.overtable_pose[2] + 0.2: # z < table_height
            attempts += 1
            if attempts > MAX_ATTEMPTS:
                print("Objects stayed at:", obj.get_base_pos_orient()[0][2] - self.ts.overtable_pose[2] - 0.20)
                raise Exception("Something happened when object falling. Exiting")
            m.step_simulation(1, realtime=self.render)

        pose_obj_landed, _ = obj.get_base_pos_orient()
        
        # release_vel = np.array([-vel * np.cos(th + np.pi), 0., vel * np.sin(th + np.pi)])

        delta_x = abs(pose_obj_landed[0] - pose_released[0])
        delta_z = pose_obj_landed[2] - pose_released[2]
        angle_to_base = np.pi + angle_released

        vel_ballistic = vel_ballistic_model(delta_x, delta_z, angle_to_base)

        residual_vel = vel - vel_ballistic

        # print("vels:", vel, vel_physics)
        distance_reached = np.abs(delta_x)
        # print(distance_reached)

        # actual_release_vel = np.array([-actual_release_vel * np.cos(th + np.pi), 0., actual_release_vel * np.sin(th + np.pi)])
        # for t in np.linspace(0.3,0.8,30):
        #     pose = pose_released + release_vel * t + 0.5 * acc * t**2
        #     m.Shape(m.Sphere(radius=0.005), static=True, collision=False,
        #          position=pose, rgba=[0, 1, 0, 1])

        # if self.train:
        p.removeBody(obj.body)

        robot.control(self.ts.start_joints, set_instantly=True)
        return residual_vel, distance_reached
        

    def pick_object(self, object_pose:np.ndarray, hold_pose:np.ndarray = np.array([0.,0.,0.25])):
        # Open gripper
        self.ts.robot.set_gripper_position([1] * 2, set_instantly=True)

        goal_pose = (object_pose + hold_pose, m.get_quaternion(np.array([np.pi, 0.4, 0])))
        self.moveto(ee_pose=goal_pose)

        goal_pose = (object_pose + 0.4*hold_pose, m.get_quaternion(np.array([np.pi, 0.4, 0])))
        self.moveto(ee_pose=goal_pose)

        goal_pose = (object_pose + 0.2*hold_pose, m.get_quaternion(np.array([np.pi, 0.4, 0])))
        self.moveto(ee_pose=goal_pose)

        m.step_simulation(steps=100, realtime=self.render)

        robot_move = (object_pose + np.array([0.,0.,0.0]), m.get_quaternion(np.array([np.pi, 0.4, 0])))
        self.moveto(ee_pose=robot_move)

        # Close gripper
        self.ts.robot.set_gripper_position([0] * 2, force=2000)
        m.step_simulation(steps=100, realtime=self.render)

        goal_pose = (object_pose + 0.2*hold_pose, m.get_quaternion(np.array([np.pi, 0.4, 0])))
        self.moveto(ee_pose=goal_pose)

        goal_pose = (object_pose + 0.4*hold_pose, m.get_quaternion(np.array([np.pi, 0.4, 0])))
        self.moveto(ee_pose=goal_pose)

        robot_move = (object_pose + hold_pose, m.get_quaternion(np.array([np.pi, 0.4, 0])))
        self.moveto(ee_pose=robot_move)

        m.step_simulation(steps=100, realtime=self.render)


    def moveto(self, ee_pose = None, joint_angles = None):
        """
        if joint_angles passed, it overwrites ee_pose
        """
        robot = self.ts.robot

        if joint_angles is None and ee_pose is not None:
            joint_angles = robot.ik(self.ts.robot.end_effector, target_pos=ee_pose[0], target_orient=ee_pose[1],
                                        use_current_joint_angles=True)
        elif joint_angles is None:
            return
            
        robot.control(joint_angles)
        
        attempts = 0
        while np.linalg.norm(robot.get_joint_angles(robot.controllable_joints) - joint_angles) > 0.05:
            attempts += 1
            if attempts > MAX_ATTEMPTS:
                raise Exception("Something happened when moving to pose. Exiting")
            m.step_simulation(realtime=self.render)
        return
        

if __name__ == '__main__':
    train = True
    render = not train

    train_batches = 1000
    random_seeds = np.random.randint(0, 2**32 - 1, train_batches)
    for i in tqdm(range(train_batches)):
        env = m.Env(time_step =0.01, render=render, seed=random_seeds[i])
        ground = m.Ground()
        try:
            t = Thrower(env, id = i, render = render, train = train)
            t.run_sequence()
            t.store_data_stream()
        except Exception as e:
            print(e)
        env.disconnect()
        
# Create environment
## Robot
## Tray
## Array Objects 30?? -> 5 min?
## Basket


# a = -9.81
# b = vel * np.sin(th + np.pi)
# c = pose_released[2] - self.ts.overtable_pose[2]

# t_break = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

# final_pose = pose_released + t_break*release_vel + t_break**2*acc
# m.Shape(m.Sphere(radius=0.05), static=True, collision=False,
#                 position=final_pose, rgba=[0, 1, 0, 1])

# Create sequence of steps.
# Pick sequence. Random grasping. If passed, use it.
# Motion and read data of sensor (q, torque)
# Throw using screw coordinates. Simple PID.
# Range of speed and release place. And passed couple (r_i, v_i)
# Estimation of residual velocity. Velocity thrown - ballistic model

# Controller distance -> v_i, r_i

# Train neural network.
# Transformer encoder + distance in feedforward
# In: Serial data (torque sensor in motion), mass?, distance desired. Out: (r_i, v_i)
# Offline first.


