import numpy as np
import pybullet as p

# def get_joint_state(robot_id, joint_indices):
#     """Get joint positions and velocities."""
#     joint_states = p.getJointStates(robot_id, joint_indices)
#     positions = np.array([state[0] for state in joint_states])
#     velocities = np.array([state[1] for state in joint_states])
#     return positions, velocities

GRAVITY = 9.81

def vel_ballistic_model(delta_x, delta_z, angle_to_base):
    numerator = GRAVITY * delta_x**2
    denominator = 2 * np.cos(angle_to_base)**2 * (delta_x * np.tan(angle_to_base) - delta_z)

    vel_ballistic_model = np.sqrt(numerator / denominator)
    return vel_ballistic_model

def circular_trajectory(velocity, center_pose, dt=0.02, radius=0.35):
    angular_velocity = velocity / radius

    # The angle changes over time (θ = ω * t)
    dtheta = angular_velocity * dt

    # Literal angle range
    start, end = -np.pi - np.pi/6, - 3*np.pi/4
    num_steps = int((end - start) // dtheta) + 1  # Calculate the number of steps
    ths = np.linspace(start, start + dtheta * (num_steps - 1), num_steps)

    x = radius * np.sin(ths) + center_pose[1]
    y = np.zeros_like(ths) + center_pose[0]
    z = radius * np.cos(ths) + center_pose[2]
    
    # Calculate the velocity at each point on the trajectory
    velocity_x = -velocity * np.sin(ths)
    velocity_y = np.zeros_like(ths)
    velocity_z = velocity * np.cos(ths)

    return list(zip(x, y, z, ths, velocity_x, velocity_y, velocity_z))
    