import mengine as m
import numpy as np
import os
import pybullet as p

# class ObjectThrown:
#     def __init__(self, pose):
#         # Define the mass, inertia tensor (xx, yy, zz, xy, xz, yz), and the link size for base link
#         mass_base = 0.02
#         inertia_base = [0.0002, 0.0002, 0.0002, 0.0, 0.0, 0.0]  # [ixx, iyy, izz, ixy, ixz, iyz]
#         collision_shape_base = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.04])
#         visual_shape_base = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.04], rgbaColor=[0, 0, 1, 1])

#         # Define the mass, inertia tensor (xx, yy, zz, xy, xz, yz), and the link size for top bar
#         mass_top = 0.01
#         inertia_top = [0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0]  # [ixx, iyy, izz, ixy, ixz, iyz]
#         collision_shape_top = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.025, 0.015])
#         visual_shape_top = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.025, 0.015], rgbaColor=[1, 0, 0, 1])

#         # Add links to the multi-body (base and top bars) with inertia properties
#         base_link_id = p.createMultiBody(
#             baseMass=mass_base,
#             baseCollisionShapeIndex=collision_shape_base,
#             baseVisualShapeIndex=visual_shape_base,
#             basePosition=pose + np.array([0, 0, 0.04]),
#             baseOrientation=[0, 0, 0, 1],  # Quaternion for no rotation
#             baseInertialFramePosition=pose + np.array([0, 0.08, 0.08]),
#             baseInertialFrameOrientation=[0, 0, 0, 1],
#             baseInertiaTensor=inertia_base
#         )

#         top_bar_link_id = p.createMultiBody(
#             baseMass=mass_top,
#             baseCollisionShapeIndex=collision_shape_top,
#             baseVisualShapeIndex=visual_shape_top,
#             basePosition=pose + np.array([0, 0.08, 0.08]),
#             baseOrientation=[0, 0, 0, 1],  # Quaternion for no rotation
#             baseInertialFramePosition=pose + np.array([0, 0.08, 0.08]),
#             baseInertialFrameOrientation=[0, 0, 0, 1],
#             baseInertiaTensor=inertia_top,
#             linkMasses=[0],  # No mass for child links
#             linkCollisionShapeIndices=[-1],  # No collision for child links
#             linkVisualShapeIndices=[-1],  # No visual for child links
#         )

#         # Set the joint between the two links (fixed joint)
#         p.createConstraint(
#             parentBodyUniqueId=base_link_id,
#             parentLinkIndex=-1,  # Parent link is the base link
#             childBodyUniqueId=top_bar_link_id,
#             childLinkIndex=-1,  # Child link is the top bar link
#             jointType=p.JOINT_FIXED,
#             jointAxis=[0, 0, 0],  # Fixed joint
#             parentFramePosition=[0, 0, 0.08],  # Position where the links connect
#             childFramePosition=[0, 0, 0]
#         )

class ThrowerSetup:
    def __init__(self, env, base_pose: np.ndarray = None, render: bool = False):
        self.env = env

        self.base_pose = np.array([0.,0.,0.])

        if base_pose:
            self.base_pose = base_pose
            
        robot_base = self.base_pose + np.array([0.5, 0, 0.76]) # Literal for over table
        self.robot = m.Robot.Panda(position=robot_base, orientation=m.get_quaternion([0,0,np.pi]))

        self.start_joints = np.array([0., 0., 0., -2.8, 0., 7*np.pi/7, -5*np.pi/6])

        self.robot.control(self.start_joints, set_instantly=True)

        # Link Thrower 
        self.thrower_link = 3
        rel_pos, _ = self.robot.global_to_local_coordinate_frame(
                            self.robot.get_link_pos_orient(self.robot.end_effector)[0], 
                            link=self.thrower_link, rotation_only=False
                            )
        self.thrower_radius = np.linalg.norm(rel_pos)

        # self.robot.motor_gains = 0.1

        table_base = self.base_pose + np.array([-0.2, 0., 0.])
        table = m.URDF(filename=os.path.join(m.directory, 'table', 'table.urdf'), static=True, position=table_base,
               orientation=[0, 0, 0, 1])

        table_base = self.base_pose + np.array([-1.7, 0., 0.])
        table_2 = m.URDF(filename=os.path.join(m.directory, 'table', 'table.urdf'), static=True, position=table_base,
               orientation=[0, 0, 0, 1])

        self.array_objects = []
        ARRAY_OBJ_SHAPE = (2, 5)
        ARRAY_OBJ_SEPARATION = 0.15 # 14 cm 
        
        self.overtable_pose = self.base_pose + np.array([0.0, 0.0, 0.76])
        corner_obj_pose = self.overtable_pose + np.array([-0.09, -0.32, 0.0])

        for i in range(ARRAY_OBJ_SHAPE[0]):
            for j in range(ARRAY_OBJ_SHAPE[1]):
                object_pose = corner_obj_pose + np.array([ARRAY_OBJ_SEPARATION*i, ARRAY_OBJ_SEPARATION*j, 0.0])

                # mass = np.random.uniform(0.004, 0.05)
                # obj = m.Shape(m.Mesh(filename='./objects/hammer/meshes/model.obj', scale=[0.48, 0.48, 0.48]), static=False, mass=0.250, position=object_pose, orientation=m.get_quaternion([0,0,np.pi]), rgba=None, visual=True, collision=True)
                # obj = ObjectThrown(object_pose)
                obj = m.URDF(filename='./objects/t_shape_object/t_shape_object.urdf', static=False, position=object_pose, orientation=m.get_quaternion([0,-np.pi/2,0]))
                obj.set_whole_body_frictions(lateral_friction=2000, spinning_friction=2000, rolling_friction=2000)

                for link in obj.all_joints:
                    p.changeDynamics(obj.body, link,     
                                linearDamping=0.01,
                                angularDamping=0.01,
                                physicsClientId=obj.id)

                self.array_objects.append(obj)

        # if render:
        #     self.env.set_gui_camera(look_at_pos=robot_base, distance=1.5, pitch=-30)



if __name__ == '__main__':
    render = True
    env = m.Env(render=render)
    ground = m.Ground()
    ts = ThrowerSetup(env, render=render)
    m.step_simulation(steps=1000, realtime=True)
