import hppfcl
import math
import numpy as np
import pinocchio as pin
import time
from tqdm import tqdm
from typing import Union, List, Tuple, Callable

from utils.meshcat_viewer_wrapper import MeshcatVisualizer, colors
from system import System
from pinocchio.utils import rotate
# SIMPLE CASE (reduced ~ 2D)
def add_obstacles_reduced(robot):
    def XYZRPYtoSE3(xyzrpy):
        rotate = pin.utils.rotate
        R = rotate('x',xyzrpy[3]) @ rotate('y',xyzrpy[4]) @ rotate('z',xyzrpy[5])
        p = np.array(xyzrpy[:3])
        return pin.SE3(R,p)
    # Capsule obstacles will be placed at these XYZ-RPY parameters
    oMobs = [ [ 0.40,  0.,  0.30, np.pi/2,0,0],
              [-0.08, -0.,  0.69, np.pi/2,0,0],
              [ 0.23, -0.,  0.04, np.pi/2, 0 ,0 ],
              [-0.32,  0., -0.08, np.pi/2, 0, 0]]
    rad,length = .1,0.4                                  # radius and length of capsules
    for i,xyzrpy in enumerate(oMobs):
        obs = pin.GeometryObject.CreateCapsule(rad,length)  # Pinocchio obstacle object
        obs.meshColor = np.array([ 1.0, 0.2, 0.2, 1.0 ])    # Don't forget me, otherwise I am transparent ...
        obs.name = "obs%d"%i                                # Set object name
        obs.parentJoint = 0                                 # Set object parent = 0 = universe
        obs.placement = XYZRPYtoSE3(xyzrpy)  # Set object placement wrt parent
        robot.collision_model.addGeometryObject(obs)  # Add object to collision model
        robot.visual_model   .addGeometryObject(obs)  # Add object to visual model

# HARD CASE (6 degrees of freedom)
def add_obstacles_hard(robot):
    def addCylinderToUniverse(name, radius, length, placement, color=colors.red):
        geom = pin.GeometryObject(
            name,
            0,
            hppfcl.Cylinder(radius, length),
            placement
        )
        new_id = robot.collision_model.addGeometryObject(geom)
        geom.meshColor = np.array(color)
        robot.visual_model.addGeometryObject(geom)
        
        for link_id in range(robot.model.nq):
            robot.collision_model.addCollisionPair(
                pin.CollisionPair(link_id, new_id)
            )
        return geom
    [robot.collision_model.removeGeometryObject(e.name) for e in 
     robot.collision_model.geometryObjects if e.name.startswith('world/')]

    # Add a red box in the viewer
    radius = 0.1
    length = 1.

    cylID = "world/cyl1"
    placement = pin.SE3(pin.SE3(rotate('z',np.pi/2), np.array([-0.5,0.4,0.5])))
    addCylinderToUniverse(cylID,radius,length,placement,color=[.7,.7,0.98,1])


    cylID = "world/cyl2"
    placement = pin.SE3(pin.SE3(rotate('z',np.pi/2), np.array([-0.5,-0.4,0.5])))
    addCylinderToUniverse(cylID,radius,length,placement,color=[.7,.7,0.98,1])

    cylID = "world/cyl3"
    placement = pin.SE3(pin.SE3(rotate('z',np.pi/2), np.array([-0.5,0.7,0.5])))
    addCylinderToUniverse(cylID,radius,length,placement,color=[.7,.7,0.98,1])


    cylID = "world/cyl4"
    placement = pin.SE3(pin.SE3(rotate('z',np.pi/2), np.array([-0.5,-0.7,0.5])))
    addCylinderToUniverse(cylID,radius,length,placement,color=[.7,.7,0.98,1])




def add_special_locations(robot, viz, q_list:List[Tuple[np.ndarray, str, str]], robot_effector=22):
    colors = {
        "red": [1., 0., 0., 1.],
        "green": [0., 1., 0., 1.]
    }
    for q, name, color in q_list:
        M = robot.framePlacement(q, robot_effector)
        name = f"world/{name}"
        ball_radius = 0.05
        viz.addSphere(name, ball_radius, colors.get(color, [1., 1., 1., 1.]))
        viz.applyConfiguration(name, M)
