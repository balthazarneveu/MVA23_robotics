import example_robot_data as robex
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


def add_obstacles(robot, collision_model, visual_model):
    def addCylinderToUniverse(name, radius, length, placement, color=colors.red):
        geom = pin.GeometryObject(
            name,
            0,
            hppfcl.Cylinder(radius, length),
            placement
        )
        new_id = collision_model.addGeometryObject(geom)
        geom.meshColor = np.array(color)
        visual_model.addGeometryObject(geom)
        
        for link_id in range(robot.model.nq):
            collision_model.addCollisionPair(
                pin.CollisionPair(link_id, new_id)
            )
        return geom
    [collision_model.removeGeometryObject(e.name) for e in 
     collision_model.geometryObjects if e.name.startswith('world/')]

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


def init_problem_obstacles():
    robot = robex.load('ur5')
    collision_model = robot.collision_model
    visual_model = robot.visual_model
    add_obstacles(robot, collision_model, visual_model)
    viz = MeshcatVisualizer(robot)
    return robot, viz


def add_special_locations(robot, q_list:List[Tuple[np.ndarray, str, str]], robot_effector=22):
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

def solve(robot):
    system = System(robot)
    print(system.distance(np.array([[-np.pi+0.1], [0.]]), np.array([[np.pi-0.1]])))


if __name__ == "__main__":
    q_i = np.array([1., -1.5, 2.1, -.5, -.5, 0])
    q_g = np.array([3., -1., 1, -.5, -.5, 0])
    robot, viz = init_problem_obstacles()
    add_special_locations(
        robot,
        [
            (q_i, "initial", "red"),
            (q_g, "goal", "green")
        ]
         
    )
    solve(robot)
    while True:
        time.sleep(0.1)
