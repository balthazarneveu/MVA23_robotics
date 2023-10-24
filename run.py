import example_robot_data as robex
import numpy as np
import pinocchio as pin
import time
from typing import Union, List, Tuple, Callable

from utils.meshcat_viewer_wrapper import MeshcatVisualizer, colors
from system import System
from pinocchio.utils import rotate
from world import add_obstacles_reduced, add_obstacles_hard, add_special_locations

def init_problem(reduced=False):
    
    """_summary_

    Args:
        reduced (bool, optional): Simplified problem. Defaults to False.

    Returns:
        _type_: _description_
    """
    robot = robex.load('ur5')
    if reduced:
        unlocks = [1,2]
        robot.model,[robot.visual_model,robot.collision_model]\
            = pin.buildReducedModel(robot.model,[robot.visual_model,robot.collision_model],
                                    [ i+1 for i in range(robot.nq) if i not in unlocks ],robot.q0)
        robot.data = robot.model.createData()
        robot.collision_data = robot.collision_model.createData()
        robot.visual_data = robot.visual_model.createData()
        robot.q0 = robot.q0[unlocks].copy()
    if reduced:
        add_obstacles_reduced(robot)
    else:
        add_obstacles_hard(robot)
    viz = MeshcatVisualizer(robot)
    return robot, viz


def solve(robot):
    system = System(robot)
    print(system.distance(np.array([[-np.pi+0.1], [0.]]), np.array([[np.pi-0.1]])))

def main(reduced=True):
    if reduced:
        q_i= np.deg2rad([-90., 40.])
        q_g= np.deg2rad([-79., 64.])
    else:
        q_i = np.array([1., -1.5, 2.1, -.5, -.5, 0])
        q_g = np.array([3., -1., 1, -.5, -.5, 0])
    robot, viz = init_problem(reduced=reduced)
    add_special_locations(
        robot,
        viz,
        [
            (q_i, "initial", "red"),
            (q_g, "goal", "green")
        ]
    )
    solve(robot)
    while True:
        time.sleep(0.1)
if __name__ == "__main__":
    main(reduced=False)
    
