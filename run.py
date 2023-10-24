import example_robot_data as robex
import numpy as np
import pinocchio as pin
import time
from typing import Union, List, Tuple, Callable

from utils.meshcat_viewer_wrapper import MeshcatVisualizer, colors
from system import System
from pinocchio.utils import rotate
from world import add_obstacles_reduced, add_obstacles_hard, add_special_locations
from pinocchio.robot_wrapper import RobotWrapper
from rrt import RRT

def initialize_problem(reduced=False) -> Tuple[RobotWrapper, MeshcatVisualizer]:
    """Initialize a world with a ur5 robot

    Args:
        reduced (bool, optional): Simplified problem. Defaults to False.

    Returns:
        robot, viz
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


def solve(robot, viz, q_i, q_g, reduced=False):
    system = System(robot, dof=2 if reduced else 6)
    system.add_visualizer(viz)
    system.display_motion([q_i, q_g])
    # print(system.distance(np.array([[-np.pi+0.1], [0.]]), np.array([[np.pi-0.1]])))
    return system

def main(reduced=True):
    if reduced:
        q_i= np.deg2rad([90., 0.])
        # q_i= np.deg2rad([-90., 40.])
        q_g= np.deg2rad([-79., 64.])
    else:
        q_i = np.array([1., -1.5, 2.1, -.5, -.5, 0])
        q_g = np.array([3., -1., 1, -.5, -.5, 0])
    robot, viz = initialize_problem(reduced=reduced)
    add_special_locations(
        robot,
        viz,
        [
            (q_i, "initial", "red"),
            (q_g, "goal", "green")
        ]
    )
    system = solve(robot, viz, q_i, q_g, reduced=reduced)
    if False:
        system.display_edge(q_i, q_g)
        while True:
            time.sleep(0.5)
            system.display_motion([q_i, q_g], step=0.5)
    
    rrt = RRT(
        system,
        N_bias=20,
        l_min=0.2,
        l_max=0.5,
        steer_delta=0.1,
    )
    eps_final = .1
    def validation(key):
        vec = robot.framePlacement(key, 22).translation - robot.framePlacement(q_g, 22).translation
        return (float(np.linalg.norm(vec)) < eps_final)

    rrt.solve(q_i, validation, qg=q_g)
    while True:
        time.sleep(0.5)
        system.display_motion(rrt.get_path(q_g))
if __name__ == "__main__":
    main(reduced=True)
    
