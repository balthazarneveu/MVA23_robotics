import numpy as np
from typing import Union, List, Tuple, Callable

from utils.datastructures.mtree import MTree
from utils.collision_wrapper import CollisionWrapper
from pinocchio.robot_wrapper import RobotWrapper
import time
import pinocchio as pin


class System():
    def __init__(self, robot: RobotWrapper, dof=6):
        self.robot = robot
        robot.gmodel = robot.collision_model
        self.display_edge_count = 0
        self.colwrap = CollisionWrapper(robot)  # For collision checking
        self.nq = self.robot.nq
        self.display_count = 0
        self.dof = dof # degrees of freedom
    
    @staticmethod
    def distance(q1: np.ndarray, q2: np.ndarray) -> Union[float, np.array]:
        """
        Distance between q1 and q2 in the config space
        supports batch of config.
        """
        if len(q2.shape) > len(q1.shape):
            q1 = q1[None, ...]
        
        e = np.mod(np.abs(q1 - q2), 2 * np.pi)
        e[e > np.pi] = 2 * np.pi - e[e > np.pi]
        return np.linalg.norm(e, axis=-1)

    def random_config(self, free=True) -> np.ndarray:
        """
        Return a random configuration which is not in collision if free=True
        Does not set the robot in this configuration
        """
        q = 2 * np.pi * np.random.rand(self.dof) - np.pi  # [-pi, pi]^6
        if not free:
            return q
        while self.is_colliding(q):
            q = 2 * np.pi * np.random.rand(self.dof) - np.pi
        return q

    def is_colliding(self, q: np.array) -> bool:
        """
        Uses CollisionWrapper to decide if a configuration is in collision
        """
        # @TODO: are self collisions handled here
        self.colwrap.computeCollisions(q)
        collisions = self.colwrap.getCollisionList()
        return (len(collisions) > 0)

    def get_path(self, q1: np.array, q2: np.array, l_min=None, l_max=None, eps: float=0.2) -> np.ndarray:
        """Generate a straight continuous path (in the config space) between q1 and q2
        with precision eps (in radians) between q1 and q2 
        If l_min or l_max is mentioned, extrapolate or cut the path such
            - if l_min > d , extrapolate path_length=l_min
            - if l_max < d, cut path_length=l_max
        otherwise, if l_min<d(q1,q2)<l_max , path_length=d(q1,q2)
        
        ```

                         cut l_max <d         standard    extrapolate l_min>d
                              |                  |             |
                              v                  v             v
        [q1, q2] -> [q1, ... , ... ,... , ... ,  q2]    ........
                        <->
                        eps
        ```
        """
        q1 = np.mod(q1 + np.pi, 2 * np.pi) - np.pi
        q2 = np.mod(q2 + np.pi, 2 * np.pi) - np.pi

        diff = q2 - q1
        query = np.abs(diff) > np.pi
        q2[query] = q2[query] - np.sign(diff[query]) * 2 * np.pi

        d = self.distance(q1, q2)
        if d < eps:
            # precision is higher than the provided segment
            # return the original segment (q1 & q2 endpoints, nothing in-between)
            return np.stack([q1, q2], axis=0)
        
        if l_min is not None or l_max is not None:
            new_d = np.clip(d, l_min, l_max)
        else:
            new_d = d
            
        N = int(new_d / eps + 2)

        return np.linspace(q1, q1 + (q2 - q1) * new_d / d, N)
        
    def is_free_path(self, q1: np.ndarray, q2: np.ndarray, l_min=0.2, l_max=1., eps=0.2) -> Tuple[bool, np.ndarray]:
        """
        Create a path and check collision to return the last
        non-colliding configuration.
        Return X, q  -> q is the last state before collision, 
        in a max range of length l_max (steer a unitary vector in the diretion of q2)
        if no obstacle in the way, we shall have q= q1 + l_max*(q2-q1)/|q2-q1|
        where X is a boolean which state if the steering has worked.
        We require at least l_min must be cover without collision to validate the path.
        """
        q_path = self.get_path(q1, q2, l_min, l_max, eps)
        N = len(q_path)
        N_min = N - 1 if l_min is None else min(N - 1, int(l_min / eps))
        for i in range(N):
            if self.is_colliding(q_path[i]):
                break
        if i < N_min:
            return False, None
        if i == N - 1:
            return True, q_path[-1] # return the last point of the path
        return True, q_path[i - 1] # return the last point before collision
    
    # --------------------------------------------------------------------------------
    # Visualization functions
    # --------------------------------------------------------------------------------
    def add_visualizer(self, viz):
        self.viz = viz # Avoids using global variables

    def reset(self):
        """
        Reset the system visualization
        """
        for i in range(self.display_count):
            self.viz.delete(f"world/sph{i}")
            self.viz.delete(f"world/cil{i}")
        self.display_count = 0
    
    def display_edge(self, q1: np.ndarray, q2: np.ndarray, radius=0.01, color=[0., 1., 0., 1]):
        """Visualize an edge"""
        M1 = self.robot.framePlacement(q1, 22)  # Placement of the end effector tip.
        M2 = self.robot.framePlacement(q2, 22)  # Placement of the end effector tip.
        middle = .5 * (M1.translation + M2.translation)
        direction = M2.translation - M1.translation
        length = np.linalg.norm(direction)
        dire = direction / length
        orth = np.cross(dire, np.array([0, 0, 1]))
        orth2 = np.cross(dire, orth)
        Mcyl = pin.SE3(np.stack([orth2, dire, orth], axis=1), middle)
        name = f"world/sph{self.display_count}"
        self.viz.addSphere(name, radius, [1.,0.,0.,1])
        self.viz.applyConfiguration(name,M2)
        name = f"world/cil{self.display_count}"
        self.viz.addCylinder(name, length, radius / 4, color)
        self.viz.applyConfiguration(name, Mcyl)
        self.display_count +=1
        
    def display_motion(self, qs: np.ndarray, step=1e-1):
        # Given a point path display the smooth movement
        for i in range(len(qs) - 1):
            for q in self.get_path(qs[i], qs[i+1])[:-1]:
                self.viz.display(q)
                time.sleep(step)
        self.viz.display(qs[-1])
