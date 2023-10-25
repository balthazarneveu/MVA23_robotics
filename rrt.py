import numpy as np
from system import System
from typing import Callable
from utils.datastructures.storage import Storage
from utils.datastructures.pathtree import PathTree
from utils.datastructures.mtree import MTree
from tqdm import tqdm

class RRT():
    """
    Can be splited into RRT base because different rrt
    have factorisable logic
    """
    def __init__(
        self,
        system: System,
        node_max: int = 500000,
        iter_max: int = 1000000,
        N_bias: int=10,
        l_min: float =.2,
        l_max: float =.5,
        steer_delta: float=.1,
    ) -> None:
        '''Class implementing the RRT (Rapidly exploring Random Tree) algorithm.
        Creates a tree (a special kind of graph)
        starting from the source
        and progressively sampling free paths until reaching the target.

        Args:
            system (System): contains the Robotwrapper and a few helpers to perform computations in the configuration space.
            node_max (int, optional): maximum number of nodes in the tree. Defaults to 500000.
            iter_max (int, optional): maximum number of iterations. Defaults to 1000000.
            N_bias (int, optional): Every N_bias iterations, replace the random candidate by the goal itself. Defaults to 10.
            l_min (float, optional): minimal length granted of the new edges added to the graph. Defaults to .2.
            l_max (float, optional): maximal length granted of the new edges added to the graph. Defaults to .5.
            steer_delta (float, optional): _description_. Defaults to .1.
        '''
        self.system = system
        # params
        self.l_max = l_max
        self.l_min = l_min
        self.N_bias = N_bias
        self.node_max = node_max
        self.iter_max = iter_max
        self.steer_delta = steer_delta
        # intern
        self.NNtree = None
        self.storage = None
        self.pathtree = None
        # The distance function will be called on N, dim object
        self.real_distance = self.system.distance
        # Internal for computational_opti in calculating distance
        self._candidate = None
        self._goal = None
        self._cached_dist_to_candidate = {}
        self._cached_dist_to_goal = {}

    def distance(self, q1_idx: int, q2_idx: int) -> float:
        """Compute the real distance
        
        Args:
            q1_idx (int): start node index (in the tree)
            q2_idx (int): end node index (in the tree)

        Returns:
            float: distance
        """
        # Along the tree of the feasible path between q1 & q2
        # (not the naïve straight line euclian distance in the configuration space)

        if isinstance(q2_idx, int):
            if q1_idx == q2_idx:
                return 0. # requesting twice the same node -> 0
            if q1_idx == -1 or q2_idx == -1:
                if q2_idx == -1:
                    q1_idx, q2_idx = q2_idx, q1_idx
                if q2_idx not in self._cached_dist_to_candidate:
                    self._cached_dist_to_candidate[q2_idx] = self.real_distance(
                        self._candidate, self.storage[q2_idx]
                    )
                return self._cached_dist_to_candidate[q2_idx]
            if q1_idx == -2 or q2_idx == -2:
                if q2_idx == -2:
                    q1_idx, q2_idx = q2_idx, q1_idx
                if q2_idx not in self._cached_dist_to_goal:
                    self._cached_dist_to_goal[q2_idx] = self.real_distance(
                        self._goal, self.storage[q2_idx]
                    )
                return self._cached_dist_to_goal[q2_idx]
            return self.real_distance(self.storage[q1_idx], self.storage[q2_idx])
        if q1_idx == -1:
            q = self._candidate
        elif q1_idx == -2:
            q = self._goal
        else:
            q = self.storage[q1_idx]
        return self.real_distance(q, self.storage[q2_idx])

    def new_candidate(self) -> np.ndarray:
        """Sample a candidate in the free space. 
        Init self._candidate, reset _cached_dist_to_candidate
        """
        q = self.system.random_config(free=True)
        self._candidate = q
        self._cached_dist_to_candidate = {}
        return q

    def solve(self, qi: np.ndarray, validate: Callable, qg: np.ndarray=None) -> bool:
        """Run the algorithm

        Args:
            qi (np.ndarray): initial configuration
            validate (Callable): function which returns True if you reach your target
            qg (np.ndarray, optional): goal/target configuration. 
            Defaults to None (validate may be enough to contain the fact of reaching the target).
            Goal is required when using the N_bias

        Returns:
            bool: sucess
        """
        self.system.reset()
        self._goal = qg
        
        # Reset internal datastructures
        self.storage = Storage(self.node_max, self.system.nq)
        self.pathtree = PathTree(self.storage) # to store the tree of the path only
        self.NNtree = MTree(self.distance) # to store the exploration tree
        # Root of the tree = initial state qi
        qi_idx = self.storage.add_point(qi)
        self.NNtree.add_point(qi_idx)
        self.it_trace = []

        found = False
        iterator = range(self.iter_max)
        for i in tqdm(iterator):
            # Sample new candidate
            if i % self.N_bias == 0: 
                # Every N_bias times, let's force the candadidate to the goal!
                q_new = self._goal
                q_new_idx = -2
            else:
                # Standard random sampling
                q_new = self.new_candidate()
                q_new_idx = -1

            # Find closest neighboor to q_new in the NNtree
            q_near_idx, d = self.NNtree.nearest_neighbour(q_new_idx) # q_new_idx = -1 (or -2 in case of bias in favor of the target).
            
            # Steer from it toward the new checking for colision, with  a max length of lmax.
            # q_prox is a single feasible state (not the whole segment)
            success, q_prox = self.system.is_free_path(
                self.storage.data[q_near_idx],
                q_new,
                l_min=self.l_min,
                l_max=self.l_max,
                eps=self.steer_delta
            )

            if not success:
                self.it_trace.append(0)
                continue
            self.it_trace.append(1)
            
            # Add the q_prox point as new node in data structure
            q_prox_idx = self.storage.add_point(q_prox) # add the q_prox point to the storage, get new index
            self.NNtree.add_point(q_prox_idx) # add the node q_prox to the graph
            self.pathtree.update_link(q_prox_idx, q_near_idx)
            self.system.display_edge(self.storage[q_near_idx], self.storage[q_prox_idx])

            # Test if it reach the goal
            if validate(q_prox):
                q_g_idx = self.storage.add_point(q_prox)
                self.NNtree.add_point(q_g_idx)
                self.pathtree.update_link(q_g_idx, q_prox_idx)
                found = True
                break
        self.iter_done = i + 1
        self.found = found
        return found

    def get_path(self, q_g):
        assert self.found
        path = self.pathtree.get_path()
        return np.concatenate([path, q_g[None, :]])