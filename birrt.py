import numpy as np
from system import System
from typing import Callable
from utils.datastructures.storage import Storage
from utils.datastructures.pathtree import PathTree
from utils.datastructures.mtree import MTree
from tqdm import tqdm

FWD = "forward"
BWD = "backward"

class BiRRT():
    def __init__(
        self,
        system: System,
        node_max: int = 500000,
        iter_max: int = 1000000,
        l_min=.2,
        l_max=.5,
        steer_delta=.1,
    ):
        # Initialize attributes:
        self.system = system
        self.l_max = l_max
        self.l_min = l_min
        self.node_max = node_max
        self.iter_max = iter_max
        self.steer_delta = steer_delta

        # New: duplicate this attribute as dictionaries with two keys:
        # "forward" and "backward". See `solve` below.
        self._cached_dist_to_candidate = {}
        self.storage = {}
        self.pathtrees = {}
        self.trees = {}

        self.real_distance = self.system.distance
        self.it_trace = []

    def tree_distance(self, direction: str, q1_idx: int, q2_idx: int) -> float:
        """Compute the real distance
        
        Args:
            q1_idx (int): start node index (in the tree)
            q2_idx (int): end node index (in the tree)

        Returns:
            float: distance
        """
        # Adapt from RRT.distance
        # There is now a direction string to select the underlying tree,
        # either "forward" (from q_init) or "backward" (from q_goal).
        q1, q2 = self.storage[direction][q1_idx], self.storage[direction][q2_idx]
        if q1_idx == q2_idx:
            return 0.
        if q1_idx == -1:
            q1 = self._candidate
        if q2_idx == -1:
            q2 = self._candidate
        dist = self.real_distance(q1, q2)
        return dist

    def forward_distance(self, q1_idx, q2_idx):
        return self.tree_distance(FWD, q1_idx, q2_idx)

    def backward_distance(self, q1_idx, q2_idx):
        return self.tree_distance(BWD, q1_idx, q2_idx)

    def new_candidate(self):
        """Sample a candidate in the free space.
        """
        q = self.system.random_config(free=True)
        self._candidate = q
        for direction in (FWD, BWD):
            self._cached_dist_to_candidate[direction] = {}
        return q
    
    def reset(self):
        # Reset internal datastructures
        for direction in (FWD, BWD):
            self._cached_dist_to_candidate[direction] = {}
            self.storage[direction] = Storage(self.node_max, self.system.nq)
            self.pathtrees[direction] = PathTree(self.storage[direction])
        self.trees = {
            FWD: MTree(self.forward_distance),
            BWD: MTree(self.backward_distance),
        }
        
    def solve(self, qi: np.ndarray, qg: np.ndarray= None):
        """Run the BiRRT algorithm
        
        Args:
            qi (np.ndarray): initial configuration
            qg (np.ndarray, optional): goal/target configuration. 
        """
        assert qg is not None
        # Reset internal datastructures
        self.reset()
        # Root of the forward tree = initial state qi
        qi_idx = self.storage[FWD].add_point(qi)
        self.trees[FWD].add_point(qi_idx)
        # Root of the backward tree = goal state qg
        qg_idx = self.storage[BWD].add_point(qg)
        self.trees[BWD].add_point(qg_idx)
        self.found = False
        iterator = range(self.iter_max)
        for iter in tqdm(iterator):
            if self.found:
                break
            # Standard random sampling
            q_new = self.new_candidate()
            # q_new = np.array([0., 0.]) # FOR TESTING
            q_new_idx = -1
            # Find closest neighboor to q_new in the NNtree
            extension_nodes = {FWD: None, BWD:None}
            for direction in [FWD, BWD]:
                if self.found:
                    break
                q_near_idx, _d_fwd = self.trees[direction].nearest_neighbour(q_new_idx) # q_new_idx = -1
                # if direction==FWD:
                #     print(q_near_idx)
                success, q_prox = self.system.is_free_path(
                    self.storage[direction].data[q_near_idx],
                    q_new,
                    l_min=self.l_min,
                    l_max=self.l_max,
                    eps=self.steer_delta
                )
                if not success:
                    # print(f"{direction}: FAILED TO CREATE A NEW BRANCH STEERING IN THE q_new direction")
                    self.it_trace.append(0)
                    continue
                self.it_trace.append(1)
                # Add the q_prox point as new node in data structure
                q_prox_idx = self.storage[direction].add_point(q_prox) # add the q_prox point to the storage, get new index
                self.trees[direction].add_point(q_prox_idx) # add the node q_prox to the graph
                self.pathtrees[direction].update_link(q_prox_idx, q_near_idx)
                
                self.system.display_edge(
                    self.storage[direction][q_near_idx],
                    self.storage[direction][q_prox_idx], 
                    color=[1., 1., 1., 1.] if direction == FWD else [1., 0.5, 1., 1.]
                )
                extension_nodes[direction] = q_prox_idx
            DIR_LIST = [FWD, BWD]
            for dir_idx, dir in enumerate(DIR_LIST):
                if self.found:
                    break
                oppo_dir = DIR_LIST[(1-dir_idx)%2]
                assert oppo_dir!=dir
                q_prox_idx = extension_nodes[dir]
                if q_prox_idx is None:
                    continue
                else:
                    newest = self.storage[dir].data[q_prox_idx] # latest candidate
                    self._candidate = newest
                    nearest_idx_oppo, _ = self.trees[oppo_dir].nearest_neighbour(-1) # find nearest neighbor in the opposite tree
                    nearest_oppo = self.storage[oppo_dir].data[nearest_idx_oppo]
                    success, _ = self.system.is_free_path(
                        newest,
                        nearest_oppo,
                        l_min=None,
                        l_max=None,
                        eps=0.1 #@TODO: WARNING ON ACCURACY!!!! We can simply loop!
                    )
                    if success:
                        print(f"DONE! {dir} Bridge found between {newest} {nearest_oppo}")
                        bridge_index = self.storage[dir].add_point(nearest_oppo) #ADD THE BRIDGE! add the nearest_oppo to the current graph
                        self.pathtrees[dir].update_link(bridge_index, q_prox_idx) #Update the bridge
                        self.found = True # WE HAVE A PATH!
                        self.system.display_edge(newest, nearest_oppo, radius=0.015, color=[0., 0., 1., 1.])
                        break
        return self.found
    def get_path(self):
        assert self.found
        forward_path = self.pathtrees["forward"].get_path()
        backward_path = self.pathtrees["backward"].get_path()
        return np.concatenate([forward_path, backward_path[::-1]])