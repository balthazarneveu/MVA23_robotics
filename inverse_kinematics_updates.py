import pinocchio as pin
import numpy as np
import time
from typing import List, Union, Optional, Tuple

def extract_dim(vec: np.ndarray, start:int, end:int) -> np.ndarray:
    return vec[start:end, ...]


def get_config_velocity_update_translation_with_proj(
    q: np.ndarray,
    rob: pin.RobotWrapper,
    index_object: int,
    o_M_target: pin.SE3,
    constraints: Tuple[int, int] =(0, 3),
    projector: np.ndarray=None,
    vq_prev: np.ndarray=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get a configuration update (velocity vq) to move the object to the target position.
    When projector and vq_prev are None, this function behaves like the first iteration
    Which means that Identity projects into the whole space without constraint.
    Warning: Calling this function assumes that Foward Kinematics data is up to date.
    `pin.framesForwardKinematics(rob.model, rob.data, q)`

    Args:
        q (np.ndarray): current configuration state 
            - (joint angles, basis position, etc...)
            - 15 scalars in case of the Tiago robot
        rob (pin.RobotWrapper): Robot instance.
        index_object (int): index of an object in the robot model (like effector or basis)
        o_M_target (pin.SE3): Target object position. SE(3) used here simply for its translation.
        constraints (Tuple[int, int], optional): Constrain only certain dimension of the target vector (from a to b)
            - Defaults to (0, 3) meaning no constraint.
            - (0,1) means constraining on the x axis.
            - (1,2) means constraining on the y axis.
            - (2,3) means constraining on the z axis.
            - (0,2) means constraining on the x & y axis.
        projector (np.ndarray, optional): Previous task projector matrix. Defaults to None.
            Required not to deviate from the previous task direction - only evolve in the orthogonal space.
        vq_prev (np.ndarray, optional): Previous task velocity update. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: vq, projector
    """
    if projector is None:
        projector = np.eye(rob.nv) # Identity matrix
    if vq_prev is None:
        vq_prev =   np.zeros(rob.nv) # Null vector
    
    # Current object location -> o_Mcurrent 
    o_Mcurrent = rob.data.oMf[index_object]
    
    # Compute the error between the current object and the target object -> obj_2_goal
    obj_2_goal = (o_M_target.translation - o_Mcurrent.translation)
    obj_2_goalC = extract_dim(obj_2_goal, *constraints) # constraint on some specific dimensions

    # Compute the jacobian of the object -> o_J_obj , constrained on specific dimensions.
    o_J_obj = pin.computeFrameJacobian(rob.model, rob.data, q, index_object, pin.LOCAL_WORLD_ALIGNED)
    o_J_objC = extract_dim(o_J_obj, *constraints) # + constraint on some specific dimensions

    
    new_error = (obj_2_goalC - o_J_objC @ vq_prev)
    
    J = o_J_objC @ projector
    Jinv = np.linalg.pinv(J) # pinv(J2@P1)
    
    vq = vq_prev + Jinv @ new_error
    # Compute updated projector. 

    new_proj = projector - Jinv @ J
    # Note the special case when projector is the identity matrix,
    # we get the same result as the first iteration.

    return vq, new_proj

