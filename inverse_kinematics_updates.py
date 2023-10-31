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
    o_Mtarget: pin.SE3,
    constraints: Tuple[int, int] =(0, 3),
    projector: np.ndarray=None,
    vq_prev: np.ndarray=None
) -> Tuple[np.ndarray, np.ndarray]:
    
    if projector is None:
        projector = np.eye(rob.nv)
    if vq_prev is None:
        vq_prev =   np.zeros(rob.nv)
    
    # Current object location -> o_Mcurrent 
    o_Mcurrent = rob.data.oMf[index_object]
    
    # Compute the error between the current object and the target object -> obj_2_goal
    obj_2_goal = (o_Mtarget.translation - o_Mcurrent.translation)

    # Compute the jacobian of the object -> o_J_obj
    o_J_obj = pin.computeFrameJacobian(rob.model, rob.data, q, index_object, pin.WORLD)
    o_J_objC = extract_dim(o_J_obj, *constraints) # constraint over specific dimension
    
    obj_2_goalC = extract_dim(obj_2_goal, *constraints) # constrain over specific dimension
    
    new_error = (obj_2_goalC- o_J_objC @ vq_prev)
    
    jac_with_proj = o_J_objC @ projector
    inv_jac_with_proj = np.linalg.pinv(jac_with_proj) # pinv(J2@P1)
    
    vq = vq_prev + inv_jac_with_proj @ new_error

    new_proj = projector - inv_jac_with_proj@jac_with_proj    
    return vq, new_proj

