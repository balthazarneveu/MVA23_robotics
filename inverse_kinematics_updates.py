import pinocchio as pin
import numpy as np
import time
from typing import List, Union, Optional, Tuple

def extract_dim(vec: np.ndarray, start:int, end:int) -> np.ndarray:
    return vec[start:end, ...]

def get_config_velocity_update_translation(
    q: np.ndarray,
    rob: pin.RobotWrapper,
    index_object: int,
    o_Mtarget: pin.SE3,
    constraints: Tuple[int, int] =(0, 3)
) -> Tuple[np.ndarray, np.ndarray]:
    """Make a step toward the constrained position for object
    `index_object`
    """
    o_Mcurrent = rob.data.oMf[index_object]
    obj_Jobj = pin.computeFrameJacobian(rob.model, rob.data, q, index_object, pin.WORLD)
    obj_JobjC = extract_dim(obj_Jobj, *constraints) # constrain over specific dimension
    obj_2_goal = (o_Mtarget.translation - o_Mcurrent.translation)
    obj_2_goalC = extract_dim(obj_2_goal, *constraints) # constrain over specific dimension
    obj_JobjC_inv = np.linalg.pinv(obj_JobjC)
    vq = obj_JobjC_inv @ obj_2_goalC
    projector = np.eye(rob.nv) - obj_JobjC_inv @ obj_JobjC
    return vq, projector


def get_config_velocity_update_translation_with_proj(
    q: np.ndarray,
    rob: pin.RobotWrapper,
    index_object: int,
    o_Mtarget: pin.SE3,
    constraints: Tuple[int, int] =(0, 3),
    projector: np.ndarray=None,
    vq_prev: np.ndarray=None
) -> Tuple[np.ndarray, np.ndarray]:
    
    o_Mcurrent = rob.data.oMf[index_object]
    obj_Jobj = pin.computeFrameJacobian(rob.model, rob.data, q, index_object, pin.WORLD)
    obj_JobjC = extract_dim(obj_Jobj, *constraints) # constrain over specific dimension
    obj_2_goal = (o_Mtarget.translation - o_Mcurrent.translation) # v2*
    obj_2_goalC = extract_dim(obj_2_goal, *constraints) # constrain over specific dimension
    
    new_error = (obj_2_goalC-obj_JobjC@vq_prev)
    jac_with_proj = obj_JobjC @ projector
    inv_jac_with_proj = np.linalg.pinv(jac_with_proj) # pinv(J2@P1)
    vq = vq_prev + inv_jac_with_proj @ new_error

    new_proj = projector - inv_jac_with_proj@jac_with_proj    
    return vq, new_proj

