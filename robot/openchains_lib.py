import torch
import numpy as np

from robot.utils import *

## For Mass Computes
import platform
if platform.system() == "Linux":
	from utils.suppress_logging import suppress_output
	with suppress_output():
		import pybullet as p
elif platform.system() == "Windows":
	import pybullet as p
else:
	print('OS is not Linux or Windows!')

def compute_S_screw(A_screw, initialLinkFrames_from_base):
	"""_summary_
	Args:
			A_screw (torch.tensor): (n_joints, 6)
			initialLinkFrames_from_base (torch.tensor): (n_joints, 4, 4)
	"""
	S_screw = []
	for M, A in zip(initialLinkFrames_from_base, A_screw):
			S_temp = Adjoint(M.unsqueeze(0)).squeeze(0)@A.unsqueeze(-1)
			S_screw.append(S_temp)
	S_screw = torch.cat(S_screw, dim=-1).permute(1, 0)
	return S_screw # (n_joints, 6)

class Franka:
    def __init__(self, device='cpu', _robot_body_id=None):
        self.A_screw = torch.tensor([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=torch.float32).to(device) 
        
        
        self.M = torch.zeros(7, 4, 4).to(device) 
        self.M[0] = torch.tensor([[1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0.333], 
                [0, 0, 0, 1]]).to(device) 

        self.M[1] = torch.tensor([[1, 0, 0, 0], 
                [0, 0, 1, 0], 
                [0,-1, 0, 0], 
                [0, 0, 0, 1.0]]).to(device) 

        self.M[2] = torch.tensor([[1, 0, 0, 0], 
                [0, 0, -1, -0.316], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1]]).to(device) 

        self.M[3] = torch.tensor([[1, 0, 0, 0.0825], 
                [0, 0,-1, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1]]).to(device) 

        self.M[4] = torch.tensor([[1, 0, 0, -0.0825], 
                [0, 0, 1, 0.384], 
                [0,-1, 0, 0], 
                [0, 0, 0, 1]]).to(device) 
            
        self.M[5] = torch.tensor([[1, 0, 0, 0], 
                [0, 0,-1, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1.0]]).to(device) 

        self.M[6] = torch.tensor([[1, 0, 0, 0.088], 
                [0, 0, -1, 0], 
                [0, 1, 0, 0], 
                [0, 0, 0, 1]]).to(device) 

        self.initialLinkFrames_from_base = torch.zeros(7, 4, 4).to(device) 
        self.initialLinkFrames_from_base[0] = self.M[0]
        for i in range(1, 7):
            self.initialLinkFrames_from_base[i] = self.initialLinkFrames_from_base[i-1]@self.M[i]
        
        self.LLtoEE = torch.tensor(
                [[0.7071, 0.7071, 0, 0], 
                [-0.7071, 0.7071, 0, 0], 
                [0, 0, 1, 0.107], 
                [0, 0, 0, 1]]).to(device)
        
        self.EEtoLeftFinger = torch.tensor(
                [[1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0.0584], 
                [0, 0, 0, 1]]).to(device)
        
        self.EEtoRightFinger = torch.tensor(
                [[-1, 0, 0, 0], 
                [0, -1, 0, 0], 
                [0, 0, 1, 0.0584], 
                [0, 0, 0, 1]]).to(device)
        
        self.initialEEFrame = self.initialLinkFrames_from_base[-1]@self.LLtoEE
        
        self.S_screw = compute_S_screw(
                self.A_screw, 
                self.initialLinkFrames_from_base
        )
        
        self.inertias = torch.zeros(7, 6, 6).to(device)
        if _robot_body_id is not None:
            for i in range(7):
                results = p.getDynamicsInfo(_robot_body_id, i)
                m = results[0] 
                (ixx, iyy, izz) = results[2]
                self.inertias[i] = torch.tensor(
                        [
                                [ixx, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, iyy, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, izz, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, m, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, m, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, m]
                        ]
                ).to(device)
                
        #############################################
        #################### Limits #################
        #############################################
        # https://frankaemika.github.io/docs/control_parameters.html
        # self.JointPos_Limits = [
        #         [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], 
        #         [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.752, 2.8973]]
        
        offset = 0.02
        self.JointPos_Limits = [
        [-2.8973+offset, -1.7628+offset, -2.8973+offset, -3.0718+offset, -2.8973+offset, -0.0175+offset, -2.8973+offset], 
        [2.8973-offset, 1.7628-offset, 2.8973-offset, -0.0698-offset, 2.8973-offset, 3.752-offset, 2.8973-offset]]
        self.JointVel_Limits = [
                2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
        self.JointAcc_Limits = [
                15, 7.5, 10, 12.5, 15, 20, 20
        ]
        self.JointJer_Limits = [
                7500, 3750, 5000, 6250, 7500, 10000, 10000
        ]
        self.JointTor_Limits = [
                87, 87, 87, 87, 12, 12, 12
        ]
        
        self.CarVelocity_Limits = [
                2*2.5, 2*1.7 
        ]