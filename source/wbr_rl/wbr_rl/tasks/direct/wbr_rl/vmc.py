import numpy as np
import math


class vmcsolver:
    """五连杆VMC求解器"""
    
    def __init__(self):
        """
        初始化VMC求解器
        
        Args:
            motor_distance: 两个电机之间的距离
            L1: 第一连杆长度
            L2: 第二连杆长度
        """
        # 连杆参数
        self.VMC_MotorDistance = 0
        self.VMC_HalfMotorDistance = 0
        self.VMC_L1 = 0.25
        self.VMC_L2 = 0.33
        
        # 雅可比矩阵
        self.J_mat = np.zeros((2, 2), dtype=np.float32)
        self.JT_mat = np.zeros((2, 2), dtype=np.float32)
        self.JT_inv_mat = np.zeros((2, 2), dtype=np.float32)
        
        # 关节电机弧度
        self.phi1 = 0.0
        self.phi4 = 0.0
        
        # 极限值
        self.phi1_max = math.pi
        self.phi4_max = math.pi / 2
        self.phi1_min = math.pi / 2
        self.phi4_min = 0.0
        
        # 倒立摆长度
        self.PendulumLength = 0.0
        # 倒立摆角度
        self.PendulumRadian = math.pi / 2
        # 倒立摆坐标
        self.CoorC = np.zeros(2, dtype=np.float32)
        # 第二象限节点坐标
        self.CoorB = np.zeros(2, dtype=np.float32)
        self.U2 = 0.0
        # 第三象限节点坐标
        self.CoorD = np.zeros(2, dtype=np.float32)
        self.U3 = 0.0
    
    def Resolve(self, phi1_fdb: float, phi4_fdb: float):
        """
        正运动学求解：从关节角度计算倒立摆位置和雅可比矩阵
        
        Args:
            phi1_fdb: rear (大于90度)
            phi4_fdb: front (小于90度)
        """
        self.phi1 = phi1_fdb
        self.phi4 = phi4_fdb
        
        SIN1 = math.sin(self.phi1)
        COS1 = math.cos(self.phi1)
        SIN4 = math.sin(self.phi4)
        COS4 = math.cos(self.phi4)
        
        # 计算BD向量
        xdb = self.VMC_MotorDistance + self.VMC_L1 * (COS4 - COS1)
        ydb = self.VMC_L1 * (SIN4 - SIN1)
        
        A0 = 2 * self.VMC_L2 * xdb
        B0 = 2 * self.VMC_L2 * ydb
        C0 = xdb * xdb + ydb * ydb
        
        # 计算u2
        u2t = math.atan2((B0 + math.sqrt(A0 * A0 + B0 * B0 - C0 * C0)), (A0 + C0))
        self.U2 = 2.0 * u2t
        
        # 计算B点坐标
        self.CoorB[0] = self.VMC_L1 * COS1 - self.VMC_HalfMotorDistance
        self.CoorB[1] = self.VMC_L1 * SIN1
        
        # 计算C点坐标（倒立摆端点）
        self.CoorC[0] = self.CoorB[0] + self.VMC_L2 * math.cos(self.U2)
        self.CoorC[1] = self.CoorB[1] + self.VMC_L2 * math.sin(self.U2)
        
        # 计算D点坐标
        self.CoorD[0] = self.VMC_L1 * COS4 + self.VMC_HalfMotorDistance
        self.CoorD[1] = self.VMC_L1 * SIN4
        
        # 计算u3
        u3t = math.atan2((self.CoorD[1] - self.CoorC[1]), (self.CoorD[0] - self.CoorC[0]))
        self.U3 = math.pi + u3t
        
        # 计算倒立摆长度和角度
        self.PendulumRadian = math.atan2(self.CoorC[1], self.CoorC[0])
        self.PendulumLength = math.sqrt(self.CoorC[0] * self.CoorC[0] + self.CoorC[1] * self.CoorC[1])
        
        # 计算雅可比矩阵及其转置和逆
        sin32 = math.sin(self.U3 - self.U2)
        sin12 = math.sin(self.phi1 - self.U2)
        sin34 = math.sin(self.U3 - self.phi4)
        cos03 = math.cos(self.PendulumRadian - self.U3)
        cos02 = math.cos(self.PendulumRadian - self.U2)
        sin03 = math.sin(self.PendulumRadian - self.U3)
        sin02 = math.sin(self.PendulumRadian - self.U2)
        
        # 雅可比矩阵 J
        self.J_mat[0, 0] = self.VMC_L1 * sin03 * sin12 / sin32
        self.J_mat[0, 1] = self.VMC_L1 * sin02 * sin34 / sin32
        self.J_mat[1, 0] = self.VMC_L1 * cos03 * sin12 / (sin32 * self.PendulumLength)
        self.J_mat[1, 1] = self.VMC_L1 * cos02 * sin34 / (sin32 * self.PendulumLength)
        
        # 雅可比矩阵转置 J^T
        self.JT_mat[0, 0] = self.VMC_L1 * sin03 * sin12 / sin32
        self.JT_mat[0, 1] = self.VMC_L1 * cos03 * sin12 / (sin32 * self.PendulumLength)
        self.JT_mat[1, 0] = self.VMC_L1 * sin02 * sin34 / sin32
        self.JT_mat[1, 1] = self.VMC_L1 * cos02 * sin34 / (sin32 * self.PendulumLength)
        
        # 雅可比矩阵转置的逆 (J^T)^-1
        epsilon = 1e-6
        if abs(sin12) < epsilon:
            sin12 = epsilon if sin12 >= 0 else -epsilon
        if abs(sin34) < epsilon:
            sin34 = epsilon if sin34 >= 0 else -epsilon
        self.JT_inv_mat[0, 0] = -cos02 / (sin12 * self.VMC_L1)
        self.JT_inv_mat[0, 1] = cos03 / (sin34 * self.VMC_L1)
        self.JT_inv_mat[1, 0] = sin02 * self.PendulumLength / (sin12 * self.VMC_L1)
        self.JT_inv_mat[1, 1] = -sin03 * self.PendulumLength / (sin34 * self.VMC_L1)
    
    def VMCCal(self, F: np.ndarray) -> np.ndarray:
        """
        VMC正向计算：从虚拟力计算关节力矩
        
        Args:
            F: 虚拟力向量 [F_r, F_theta] (2,)
                F_r: 径向力
                F_theta: 切向力
        
        Returns:
            T: 关节力矩向量 [T1, T4] (2,)
        """
        T = np.zeros(2, dtype=np.float32)
        T[0] = self.JT_mat[0, 0] * F[0] + self.JT_mat[0, 1] * F[1]
        T[1] = self.JT_mat[1, 0] * F[0] + self.JT_mat[1, 1] * F[1]
        return T
    
    def VMCRevCal(self, T: np.ndarray) -> np.ndarray:
        """
        VMC逆向计算：从关节力矩计算虚拟力
        
        Args:
            T: 关节力矩向量 [T1, T4] (2,)
        
        Returns:
            F: 虚拟力向量 [F_r, F_theta] (2,)
        """
        F = np.zeros(2, dtype=np.float32)
        F[0] = self.JT_inv_mat[0, 0] * T[0] + self.JT_inv_mat[0, 1] * T[1]
        F[1] = self.JT_inv_mat[1, 0] * T[0] + self.JT_inv_mat[1, 1] * T[1]
        return F
    
    def VMCVelCal(self, phi_dot: np.ndarray) -> np.ndarray:
        """
        VMC速度计算：从关节速度计算虚拟速度
        
        Args:
            phi_dot: 关节速度向量 [phi1_dot, phi4_dot] (2,)
        
        Returns:
            v_dot: 虚拟速度向量 [v_r, v_theta] (2,)
        """
        v_dot = np.zeros(2, dtype=np.float32)
        v_dot[0] = self.J_mat[0, 0] * phi_dot[0] + self.J_mat[0, 1] * phi_dot[1]
        v_dot[1] = self.J_mat[1, 0] * phi_dot[0] + self.J_mat[1, 1] * phi_dot[1]
        return v_dot
    
    # Getter方法
    def GetPendulumLen(self) -> float:
        """获取倒立摆长度"""
        return self.PendulumLength
    
    def GetPendulumRadian(self) -> float:
        """获取倒立摆角度（弧度）"""
        return self.PendulumRadian
    
    def GetPhi4(self) -> float:
        """获取关节4角度"""
        return self.phi4
    
    def GetPhi1(self) -> float:
        """获取关节1角度"""
        return self.phi1
    
    def GetPendulumCoor(self) -> np.ndarray:
        """获取倒立摆坐标（C点）"""
        return self.CoorC.copy()
    
    def GetCoorB(self) -> np.ndarray:
        """获取B点坐标"""
        return self.CoorB.copy()
    
    def GetCoorD(self) -> np.ndarray:
        """获取D点坐标"""
        return self.CoorD.copy()

