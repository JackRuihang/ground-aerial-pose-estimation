import torch
import torch.nn as nn

# 加入score weights
class LMModule(nn.Module):
    """
    可导 LM 求解器，求解 theta, tx, ty
    """
    def __init__(self, damping=1e-3, n_iter=10, dtype=torch.float32, device=None):
        super().__init__()
        self.n_iter = n_iter
        self.damping = nn.Parameter(torch.tensor(damping, dtype=dtype, device=device))
        self.J = torch.tensor([[0., -1.],[1., 0.]], device=device)


    def residual(self, params, dP, P, weights=None):
        # print(dP.shape, P.shape)
        B, N, _ = dP.shape
        theta = params[:, 0:1]   # [B,1]
        tx = params[:, 1:2]      # [B,1]
        ty = params[:, 2:3]      # [B,1]

        cos = torch.cos(theta)   # [B,1]
        sin = torch.sin(theta)   # [B,1]

        di_norm = dP.norm(dim=2, keepdim=True).clamp(min=1e-12)
        di_unit = dP / di_norm   # [B,N,2]

        # v = R @ di_unit
        # v_x = cos*dx - sin*dy, v_y = sin*dx + cos*dy
        dx = di_unit[..., 0:1]  # [B,N,1]
        dy = di_unit[..., 1:2]  # [B,N,1]
        v_x = cos.unsqueeze(1) * dx - sin.unsqueeze(1) * dy  # [B,N,1]
        v_y = sin.unsqueeze(1) * dx + cos.unsqueeze(1) * dy  # [B,N,1]

        # u = J @ v = (-v_y, v_x)
        u_x = -v_y  # [B,N,1]
        u_y = v_x   # [B,N,1]

        t = torch.cat([tx, ty], dim=1).unsqueeze(1)   # [B,1,2]
        diff = P - t  # [B,N,2]
        # r_i = u_x * diff_x + u_y * diff_y
        r = (u_x.squeeze(-1) * diff[..., 0]) + (u_y.squeeze(-1) * diff[..., 1])  # [B,N]
        # print("r shape: ", r.shape)
        if weights is not None:
            r = r * weights  # [B,N]
        return r  # [B,N]


    def compute_jacobian(self, params, dP, P, weights=None):
        """
        Analytical Jacobian, fully vectorized.
        returns J: [B, N, 3]  (每个点一个残差)
        """
        # print("Using analytic Jacobian")
        B, N, _ = dP.shape
        theta = params[:, 0:1]   # [B,1]
        tx = params[:, 1:2]
        ty = params[:, 2:3]

        cos = torch.cos(theta)   # [B,1]
        sin = torch.sin(theta)   # [B,1]

        di_norm = dP.norm(dim=2, keepdim=True).clamp(min=1e-12)
        di_unit = dP / di_norm   # [B,N,2]
        dx = di_unit[..., 0:1]   # [B,N,1]
        dy = di_unit[..., 1:2]   # [B,N,1]

        # v_x, v_y
        v_x = cos.unsqueeze(1) * dx - sin.unsqueeze(1) * dy  # [B,N,1]
        v_y = sin.unsqueeze(1) * dx + cos.unsqueeze(1) * dy  # [B,N,1]

        # u = (-v_y, v_x)
        u_x = -v_y  # [B,N,1]
        u_y = v_x   # [B,N,1]

        # Partial derivatives wrt tx, ty:
        # ∂r/∂tx = -u_x  ;  ∂r/∂ty = -u_y
        dr_dtx = -u_x.squeeze(-1)  # [B,N]
        dr_dty = -u_y.squeeze(-1)  # [B,N]

        # ∂v_x/∂theta = -sin*dx - cos*dy
        # ∂v_y/∂theta =  cos*dx - sin*dy
        dvx_dtheta = -sin.unsqueeze(1) * dx - cos.unsqueeze(1) * dy  # [B,N,1]
        dvy_dtheta =  cos.unsqueeze(1) * dx - sin.unsqueeze(1) * dy  # [B,N,1]

        # u_x = -v_y -> du_x/dtheta = -dv_y/dtheta
        # u_y = v_x  -> du_y/dtheta = dv_x/dtheta
        dux_dtheta = -dvy_dtheta  # [B,N,1]
        duy_dtheta = dvx_dtheta   # [B,N,1]

        diff = P - torch.cat([tx, ty], dim=1).unsqueeze(1)  # [B,N,2]
        diff_x = diff[..., 0:1]  # [B,N,1]
        diff_y = diff[..., 1:2]  # [B,N,1]

        # dr/dtheta = du_x*(P_x - tx) + du_y*(P_y - ty)
        dr_dtheta = (dux_dtheta * diff_x + duy_dtheta * diff_y).squeeze(-1)  # [B,N]

        # Stack to J: [B, N, 3]
        J = torch.stack([dr_dtheta, dr_dtx, dr_dty], dim=2)  # [B,N,3]
        if weights is not None:
            J = J * weights.unsqueeze(2)  # [B,N,3]
        return J  # [B,N,3]


    def LM_update(self, params, r, J):
        """
        params: [B,3]
        r: [B,N]
        J: [B,N,3]
        """
        B, N, n = J.shape
        # reshape to [B, M, n] where M = N residuals
        JT = J.transpose(1, 2)  # [B,3,N]
        # H = J^T J  -> [B,3,3]
        H = torch.bmm(JT, J)  # [B,3,3]
        # damping (ensure positive)
        damp = torch.abs(self.damping)
        eye = torch.eye(n, device=H.device, dtype=H.dtype).unsqueeze(0)  # [1,3,3]
        H_damped = H + eye * damp

        # rhs = J^T r -> [B,3,1]
        rhs = torch.bmm(JT, r.unsqueeze(2))  # [B,3,1]

        # solve H_damped * x = rhs  (batched)
        # use torch.linalg.solve
        delta = -torch.linalg.solve(H_damped, rhs)  # [B,3,1]
        delta = delta.squeeze(2)  # [B,3]

        # produce new params WITHOUT in-place modification
        new_params = params + delta  # [B,3]
        return new_params
    
    def wrap_to_pi(self, theta):
        return theta
    
    def check_and_flip_theta(self, params, dP, P):
        """
        params: [B,3] -> theta, tx, ty
        dP: [B,N,2]  -> panorama coordinates
        P: [B,N,2]   -> satellite coordinates
        """
        theta = params[:, 0:1]  # [B,1]
        tx = params[:, 1:2]     # [B,1]
        ty = params[:, 2:3]     # [B,1]

        # 旋转矩阵 R 对每个 batch 计算
        cos = torch.cos(theta).unsqueeze(1)  # [B,1,1]
        sin = torch.sin(theta).unsqueeze(1)  # [B,1,1]
        R = torch.zeros(dP.shape[0], 2, 2, device=dP.device, dtype=dP.dtype)  # [B,2,2]
        R[:,0,0] = torch.cos(theta.squeeze(1))
        R[:,0,1] = -torch.sin(theta.squeeze(1))
        R[:,1,0] = torch.sin(theta.squeeze(1))
        R[:,1,1] = torch.cos(theta.squeeze(1))

        # 计算 r_i: ri = (R @ dP) · (P - t) / ||dP||^2
        # 批量向量化
        t = torch.cat([tx, ty], dim=1).unsqueeze(1)  # [B,1,2]
        Pd = P - t                                   # [B,N,2]
        # R @ dP
        R_dP = torch.einsum('bij,bnj->bni', R, dP)  # [B,N,2]
        # 点乘
        numerator = (R_dP * Pd).sum(dim=2)          # [B,N]
        denominator = (dP**2).sum(dim=2)            # [B,N]
        ri = numerator / (denominator + 1e-12)      # [B,N]

        # 平均 r_i
        ri_mean = ri.mean(dim=1)  # [B]

        mask = torch.sigmoid(-ri_mean*10).unsqueeze(1)  # 0~1连续
        theta_new = theta + mask * torch.pi

        # 更新 params
        params_new = torch.cat([theta_new, tx, ty], dim=1)
        return params_new, ri

    def forward(self, dP, P, weights=None, init_params=None):
        """
        dP: [B,N,2]  -> delta points
        P:  [B,N,2]  -> target points
        init_params: [B,3] initial theta, tx, ty
        """
        B = dP.shape[0]
        device = dP.device
        dtype = dP.dtype

        if weights is not None:
            weights = weights.reshape(B, -1)  # [B,N]

        if init_params is None:
            params = torch.zeros(B, 3, device=device, dtype=dtype)
        else:
            params = init_params.to(device=device, dtype=dtype)
        # params = params.requires_grad_(True)
        # Iterative LM
        for _ in range(self.n_iter):
            r = self.residual(params, dP, P, weights)        # [B,N]
            J = self.compute_jacobian(params, dP, P, weights)  # [B,N,3]
            params = self.LM_update(params, r, J)   # [B,3]
            # avoid in-place; create new tensor with theta wrapped
            theta_wrapped = self.wrap_to_pi(params[:, 0:1])
            params = torch.cat([theta_wrapped, params[:, 1:2], params[:, 2:3]], dim=1)
            # make sure params requires grad for next iter (it will be non-leaf)
            # params = params.requires_grad_(True)

        # params, ri = self.check_and_flip_theta(params, dP, P)

        return params
