# -*- coding: GBK -*-
import numpy as np
import cv2
from scipy import ndimage

WINDOW_SIZE = 35
MAX_DISPARITY = 60
PLANE_PENALTY = 120

# 平面参数存储类（向量化友好）
class PlaneParams:
    """存储平面参数 (a, b, c)，使得视差 d = a*x + b*y + c"""
    def __init__(self, shape):
        self.shape = shape  # (rows, cols)
        # 存储平面参数: a, b, c
        self.a = np.zeros(shape, dtype=np.float32)
        self.b = np.zeros(shape, dtype=np.float32)
        self.c = np.zeros(shape, dtype=np.float32)
    
    def get_at(self, y, x):
        """获取单个位置的平面参数"""
        return np.array([self.a[y, x], self.b[y, x], self.c[y, x]])
    
    def set_at(self, y, x, params):
        """设置单个位置的平面参数"""
        self.a[y, x], self.b[y, x], self.c[y, x] = params
    
def params_to_normal(a, b, c):
    """将平面参数转换为法向量表示"""       
    # 平面方程: a*x + b*y - d + c = 0
    # 所以法向量为 (a, b, -1)
    nz = np.ones(a.shape)
        
    # 归一化
    norm = np.sqrt(a**2+b**2+nz**2)
    nx=a/norm
    ny=b/norm
    nz=c/norm
        
    return nx,ny,nz

def normal_to_params(nx,ny,nz, d,x, y):
    """将法向量转换回平面参数"""
    # 避免除零
    nz[nz==0]=1e-8
    
    a = -nx / nz
    b = -ny / nz
    c = d - a * x - b * y
        
    return a, b, c

def compute_greyscale_gradient(img, grads):
    """计算灰度图像的梯度"""
    scale = 1.0
    delta = 0.0
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grads[:,:,0]= ndimage.sobel(gray,axis=0)
    grads[:,:,1]= ndimage.sobel(gray,axis=1)

    '''
    # 计算x方向梯度（Sobel）
    cv2.Sobel(gray, grads[0], cv2.CV_32F, 1, 0, ksize=3)
    # 计算y方向梯度（Sobel）
    cv2.Sobel(gray, grads[1], cv2.CV_32F, 0, 1, ksize=3)
    '''
    # 如果需要缩放，可以这样做
    if scale != 1.0 or delta != 0.0:
        grads[0] = cv2.convertScaleAbs(grads[0], alpha=scale, beta=delta)
        grads[1] = cv2.convertScaleAbs(grads[1], alpha=scale, beta=delta)

def inside(x, y, lbx, lby, ubx, uby):
    """判断点是否在范围内"""
    return lbx <= x < ubx and lby <= y < uby

# PatchMatch类（优化版）
class PatchMatch:
    def __init__(self, alpha, gamma, tau_c, tau_g):
        self.alpha = alpha
        self.gamma = gamma
        self.tau_c = tau_c
        self.tau_g = tau_g
        
        # 图像数据
        self.img_left = None
        self.img_right = None
        self.grad_left = None
        self.grad_right = None
        
        # 平面参数（左右视图）
        self.planes_left = None
        self.planes_right = None
        
        # 代价和权重
        self.cost_left = None
        self.cost_right = None
        self.weight_left = None
        self.weight_right = None
        
        # 视差图
        self.disp_left = None
        self.disp_right = None
        
        # 图像尺寸
        self.rows = 0
        self.cols = 0
    
    def plane_match_cost(self, plane_params, cx, cy, cpv):
        """计算聚合匹配代价"""
        sign = -1 if cpv == 0 else 1  # 0:左视图, 1:右视图
        half = WINDOW_SIZE // 2
        
        # 选择当前视图和参考视图
        if cpv == 0:  # 左视图
            f1, f2 = self.img_left, self.img_right
            g1, g2 = self.grad_left, self.grad_right
            weights = self.weight_left[cy, cx]
        else:  # 右视图
            f1, f2 = self.img_right, self.img_left
            g1, g2 = self.grad_right, self.grad_left
            weights = self.weight_right[cy, cx]
        
        # 生成窗口坐标
        y_range = np.arange(cy - half, cy + half + 1)
        x_range = np.arange(cx - half, cx + half + 1)
        y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')
        y_flat = y_grid.flatten()
        x_flat = x_grid.flatten()
        
        # 筛选有效坐标
        valid = (x_flat >= 0) & (x_flat < self.cols) & (y_flat >= 0) & (y_flat < self.rows)
        if not np.any(valid):
            return 0.0

        x_valid = x_flat[valid]
        y_valid = y_flat[valid]
        
        # 计算视差
        d = plane_params[0] * x_valid + plane_params[1] * y_valid + plane_params[2]

        # 惩罚无效视差
        valid_d = (d >= 0) & (d <= MAX_DISPARITY)
        cost = np.sum(~valid_d) * PLANE_PENALTY
        
        # 处理有效视差
        if not np.any(valid_d):
            return cost
        
        x_eff = x_valid[valid_d]
        y_eff = y_valid[valid_d]
        d_eff = d[valid_d]
        
        # 计算匹配点（双线性插值）
        match_x = x_eff + sign * d_eff
        x0 = np.floor(match_x).astype(int)
        x1 = x0 + 1
        wx = x1 - match_x  # 插值权重
        
        # 边界处理
        x0 = np.clip(x0, 0, self.cols - 2)
        x1 = x0+1
        
        # 双线性插值
        y_idx = y_eff.astype(int)
        color_interp = wx[:, None] * f2[y_idx, x0] + (1 - wx[:, None]) * f2[y_idx, x1]
        grad_interp_x = wx * g2[y_idx, x0, 0] + (1 - wx) * g2[y_idx, x1, 0]
        grad_interp_y = wx * g2[y_idx, x0, 1] + (1 - wx) * g2[y_idx, x1, 1]
        grad_interp = np.stack([grad_interp_x, grad_interp_y], axis=-1)
        
        # 获取权重
        wy = (y_eff - cy + half).astype(int)
        wx_idx = (x_eff - cx + half).astype(int)
        w = weights[wy, wx_idx]
        
        # 计算代价
        color_cost = np.sum(np.abs(f1[y_eff.astype(int), x_eff.astype(int)] - color_interp), axis=1)
        grad_cost = np.sum(np.abs(g1[y_eff.astype(int), x_eff.astype(int)] - grad_interp), axis=1)
        
        color_cost = np.minimum(color_cost, self.tau_c)
        grad_cost = np.minimum(grad_cost, self.tau_g)
        
        weighted_cost = w * ((1 - self.alpha) * color_cost + self.alpha * grad_cost)
        cost += np.sum(weighted_cost)
        
        return cost
    
    def plane_match_cost_vectorized(self, a_list, b_list, c_list,  x_list, y_list, cpv):
        """
        向量化版本：一次性计算所有像素点的平面匹配代价
        
        参数:
            plane_params_all: (rows, cols, 3) 每个像素点的平面参数
            cpv: 0或1，表示当前视图是左还是右
            
        返回:
            cost_map: (rows, cols) 每个像素点的匹配代价
        """
        sign = -1 if cpv == 0 else 1  # 0:左视图, 1:右视图
        half = WINDOW_SIZE // 2
        rows, cols = x_list.shape

        # 选择当前视图和参考视图
        if cpv == 0:  # 左视图
            f1, f2 = self.img_left, self.img_right
            g1, g2 = self.grad_left, self.grad_right
            weights = self.weight_left
        else:  # 右视图
            f1, f2 = self.img_right, self.img_left
            g1, g2 = self.grad_right, self.grad_left
            weights = self.weight_right
        
        # 创建所有可能的偏移坐标
        y_offsets = np.arange(-half, half + 1)
        x_offsets = np.arange(-half, half + 1)
        y_offsets_grid, x_offsets_grid = np.meshgrid(y_offsets, x_offsets, indexing='ij')
        
        # 扩展偏移以匹配所有像素
        y_offsets_exp = y_offsets_grid[None, None, :, :]  # (1, 1, W, W)
        x_offsets_exp = x_offsets_grid[None, None, :, :]  # (1, 1, W, W)
        
        # 创建所有像素的坐标网格
        y_coords = y_list[:, :, None, None]  # (rows, 1, 1, 1)
        x_coords = x_list[:, :, None, None]  # (1, cols, 1, 1)
            
        # 计算窗口内所有像素的坐标
        y_all = y_coords + y_offsets_exp  # (rows, cols, W, W)
        x_all = x_coords + x_offsets_exp  # (rows, cols, W, W)
        
        
        # 创建有效掩码（边界内的像素）
        valid_mask = (x_all >= 0) & (x_all < cols) & (y_all >= 0) & (y_all < rows)

        # 初始化代价图
        cost_map = np.zeros((rows, cols))
        
        # 为每个像素计算窗口内所有点的视差
        # 扩展平面参数以匹配窗口形状
        a = a_list[:, :, None, None]  # (rows, cols, 1, 1)
        b = b_list[:, :, None, None]  # (rows, cols, 1, 1)
        c = c_list[:, :, None, None]  # (rows, cols, 1, 1)

        # 计算视差 d = a*x + b*y + c
        d = a * x_all + b * y_all + c
        
        # 标记无效视差
        valid_d_mask = (d >= 0) & (d <= MAX_DISPARITY) & valid_mask

        valid_d = (d >= 0) & (d <= MAX_DISPARITY)
        
        # 我们只关心坐标有效且视差无效的点
        invalid_in_valid_coords = valid_mask & (~valid_d)
        
        # 惩罚无效视差
        invalid_d_count = np.sum(invalid_in_valid_coords, axis=(2, 3))
        cost_map += invalid_d_count * PLANE_PENALTY

        # 只处理有效视差的像素
        if not np.any(valid_d_mask):
            return cost_map
        
        # 获取有效视差的坐标
        valid_indices = np.where(valid_d_mask)
        
        # 准备批量处理的数据
        y_eff = y_all[valid_indices].astype(int)
        x_eff = x_all[valid_indices].astype(int)
        d_eff = d[valid_indices]
        cy = valid_indices[0] # 中心点y坐标
        cx = valid_indices[1]  # 中心点x坐标
        wy = valid_indices[2]  # 窗口内y偏移索引
        wx_idx = valid_indices[3]  # 窗口内x偏移索引

        # 计算匹配点（双线性插值）
        match_x = x_eff + sign * d_eff
        x0 = np.floor(match_x).astype(int)
        x1 = x0 + 1
        wx = x1 - match_x  # 插值权重
        
        # 边界处理
        x0 = np.clip(x0, 0, self.cols - 2)
        x1 = x0+1
        

        # 批量双线性插值
        color_interp = (wx[:, None] * f2[y_eff, x0] + 
                    (1 - wx[:, None]) * f2[y_eff, x1])
        
        grad_interp_x = (wx * g2[y_eff, x0, 0] + 
                        (1 - wx) * g2[y_eff, x1, 0])
        grad_interp_y = (wx * g2[y_eff, x0, 1] + 
                        (1 - wx) * g2[y_eff, x1, 1])
        grad_interp = np.stack([grad_interp_x, grad_interp_y], axis=-1)
        
        # 获取对应像素的颜色和梯度
        color_orig = f1[y_eff, x_eff]
        grad_orig = g1[y_eff, x_eff]
        
        # 计算代价
        color_cost = np.sum(np.abs(color_orig - color_interp), axis=1)
        grad_cost = np.sum(np.abs(grad_orig - grad_interp), axis=1)
        
        # 应用截断
        color_cost = np.minimum(color_cost, self.tau_c)
        grad_cost = np.minimum(grad_cost, self.tau_g)
        
        # 计算加权代价
        weighted_cost = ((1 - self.alpha) * color_cost + self.alpha * grad_cost)
        
        # 获取权重
        w = weights[cy, cx, wy, wx_idx]
        weighted_cost *= w
        
        # 累加到代价图
        # 使用numpy的add.at进行安全的累加
        np.add.at(cost_map, (cy, cx), weighted_cost)
        
        return cost_map
        
    def precompute_pixels_weights(self, frame, weights, ws):
        """预处理计算权重"""
        half = ws // 2
        rows, cols = frame.shape[:2]

        if frame.dtype != np.float64:
            frame = frame.astype(np.float64)

        # 生成窗口内相对坐标
        dy, dx = np.meshgrid(np.arange(-half, half+1), np.arange(-half, half+1), indexing='ij')
        dy = dy.flatten()  # (ws*ws,)
        dx = dx.flatten()  # (ws*ws,)
        n_offsets = len(dy)
        
        # 生成所有中心坐标
        cy, cx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        cy = cy.reshape(-1, 1)  # (N, 1)
        cx = cx.reshape(-1, 1)  # (N, 1)
        n_centers = cy.shape[0]
        
        # 计算窗口内所有像素坐标
        y = cy + dy.reshape(1, -1)  # (N, ws*ws)
        x = cx + dx.reshape(1, -1)  # (N, ws*ws)
        
        # 边界处理
        y = np.clip(y, 0, rows - 1)
        x = np.clip(x, 0, cols - 1)
        
        # 计算窗口坐标权重数组索引
        wy = dy + half  # (ws*ws,)
        wx = dx + half  # (ws*ws,)
        

        # 计算权重
        frame_vals = frame[cy[:, 0], cx[:, 0]]  # (N, 3)
        window_vals = frame[y, x]  # (N, ws*ws, 3)
        
        # 扩展frame_vals以便广播
        frame_vals_expanded = frame_vals[:, np.newaxis, :]  # (N, 1, 3)
        dist = np.sum(np.abs(frame_vals_expanded - window_vals), axis=2)  # (N, ws*ws)
        #print(f"dist 统计: min={dist.min():.2f}, max={dist.max():.2f}, mean={dist.mean():.2f}")
        #print(f"gamma={self.gamma}")
        #print(f"-dist/gamma 范围: {(-dist/self.gamma).min():.2f} 到 {(-dist/self.gamma).max():.2f}")

        weights_vals = np.exp(-dist / self.gamma)  # (N, ws*ws)
        
        # 关键修正：使用循环逐个赋值，或者使用高级索引
        # 方法1：使用循环（简单但可能慢）
        # for i in range(n_offsets):
        #     weights[cy[:, 0], cx[:, 0], wy[i], wx[i]] = weights_vals[:, i]
        
        # 方法2：使用高级索引（推荐）
        # 创建索引数组
        cy_idx = np.repeat(cy[:, 0], n_offsets)  # (N*ws*ws,)
        cx_idx = np.repeat(cx[:, 0], n_offsets)  # (N*ws*ws,)
        wy_idx = np.tile(wy, n_centers)  # (N*ws*ws,)
        wx_idx = np.tile(wx, n_centers)  # (N*ws*ws,)
        
        # 展平权重值
        weights_vals_flat = weights_vals.flatten()  # (N*ws*ws,)
        
        # 一次性赋值
        weights[cy_idx, cx_idx, wy_idx, wx_idx] = weights_vals_flat
        
        return weights
    
    def initialize_random_planes(self, planes, max_d):
        """初始化随机平面"""
        rows, cols = planes.shape
        RAND_HALF = 0x7FFFFFFF  # 模拟RAND_MAX/2
        # 随机法向量（归一化）
        nx = (np.random.uniform(-RAND_HALF, RAND_HALF,(rows,cols))) / RAND_HALF
        ny = (np.random.uniform(-RAND_HALF, RAND_HALF,(rows,cols))) / RAND_HALF
        nz = (np.random.uniform(-RAND_HALF, RAND_HALF,(rows,cols))) / RAND_HALF
        
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        norm[norm == 0] = 1.0  # 避免除以零
        
        nx = nx / norm
        ny = ny / norm
        nz = nz / norm

        nz_safe=nz.copy()
        nz_safe[nz==0]=1e-8
        
        # 从法向量计算平面参数
        # 对于平面方程 n·(X - P) = 0，其中X=(x,y,d)
        # 展开得: nx*x + ny*y + nz*d - n·P = 0
        # 所以: d = (-nx/nz)*x + (-ny/nz)*y + (n·P/nz)
        x_grid, y_grid = np.meshgrid(np.arange(cols), np.arange(rows),indexing='xy')
        z_grid = np.random.uniform(0,max_d,(rows,cols))
        planes.a = -nx / nz_safe
        planes.b = -ny / nz_safe
        planes.c = (nx * x_grid + ny * y_grid + nz * z_grid) / nz_safe
    
    def evaluate_planes_cost(self):
        """评估所有平面的代价""" 
        rows, cols = self.planes_left.shape
        x_list, y_list = np.meshgrid(np.arange(cols), np.arange(rows),indexing='xy')  
        self.cost_left = self.plane_match_cost_vectorized(self.planes_left.a, self.planes_left.b, self.planes_left.c, x_list, y_list, 0)
        self.cost_right = self.plane_match_cost_vectorized(self.planes_right.a, self.planes_right.b, self.planes_right.c, x_list, y_list, 1)

    def spatial_propagation(self, cpv, iter_even):
        """空间传播"""
        if cpv == 0:
            planes = self.planes_left
            costs = self.cost_left
        else:
            planes = self.planes_right
            costs = self.cost_right
        
        rows, cols = self.rows, self.cols
        
        if iter_even:
            # 从上到下，从左到右
            for y in range(rows):
                for x in range(cols):
                    # 检查上方邻居
                    if y > 0:
                        neighbor_params = planes.get_at(y-1, x)
                        new_cost = self.plane_match_cost(neighbor_params, x, y, cpv)
                        if new_cost < costs[y, x]:
                            planes.set_at(y, x, neighbor_params)
                            costs[y, x] = new_cost
                    
                    # 检查下方邻居
                    if x > 0:
                        neighbor_params = planes.get_at(y, x-1)
                        new_cost = self.plane_match_cost(neighbor_params, x, y, cpv)
                        if new_cost < costs[y, x]:
                            planes.set_at(y, x, neighbor_params)
                            costs[y, x] = new_cost
        else:
            # 从下到上，从右到左
            for y in range(rows-1, -1, -1):
                for x in range(cols-1, -1, -1):
                    # 检查下方邻居
                    if y < rows-1:
                        neighbor_params = planes.get_at(y+1, x)
                        new_cost = self.plane_match_cost(neighbor_params, x, y, cpv)
                        if new_cost < costs[y, x]:
                            planes.set_at(y, x, neighbor_params)
                            costs[y, x] = new_cost
                    
                    # 检查右方邻居
                    if x < cols-1:
                        neighbor_params = planes.get_at(y, x+1)
                        new_cost = self.plane_match_cost(neighbor_params, x, y, cpv)
                        if new_cost < costs[y, x]:
                            planes.set_at(y, x, neighbor_params)
                            costs[y, x] = new_cost
    
    def plane_refinement_vectorized(self, cpv, max_iterations=8):
        """向量化的平面细化"""
        if cpv == 0:
            planes = self.planes_left
            costs = self.cost_left
        else:
            planes = self.planes_right
            costs = self.cost_right
        
        rows, cols = self.rows, self.cols
        
        # 初始化平面参数
        a_curr = planes.a.copy()  # (rows, cols)
        b_curr = planes.b.copy()
        c_curr = planes.c.copy()
        cost_curr = costs.copy()
        
        # 创建坐标网格
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        
        # 多尺度细化：从大扰动开始，逐步减小
        scale_factors = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
        
        for scale_idx, scale in enumerate(scale_factors):
            if scale_idx >= max_iterations:
                break
                
            # 当前尺度的扰动范围
            max_dz = (MAX_DISPARITY / 2) * scale
            max_dn=1.0

            # 生成随机扰动
            # 这里我们可以生成多组随机扰动，选择最好的一个
            num_trials = 3  # 每个尺度尝试3次扰动
            
            for trial in range(num_trials):
                # 1. 生成视差扰动
                delta_z = np.random.uniform(-max_dz, max_dz, size=(rows, cols))
                z = a_curr * x_grid + b_curr * y_grid + c_curr+delta_z
                
                nx,ny,nz = params_to_normal(a_curr, b_curr, c_curr)
                delta_nx = np.random.uniform(-max_dn, max_dn,(rows,cols))
                delta_ny = np.random.uniform(-max_dn, max_dn,(rows,cols))
                delta_nz = np.random.uniform(-max_dn, max_dn,(rows,cols))
                new_nx = nx + delta_nx
                new_ny = ny + delta_ny
                new_nz = nz + delta_nz
                norm = np.sqrt(new_nx**2+new_ny**2+new_nz**2)
                norm[norm==0]=1.0                    
                new_nx=new_nx/norm
                new_ny=new_ny/norm
                new_nz=new_nz/norm
                
                # 计算新参数
                a_new,b_new,c_new=normal_to_params(nx,ny,nz, z, x_grid, y_grid)
                
                # 批量计算新代价
                new_costs = self.plane_match_cost_vectorized(
                    a_new, b_new, c_new,
                    x_grid, y_grid, cpv
                )
                
                # 找到代价降低的像素
                improved_mask = new_costs < cost_curr
                
                if np.any(improved_mask):
                    # 更新改进的像素
                    a_curr[improved_mask] = a_new[improved_mask]
                    b_curr[improved_mask] = b_new[improved_mask]
                    c_curr[improved_mask] = c_new[improved_mask]
                    cost_curr[improved_mask] = new_costs[improved_mask]
        
        # 更新平面和代价
        planes.a[:, :] = a_curr
        planes.b[:, :] = b_curr
        planes.c[:, :] = c_curr
        costs[:, :] = cost_curr
    
    def view_propagation_vectorized(self, cpv):
        """向量化的视图间传播"""
        rows, cols = self.rows, self.cols
        
        if cpv == 0:  # 从左到右传播
            src_planes = self.planes_left
            src_costs = self.cost_left
            dst_planes = self.planes_right
            dst_costs = self.cost_right
            sign = -1
        else:  # 从右到左传播
            src_planes = self.planes_right
            src_costs = self.cost_right
            dst_planes = self.planes_left
            dst_costs = self.cost_left
            sign = 1
        
        # 获取所有平面参数（向量化）
        a_src = src_planes.a  # 形状: (rows, cols)
        b_src = src_planes.b
        c_src = src_planes.c
        
        # 创建坐标网格
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        
        # 计算所有像素的视差（向量化）
        d = a_src * x_grid + b_src * y_grid + c_src  # 形状: (rows, cols)
        
        # 计算目标位置
        target_x = x_grid + sign * d  # 形状: (rows, cols)

        target_x_int = np.round(target_x).astype(int)

        # 创建有效掩码
        valid_mask = (target_x_int >= 0) & (target_x_int < cols)
        
        # 如果没有有效像素，直接返回
        if not np.any(valid_mask):
            return
        
        # 提取有效数据
        y_valid = y_grid[valid_mask]  # 形状: (n_valid,)
        x_valid = x_grid[valid_mask]
        target_x_valid = target_x_int[valid_mask]
        
        a_dst = a_src[valid_mask]
        b_dst = b_src[valid_mask]
        c_dst = c_src[valid_mask]
        d_dst=a_dst*x_valid+b_dst*y_valid+c_dst

        nx,ny,nz=params_to_normal(a_dst,b_dst,c_dst)

        a_dst,b_dst,c_dst=normal_to_params(nx,ny,nz,d_dst,target_x_valid,y_valid)
        
        # 计算新代价（需要实现向量化的代价计算）
        # 创建临时二维数组，只在有效位置填充数据
        # 创建零数组
        a_temp = np.zeros((rows, cols))
        b_temp = np.zeros((rows, cols))
        c_temp = np.zeros((rows, cols))
        cx_temp = np.zeros((rows, cols), dtype=int)
        cy_temp = np.zeros((rows, cols), dtype=int)

        # 只在有效位置填充数据
        a_temp[y_valid, x_valid] = a_dst
        b_temp[y_valid, x_valid] = b_dst
        c_temp[y_valid, x_valid] = c_dst
        cx_temp[y_valid, x_valid] = target_x_valid
        cy_temp[y_valid, x_valid] = y_valid

        # 调用向量化函数
        cost_map = self.plane_match_cost_vectorized(
            a_temp, b_temp, c_temp, cx_temp, cy_temp, 1-cpv
        )

        # 提取有效位置的代价
        new_costs = cost_map[y_valid, x_valid]
        dst_costs_flat =dst_costs[y_valid, target_x_valid]
        
        # 找出需要更新的像素
        update_mask = new_costs < dst_costs_flat
        
        if not np.any(update_mask):
            return
        
        # 提取需要更新的数据
        update_y = y_valid[update_mask]
        update_x = target_x_valid[update_mask]
        update_a = a_dst[update_mask]
        update_b = b_dst[update_mask]
        update_c = c_dst[update_mask]
        update_costs = new_costs[update_mask]
        
        # 批量更新目标视图
        dst_planes.a[update_y, update_x] = update_a
        dst_planes.b[update_y, update_x] = update_b
        dst_planes.c[update_y, update_x] = update_c
        dst_costs[update_y, update_x] = update_costs
    
    def compute_disparity_maps(self):
        """计算视差图"""
        self.disp_left = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.disp_right = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # 为每个像素计算视差
        y_coords, x_coords = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing='ij')
        
        self.disp_left = self.planes_left.a * x_coords + self.planes_left.b * y_coords + self.planes_left.c
        self.disp_right = self.planes_right.a * x_coords + self.planes_right.b * y_coords + self.planes_right.c
        

    def fill_invalid_pixels(self, y, x, planes, validity):
        """通过反推左右像素视差的方法填充遮挡像素"""
        x_lft = x - 1
        x_rgt = x + 1
        
        while x_lft >= 0 and not validity[y, x_lft]:
            x_lft -= 1
        
        while x_rgt < self.cols and not validity[y, x_rgt]:
            x_rgt += 1
        
        best_plane_x = x
        if x_lft >= 0 and x_rgt < self.cols:
            disp_l = x*planes.a[y][x_lft]+y*planes.b[y][x_lft]+planes.c[y][x_lft]
            disp_r = x*planes.a[y][x_rgt]+y*planes.b[y][x_rgt]+planes.c[y][x_rgt]
            best_plane_x = x_lft if disp_l < disp_r else x_rgt
        elif x_lft >= 0:
            best_plane_x = x_lft
        elif x_rgt < self.cols:
            best_plane_x = x_rgt
        
        params=planes.get_at(y,best_plane_x)
        planes.set_at(y, x, params)
    
    def weighted_median_filter(self, cx, cy, disparity, weights, valid, ws, use_invalid):
        """加权中值滤波"""
        half = ws // 2
        w_tot = 0.0
        disps_w = []
        
        for x in range(cx - half, cx + half + 1):
            for y in range(cy - half, cy + half + 1):
                if inside(x, y, 0, 0, self.cols, self.rows) and (use_invalid or valid[y, x]):
                    wy = y - cy + half
                    wx = x - cx + half
                    w = weights[cy, cx, wy, wx]
                    w_tot += w
                    disps_w.append((w, disparity[y, x]))
        
        disps_w.sort()
        med_w = w_tot / 2.0
        current_w = 0.0
        
        for i, (w, d) in enumerate(disps_w):
            current_w += w
            if current_w >= med_w:
                if i == 0:
                    disparity[cy, cx] = d
                else:
                    disparity[cy, cx] = (disps_w[i-1][1] + d) / 2.0
                break

    def set_images(self, img_left, img_right):
        """设置输入图像"""
        self.img_left = img_left.astype(np.float32) / 255.0
        self.img_right = img_right.astype(np.float32) / 255.0
        self.rows, self.cols = img_left.shape[:2]
        
        print("Precomputing pixels weight...")
        wmat_sizes = (self.rows, self.cols, WINDOW_SIZE, WINDOW_SIZE)
        self.weight_left = np.zeros(wmat_sizes, dtype=np.float32)
        self.weight_right = np.zeros(wmat_sizes, dtype=np.float32)
        self.precompute_pixels_weights(img_left, self.weight_left, WINDOW_SIZE)
        self.precompute_pixels_weights(img_right, self.weight_right, WINDOW_SIZE)

        print("Evaluating images gradient...")
        self.grad_left = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        self.grad_right = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        compute_greyscale_gradient(img_left, self.grad_left)
        compute_greyscale_gradient(img_right, self.grad_right)
    
        print("Initializing random planes...")
        self.planes_left = PlaneParams((self.rows, self.cols))
        self.planes_right = PlaneParams((self.rows, self.cols))
        self.initialize_random_planes(self.planes_left, MAX_DISPARITY)
        self.initialize_random_planes(self.planes_right, MAX_DISPARITY)

        print("Evaluating initial costs...")
        self.evaluate_planes_cost()
        
        self.disp_left = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.disp_right = np.zeros((self.rows, self.cols), dtype=np.float32)
    
    def process(self, iterations=3):
        """主处理流程"""
        print("Processing...")
        
        for iter in range(iterations):
            print(f"Iteration {iter+1}/{iterations}")
            iter_even = (iter % 2 == 0)           
            for view in range(2):  # 处理左右视图
                self.spatial_propagation(view, iter_even)
                self.plane_refinement_vectorized(view)
                self.view_propagation_vectorized(view)
        
        print("Computing disparity maps...")
        self.compute_disparity_maps()
    
    def post_process(self):
        """后处理，左右一致化检查，填充遮挡像素，加权中值滤波"""
        print("Executing post-processing...")
        lft_validity = np.zeros((self.rows, self.cols), dtype=np.bool_)
        rgt_validity = np.zeros((self.rows, self.cols), dtype=np.bool_)
        
        for y in range(self.rows):
            for x in range(self.cols):
                x_rgt_match = max(0, min(self.cols - 1, int(x - self.disp_left[y, x])))
                lft_validity[y, x] = np.abs(self.disp_left[y, x] - self.disp_left[y, x_rgt_match]) <= 1
                
                x_lft_match = max(0, min(self.cols - 1, int(x + self.disp_right[y, x])))
                rgt_validity[y, x] = np.abs(self.disp_right[y, x] - self.disp_right[y, x_lft_match]) <= 1
        
        for y in range(self.rows):
            for x in range(self.cols):
                if not lft_validity[y, x]:
                    self.fill_invalid_pixels(y, x, self.planes_left, lft_validity)
                if not rgt_validity[y, x]:
                    self.fill_invalid_pixels(y, x, self.planes_right, rgt_validity)
        
        self.compute_disparity_maps()
        
        for x in range(self.cols):
            for y in range(self.rows):
                self.weighted_median_filter(x, y, self.disp_left, self.weight_left, lft_validity, WINDOW_SIZE, False)
                self.weighted_median_filter(x, y, self.disp_right, self.weight_right, rgt_validity, WINDOW_SIZE, False)
        
        return self.disp_left,self.disp_right
