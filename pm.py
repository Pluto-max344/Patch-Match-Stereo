import numpy as np
import cv2
import random
from scipy import ndimage
from plane import Plane

WINDOW_SIZE = 35
MAX_DISPARITY = 60
PLANE_PENALTY = 120

# 2维平面数据结构
class Matrix2D:
    def __init__(self, rows=0, cols=0, default=None):
        self.rows = rows
        self.cols = cols
        self.data = []
        if rows > 0 and cols > 0:
            if default is not None:
                self.data = [[default for _ in range(cols)] for _ in range(rows)]
            else:
                self.data = [[None for _ in range(cols)] for _ in range(rows)]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __call__(self, row, col):
        return self.data[row][col]
    
    def set(self, row, col, value):
        self.data[row][col] = value



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

def disparity(x, y, plane):
    """计算平面内的点的视差"""
    return plane[0] * x + plane[1] * y + plane[2]

def weight(p, q, gamma=10.0):
    """自适应权重函数"""
    return np.exp(-np.linalg.norm(p - q, ord=1) / gamma)

def vec_average(x, y, wx):
    """均值函数"""
    return wx * x + (1 - wx) * y

# 寻找倾斜平面类
class PatchMatch:
    def __init__(self, alpha, gamma, tau_c, tau_g):
        """初始化"""
        self.alpha = alpha
        self.gamma = gamma
        self.tau_c = tau_c
        self.tau_g = tau_g
        self.views = [None, None]
        self.grads = [None, None]
        self.disps = [None, None]
        self.planes = [Matrix2D(), Matrix2D()]
        self.costs = [None, None]
        self.weigs = [None, None]
        self.rows = 0
        self.cols = 0
    
    def plane_match_cost(self, p, cx, cy, ws, cpv):
        """计算某个平面下的匹配误差"""
        sign = -1 + 2 * cpv
        half = ws // 2
        cost = 0.0

        f1 = self.views[cpv]
        f2 = self.views[1 - cpv]
        g1 = self.grads[cpv]
        g2 = self.grads[1 - cpv]
        w1 = self.weigs[cpv]

        # 获取图像尺寸
        f1_rows, f1_cols = f1.shape[0], f1.shape[1]
        
        # 1. 生成窗口内所有坐标（x, y）的网格并展平
        x_range = np.arange(cx - half, cx + half + 1)
        y_range = np.arange(cy - half, cy + half + 1)
        x_grid, y_grid = np.meshgrid(x_range, y_range, indexing='xy')
        x_flat = x_grid.flatten()  # 窗口内所有x坐标（一维）
        y_flat = y_grid.flatten()  # 窗口内所有y坐标（一维）

        # 2. 筛选有效坐标（在图像范围内）
        valid_mask = (x_flat >= 0) & (x_flat < f1_cols) & (y_flat >= 0) & (y_flat < f1_rows)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        if len(x_valid) == 0:
            return cost  # 无有效坐标时直接返回0

        # 3. 计算所有有效坐标的视差dsp
        dsp = p[0] * x_valid + p[1] * y_valid + p[2]  # 向量化计算视差

        # 4. 区分有效视差（在[0, MAX_DISPARITY]范围内）和无效视差
        valid_dsp_mask = (dsp >= 0) & (dsp <= MAX_DISPARITY)
        invalid_dsp_count = np.sum(~valid_dsp_mask)  # 无效视差数量（用于惩罚项）
        cost += invalid_dsp_count * PLANE_PENALTY  # 累加惩罚项

        # 筛选出有效视差对应的坐标
        x_effective = x_valid[valid_dsp_mask]
        y_effective = y_valid[valid_dsp_mask]
        dsp_effective = dsp[valid_dsp_mask]
        if len(x_effective) == 0:
            return cost  # 无有效视差时返回当前cost

        # 5. 计算匹配点x坐标（双线性插值相关）
        match = x_effective + sign * dsp_effective
        x_match = match.astype(int)  # 整数部分
        wm = 1 - (match - x_match)   # 插值权重

        # 处理x_match边界（确保x_match和x_match+1在有效范围内）
        x_match = np.clip(x_match, 0, f1_cols - 2)  # 避免x_match+1越界

        # 6. 双线性插值计算mcolo和mgrad
        # 从f2中获取插值颜色
        f2_y = y_effective.astype(int)
        f2_x0 = x_match
        f2_x1 = x_match + 1
        mcolo = wm[:, np.newaxis] * f2[f2_y, f2_x0] + (1 - wm[:, np.newaxis]) * f2[f2_y, f2_x1]

        # 从g2中获取插值梯度（g2是双通道，需分别处理）
        mgrad = np.empty((len(x_effective), 2), dtype=np.float32)
        for i in range(2):  # 处理x和y方向梯度
            mgrad[:, i] = wm * g2[f2_y, f2_x0, i] + (1 - wm) * g2[f2_y, f2_x1, i]

        # 7. 获取对应的权重w
        wy = (y_effective - cy + half).astype(int)
        wx = (x_effective - cx + half).astype(int)
        
        # 确保权重索引在有效范围内
        wy = np.clip(wy, 0, ws - 1)
        wx = np.clip(wx, 0, ws - 1)
        
        # 修正权重索引方式
        w = w1[cy, cx, wy, wx]  # 从预计算的权重矩阵中索引

        # 8. 计算不相似度并累加
        f1_vals = f1[y_effective.astype(int), x_effective.astype(int)]  # f1中对应坐标的颜色
        g1_vals = g1[y_effective.astype(int), x_effective.astype(int)]  # f1中对应坐标的梯度

        # 修正：计算颜色和梯度的L1距离（绝对值之和）
        # 对于颜色，我们需要对三个通道求和
        cost_c = np.sum(np.abs(f1_vals - mcolo), axis=1)
        # 对于梯度，我们需要对两个通道求和
        cost_g = np.sum(np.abs(g1_vals - mgrad), axis=1)
        
        cost_c = np.minimum(cost_c, self.tau_c)
        cost_g = np.minimum(cost_g, self.tau_g)
        
        # 累加加权总代价 - 确保所有数组都是1D
        w = w.flatten()
        cost_c = cost_c.flatten()
        cost_g = cost_g.flatten()
        
        weighted_cost = w * ((1 - self.alpha) * cost_c + self.alpha * cost_g)
        cost += np.sum(weighted_cost)

        return cost

    def precompute_pixels_weights(self, frame, weights, ws):
        """赋予权重"""
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

    def planes_to_disparity(self, planes, disp):
        """计算平面内视差"""
            for x in range(self.cols):
                for y in range(self.rows):
                    disp[y, x] = disparity(x, y, planes(y, x))
        
    def initialize_random_planes(self, planes, max_d):
        """初始化平面"""
        RAND_HALF = 0x7FFFFFFF  # 模拟RAND_MAX/2
        for y in range(self.rows):
            for x in range(self.cols):
                z = random.uniform(0.0, max_d)
                point = np.array([x, y, z], dtype=np.float32)
                
                nx = (random.randint(-RAND_HALF, RAND_HALF)) / RAND_HALF
                ny = (random.randint(-RAND_HALF, RAND_HALF)) / RAND_HALF
                nz = (random.randint(-RAND_HALF, RAND_HALF)) / RAND_HALF
                normal = np.array([nx, ny, nz], dtype=np.float32)
                normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else normal
                
                planes.set(y, x, Plane(point, normal))
    
    def evaluate_planes_cost(self, cpv):
        """计算平面对应匹配代价"""
        for y in range(self.rows):
            for x in range(self.cols):
                self.costs[cpv][y, x] = self.plane_match_cost(self.planes[cpv](y, x), x, y, WINDOW_SIZE, cpv)
    
    def spatial_propagation(self, x, y, cpv, iter):
        """空间传播"""
        rows = self.views[cpv].shape[0]
        cols = self.views[cpv].shape[1]
        offsets = []
        
        if iter % 2 == 0:
            offsets = [(-1, 0), (0, -1)]
        else:
            offsets = [(1, 0), (0, 1)]
        
        old_plane = self.planes[cpv](y, x)
        old_cost = self.costs[cpv][y, x]
        
        for (dx, dy) in offsets:
            nx = x + dx
            ny = y + dy
            if inside(nx, ny, 0, 0, cols, rows):
                p_neigb = self.planes[cpv](ny, nx)
                new_cost = self.plane_match_cost(p_neigb, x, y, WINDOW_SIZE, cpv)
                if new_cost < old_cost:
                    self.planes[cpv].set(y, x, p_neigb)
                    self.costs[cpv][y, x] = new_cost
    
    def view_propagation(self, x, y, cpv):
        """左右视图传播"""
        sign = -1 if cpv == 0 else 1
        view_plane = self.planes[cpv](y, x)
        
        qx = [0]
        qy = [0]
        new_plane = view_plane.view_transform(x, y, sign, qx, qy)
        mx, my = qx[0], qy[0]
        
        if inside(int(mx), int(my), 0, 0, self.views[0].shape[1], self.views[0].shape[0]):
            mx_int = int(mx)
            my_int = int(my)
            old_cost = self.costs[1 - cpv][my_int, mx_int]
            new_cost = self.plane_match_cost(new_plane, mx_int, my_int, WINDOW_SIZE, 1 - cpv)
            
            if new_cost < old_cost:
                self.planes[1 - cpv].set(my_int, mx_int, new_plane)
                self.costs[1 - cpv][my_int, mx_int] = new_cost
    
    def plane_refinement(self, x, y, cpv, max_delta_z, max_delta_n, end_dz):
        """平面细化,调整平面参数"""
        max_dz = max_delta_z
        max_dn = max_delta_n
        
        old_plane = self.planes[cpv](y, x)
        old_cost = self.costs[cpv][y, x]
        
        while max_dz >= end_dz:
            z = old_plane[0] * x + old_plane[1] * y + old_plane[2]
            delta_z = random.uniform(-max_dz, max_dz)
            new_point = np.array([x, y, z + delta_z], dtype=np.float32)
            
            n = old_plane.get_normal()
            delta_n = np.array([
                random.uniform(-max_dn, max_dn),
                random.uniform(-max_dn, max_dn),
                random.uniform(-max_dn, max_dn)
            ], dtype=np.float32)
            new_normal = n + delta_n
            new_normal = new_normal / np.linalg.norm(new_normal) if np.linalg.norm(new_normal) != 0 else new_normal
            
            new_plane = Plane(new_point, new_normal)
            new_cost = self.plane_match_cost(new_plane, x, y, WINDOW_SIZE, cpv)
            
            if new_cost < old_cost:
                self.planes[cpv].set(y, x, new_plane)
                old_cost = new_cost
            
            max_dz /= 2.0
            max_dn /= 2.0
    
    def process_pixel(self, x, y, cpv, iter):
        """处理像素，通过传播建立平面"""
        self.spatial_propagation(x, y, cpv, iter)
        self.plane_refinement(x, y, cpv, MAX_DISPARITY / 2, 1.0, 0.1)
        self.view_propagation(x, y, cpv)
    
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
            disp_l = disparity(x, y, planes(y, x_lft))
            disp_r = disparity(x, y, planes(y, x_rgt))
            best_plane_x = x_lft if disp_l < disp_r else x_rgt
        elif x_lft >= 0:
            best_plane_x = x_lft
        elif x_rgt < self.cols:
            best_plane_x = x_rgt
        
        planes.set(y, x, planes(y, best_plane_x))
    
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
    
    def set(self, img1, img2):
        """初始化"""
        self.views[0] = img1
        self.views[1] = img2
        self.rows, self.cols = img1.shape[:2]
        
        print("Precomputing pixels weight...")
        wmat_sizes = (self.rows, self.cols, WINDOW_SIZE, WINDOW_SIZE)
        self.weigs[0] = np.zeros(wmat_sizes, dtype=np.float32)
        self.weigs[1] = np.zeros(wmat_sizes, dtype=np.float32)
        self.precompute_pixels_weights(img1, self.weigs[0], WINDOW_SIZE)
        self.precompute_pixels_weights(img2, self.weigs[1], WINDOW_SIZE)
        
        print("Evaluating images gradient...")
        self.grads[0] = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        self.grads[1] = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        compute_greyscale_gradient(img1, self.grads[0])
        compute_greyscale_gradient(img2, self.grads[1])
        
        print("Precomputing random planes...")
        self.planes[0] = Matrix2D(self.rows, self.cols)
        self.planes[1] = Matrix2D(self.rows, self.cols)
        self.initialize_random_planes(self.planes[0], MAX_DISPARITY)
        self.initialize_random_planes(self.planes[1], MAX_DISPARITY)
        
        print("Evaluating initial planes cost...")
        self.costs[0] = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.costs[1] = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.evaluate_planes_cost(0)
        self.evaluate_planes_cost(1)
        
        self.disps[0] = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.disps[1] = np.zeros((self.rows, self.cols), dtype=np.float32)
    
    def process(self, iterations, reverse=False):
        """处理像素，通过传播建立平面，计算平面视差"""
        print("Processing left and right views...")
        start = reverse
        end = iterations + reverse
        
        for iter in range(start, end):
            iter_type = (iter % 2 == 0)
            print(f"Iteration {iter - reverse + 1}/{iterations}", end="\r")
            
            for work_view in range(2):
                if iter_type:
                    for y in range(self.rows):
                        for x in range(self.cols):
                            self.process_pixel(x, y, work_view, iter)
                else:
                    for y in range(self.rows - 1, -1, -1):
                        for x in range(self.cols - 1, -1, -1):
                            self.process_pixel(x, y, work_view, iter)
        print()
        
        self.planes_to_disparity(self.planes[0], self.disps[0])
        self.planes_to_disparity(self.planes[1], self.disps[1])
    
    def postProcess(self):
        """后处理，左右一致化检查，，填充遮挡像素，加权中值滤波"""
        print("Executing post-processing...")
        lft_validity = np.zeros((self.rows, self.cols), dtype=np.bool_)
        rgt_validity = np.zeros((self.rows, self.cols), dtype=np.bool_)
        
        for y in range(self.rows):
            for x in range(self.cols):
                x_rgt_match = max(0, min(self.cols - 1, int(x - self.disps[0][y, x])))
                lft_validity[y, x] = np.abs(self.disps[0][y, x] - self.disps[1][y, x_rgt_match]) <= 1
                
                x_lft_match = max(0, min(self.cols - 1, int(x + self.disps[1][y, x])))
                rgt_validity[y, x] = np.abs(self.disps[1][y, x] - self.disps[0][y, x_lft_match]) <= 1
        
        for y in range(self.rows):
            for x in range(self.cols):
                if not lft_validity[y, x]:
                    self.fill_invalid_pixels(y, x, self.planes[0], lft_validity)
                if not rgt_validity[y, x]:
                    self.fill_invalid_pixels(y, x, self.planes[1], rgt_validity)
        
        self.planes_to_disparity(self.planes[0], self.disps[0])
        self.planes_to_disparity(self.planes[1], self.disps[1])
        
        for x in range(self.cols):
            for y in range(self.rows):
                self.weighted_median_filter(x, y, self.disps[0], self.weigs[0], lft_validity, WINDOW_SIZE, False)
                self.weighted_median_filter(x, y, self.disps[1], self.weigs[1], rgt_validity, WINDOW_SIZE, False)
    