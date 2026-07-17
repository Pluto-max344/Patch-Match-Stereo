import numpy as np
import cv2
from scipy import ndimage

# 默认参数移入类中或作为默认值，不再使用全局常量固定
MAX_DISPARITY = 60
PLANE_PENALTY = 120

class PlaneParams:
    """存储平面参数 (a, b, c)"""
    def __init__(self, shape):
        self.shape = shape
        self.a = np.zeros(shape, dtype=np.float32)
        self.b = np.zeros(shape, dtype=np.float32)
        self.c = np.zeros(shape, dtype=np.float32)
    
    def get_at(self, y, x):
        return np.array([self.a[y, x], self.b[y, x], self.c[y, x]])
    
    def set_at(self, y, x, params):
        self.a[y, x], self.b[y, x], self.c[y, x] = params

def params_to_normal(a, b, c):
    nz = np.ones(a.shape)
    norm = np.sqrt(a**2+b**2+nz**2)
    return a/norm, b/norm, c/norm

def normal_to_params(nx, ny, nz, d, x, y):
    nz[nz==0]=1e-8
    a = -nx / nz
    b = -ny / nz
    c = d - a * x - b * y
    return a, b, c

def compute_greyscale_gradient(img, grads):
    # img 已经是 0-255 float，不需要缩放
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    grads[:,:,0]= ndimage.sobel(gray,axis=0)
    grads[:,:,1]= ndimage.sobel(gray,axis=1)

def inside(x, y, lbx, lby, ubx, uby):
    return lbx <= x < ubx and lby <= y < uby

class PatchMatch:
    def __init__(self, alpha, gamma, tau_c, tau_g, window_size=35):
        self.alpha = alpha
        self.gamma = gamma
        self.tau_c = tau_c
        self.tau_g = tau_g
        self.window_size = window_size  # 【新增】支持动态窗口大小
        
        self.img_left = None
        self.img_right = None
        self.grad_left = None
        self.grad_right = None
        self.planes_left = None
        self.planes_right = None
        self.cost_left = None
        self.cost_right = None
        self.weight_left = None
        self.weight_right = None
        self.disp_left = None
        self.disp_right = None
        self.rows = 0
        self.cols = 0
        self.guidance_disp_left = None
        self.guidance_disp_right = None
        self.gc_mask_left = None
        self.gc_mask_right = None
        self.lambda_gc = 0.1

    def upsample_planes(self, coarse_planes, scale_factor=2.0):
        # 使用最近邻插值防止平面参数平滑混合
        new_a = cv2.resize(coarse_planes.a, (self.cols, self.rows), interpolation=cv2.INTER_NEAREST)
        new_b = cv2.resize(coarse_planes.b, (self.cols, self.rows), interpolation=cv2.INTER_NEAREST)
        new_c = cv2.resize(coarse_planes.c, (self.cols, self.rows), interpolation=cv2.INTER_NEAREST)
        
        # d = ax + by + c. 当坐标放大S倍，视差放大S倍，C需要放大S倍
        new_c = new_c * scale_factor
        
        target_planes = PlaneParams((self.rows, self.cols))
        target_planes.a = new_a
        target_planes.b = new_b
        target_planes.c = new_c
        return target_planes

    def plane_match_cost(self, plane_params, cx, cy, cpv):
        """单点代价计算 (用于空间传播)"""
        sign = -1 if cpv == 0 else 1
        half = self.window_size // 2
        
        if cpv == 0:
            f1, f2 = self.img_left, self.img_right
            g1, g2 = self.grad_left, self.grad_right
            weights = self.weight_left[cy, cx]
        else:
            f1, f2 = self.img_right, self.img_left
            g1, g2 = self.grad_right, self.grad_left
            weights = self.weight_right[cy, cx]
        
        y_range = np.arange(cy - half, cy + half + 1)
        x_range = np.arange(cx - half, cx + half + 1)
        y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')
        y_flat = y_grid.flatten()
        x_flat = x_grid.flatten()
        
        valid = (x_flat >= 0) & (x_flat < self.cols) & (y_flat >= 0) & (y_flat < self.rows)
        if not np.any(valid): return 0.0

        x_valid = x_flat[valid]
        y_valid = y_flat[valid]
        
        d = plane_params[0] * x_valid + plane_params[1] * y_valid + plane_params[2]
        valid_d = (d >= 0) & (d <= MAX_DISPARITY)
        cost = np.sum(~valid_d) * PLANE_PENALTY
        
        if not np.any(valid_d): return cost
        
        x_eff = x_valid[valid_d]
        y_eff = y_valid[valid_d]
        d_eff = d[valid_d]
        
        match_x = x_eff + sign * d_eff
        x0 = np.floor(match_x).astype(int)
        x1 = x0 + 1
        wx = x1 - match_x
        
        x0 = np.clip(x0, 0, self.cols - 2)
        x1 = x0+1
        y_idx = y_eff.astype(int)
        
        color_interp = wx[:, None] * f2[y_idx, x0] + (1 - wx[:, None]) * f2[y_idx, x1]
        grad_interp_x = wx * g2[y_idx, x0, 0] + (1 - wx) * g2[y_idx, x1, 0]
        grad_interp_y = wx * g2[y_idx, x0, 1] + (1 - wx) * g2[y_idx, x1, 1]
        grad_interp = np.stack([grad_interp_x, grad_interp_y], axis=-1)
        
        wy = (y_eff - cy + half).astype(int)
        wx_idx = (x_eff - cx + half).astype(int)
        w = weights[wy, wx_idx]
        
        color_cost = np.sum(np.abs(f1[y_idx, x_eff.astype(int)] - color_interp), axis=1)
        grad_cost = np.sum(np.abs(g1[y_idx, x_eff.astype(int)] - grad_interp), axis=1)
        
        color_cost = np.minimum(color_cost, self.tau_c)
        grad_cost = np.minimum(grad_cost, self.tau_g)
        
        weighted_cost = w * ((1 - self.alpha) * color_cost + self.alpha * grad_cost)
        cost += np.sum(weighted_cost)
        return cost

    def plane_match_cost_vectorized(self, a_list, b_list, c_list, x_list, y_list, cpv):
        """【关键修复】分批处理防止OOM"""
        sign = -1 if cpv == 0 else 1
        half = self.window_size // 2
        rows, cols = x_list.shape
        
        if cpv == 0:
            f1, f2 = self.img_left, self.img_right
            g1, g2 = self.grad_left, self.grad_right
            weights_full = self.weight_left
            guidance_disp_full = self.guidance_disp_left
            gc_mask_full = self.gc_mask_left
        else:
            f1, f2 = self.img_right, self.img_left
            g1, g2 = self.grad_right, self.grad_left
            weights_full = self.weight_right
            guidance_disp_full = self.guidance_disp_right
            gc_mask_full = self.gc_mask_right

        cost_map = np.zeros((rows, cols), dtype=np.float32)

        # 批处理大小 (行数)
        BATCH_SIZE = 10 
        
        y_offsets = np.arange(-half, half + 1)
        x_offsets = np.arange(-half, half + 1)
        y_offsets_grid, x_offsets_grid = np.meshgrid(y_offsets, x_offsets, indexing='ij')
        
        y_offsets_exp = y_offsets_grid[None, None, :, :]
        x_offsets_exp = x_offsets_grid[None, None, :, :]

        for r_start in range(0, rows, BATCH_SIZE):
            r_end = min(r_start + BATCH_SIZE, rows)
            curr_batch_rows = r_end - r_start
            
            sub_a = a_list[r_start:r_end, :]
            sub_b = b_list[r_start:r_end, :]
            sub_c = c_list[r_start:r_end, :]
            sub_x = x_list[r_start:r_end, :]
            sub_y = y_list[r_start:r_end, :]
            
            y_coords = sub_y[:, :, None, None]
            x_coords = sub_x[:, :, None, None]
            
            y_all = y_coords + y_offsets_exp
            x_all = x_coords + x_offsets_exp
            
            d = sub_a[:, :, None, None] * x_all + sub_b[:, :, None, None] * y_all + sub_c[:, :, None, None]
            
            valid_mask = (x_all >= 0) & (x_all < cols) & (y_all >= 0) & (y_all < rows)
            valid_d = (d >= 0) & (d <= MAX_DISPARITY)
            valid_d_mask = valid_d & valid_mask 
            
            batch_cost_map = np.zeros((curr_batch_rows, cols), dtype=np.float32)
            invalid_in_bounds = valid_mask & (~valid_d)
            batch_cost_map += np.sum(invalid_in_bounds, axis=(2, 3)) * PLANE_PENALTY
            
            if not np.any(valid_d_mask):
                cost_map[r_start:r_end, :] = batch_cost_map
                continue

            match_x = x_all + sign * d
            x0 = np.floor(match_x).astype(int)
            x1 = x0 + 1
            wx = x1 - match_x
            
            x0_safe = np.clip(x0, 0, cols - 2)
            x1_safe = x0_safe + 1
            y_safe = np.clip(y_all, 0, rows - 1).astype(int)
            x_safe = np.clip(x_all, 0, cols - 1).astype(int)

            val0 = f2[y_safe, x0_safe]
            val1 = f2[y_safe, x1_safe]
            color_interp = wx[..., None] * val0 + (1 - wx[..., None]) * val1
            
            color_orig = f1[y_safe, x_safe]
            c_diff = np.sum(np.abs(color_orig - color_interp), axis=-1)
            c_diff = np.minimum(c_diff, self.tau_c)
            
            g_val0 = g2[y_safe, x0_safe]
            g_val1 = g2[y_safe, x1_safe]
            g_interp_0 = wx * g_val0[..., 0] + (1 - wx) * g_val1[..., 0]
            g_interp_1 = wx * g_val0[..., 1] + (1 - wx) * g_val1[..., 1]
            grad_interp = np.stack([g_interp_0, g_interp_1], axis=-1)
            grad_orig = g1[y_safe, x_safe]
            g_diff = np.sum(np.abs(grad_orig - grad_interp), axis=-1)
            g_diff = np.minimum(g_diff, self.tau_g)
            
            w_batch = weights_full[r_start:r_end]
            pixel_cost = (1 - self.alpha) * c_diff + self.alpha * g_diff
            
            if guidance_disp_full is not None:
                g_disp_batch = guidance_disp_full[r_start:r_end]
                g_disp_exp = g_disp_batch[:, :, None, None]
                geo_err = np.abs(d - g_disp_exp)
                geo_err = np.minimum(geo_err, 3.0)
                
                if gc_mask_full is not None:
                    mask_batch = gc_mask_full[r_start:r_end]
                    mask_exp = mask_batch[:, :, None, None]
                    pixel_cost += self.lambda_gc * 5.0 * mask_exp * geo_err
                else:
                    pixel_cost += self.lambda_gc * 5.0 * geo_err

            weighted_cost = w_batch * pixel_cost
            weighted_cost[~valid_d_mask] = 0
            
            batch_cost_map += np.sum(weighted_cost, axis=(2, 3))
            cost_map[r_start:r_end, :] = batch_cost_map

        return cost_map

    def precompute_pixels_weights(self, frame, weights, ws):
        # 简化版：复用现有逻辑，但需注意 ws 已经是 self.window_size
        half = ws // 2
        rows, cols = frame.shape[:2]
        frame = frame.astype(np.float64) # 保持0-255范围

        dy, dx = np.meshgrid(np.arange(-half, half+1), np.arange(-half, half+1), indexing='ij')
        dy, dx = dy.flatten(), dx.flatten()
        n_offsets = len(dy)
        
        cy, cx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        cy, cx = cy.reshape(-1, 1), cx.reshape(-1, 1)
        n_centers = cy.shape[0]
        
        y = np.clip(cy + dy.reshape(1, -1), 0, rows - 1)
        x = np.clip(cx + dx.reshape(1, -1), 0, cols - 1)
        
        # 这里的 wy, wx 只是索引
        wy = dy + half
        wx = dx + half
        
        frame_vals = frame[cy[:, 0], cx[:, 0]]
        window_vals = frame[y, x]
        
        dist = np.sum(np.abs(frame_vals[:, np.newaxis, :] - window_vals), axis=2)
        # 0-255 范围内，gamma=10 才有意义
        weights_vals = np.exp(-dist / self.gamma)
        
        cy_idx = np.repeat(cy[:, 0], n_offsets)
        cx_idx = np.repeat(cx[:, 0], n_offsets)
        wy_idx = np.tile(wy, n_centers)
        wx_idx = np.tile(wx, n_centers)
        
        weights[cy_idx, cx_idx, wy_idx, wx_idx] = weights_vals.flatten()

    def set_images(self, img_left, img_right):
        # 【关键修复】不除以 255.0，保持 0-255 范围
        self.img_left = img_left.astype(np.float32)
        self.img_right = img_right.astype(np.float32)
        self.rows, self.cols = img_left.shape[:2]
        
        print("Precomputing pixels weight...")
        # 使用动态窗口大小
        wmat_sizes = (self.rows, self.cols, self.window_size, self.window_size)
        self.weight_left = np.zeros(wmat_sizes, dtype=np.float32)
        self.weight_right = np.zeros(wmat_sizes, dtype=np.float32)
        self.precompute_pixels_weights(self.img_left, self.weight_left, self.window_size)
        self.precompute_pixels_weights(self.img_right, self.weight_right, self.window_size)

        print("Evaluating images gradient...")
        self.grad_left = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        self.grad_right = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        compute_greyscale_gradient(self.img_left, self.grad_left)
        compute_greyscale_gradient(self.img_right, self.grad_right)
    
        print("Initializing random planes...")
        self.planes_left = PlaneParams((self.rows, self.cols))
        self.planes_right = PlaneParams((self.rows, self.cols))
        self.initialize_random_planes(self.planes_left, MAX_DISPARITY)
        self.initialize_random_planes(self.planes_right, MAX_DISPARITY)

        print("Evaluating initial costs...")
        self.evaluate_planes_cost()
        
    def initialize_random_planes(self, planes, max_d):
        # 保持原样
        rows, cols = planes.shape
        RAND_HALF = 0x7FFFFFFF
        nx = (np.random.uniform(-RAND_HALF, RAND_HALF,(rows,cols))) / RAND_HALF
        ny = (np.random.uniform(-RAND_HALF, RAND_HALF,(rows,cols))) / RAND_HALF
        nz = (np.random.uniform(-RAND_HALF, RAND_HALF,(rows,cols))) / RAND_HALF
        
        norm = np.sqrt(nx**2 + ny**2 + nz**2)
        norm[norm == 0] = 1.0
        nx, ny, nz = nx/norm, ny/norm, nz/norm
        nz[nz==0]=1e-8
        
        x_grid, y_grid = np.meshgrid(np.arange(cols), np.arange(rows),indexing='xy')
        z_grid = np.random.uniform(0,max_d,(rows,cols))
        planes.a = -nx / nz
        planes.b = -ny / nz
        planes.c = (nx * x_grid + ny * y_grid + nz * z_grid) / nz
    
    def evaluate_planes_cost(self):
        # 保持原样
        rows, cols = self.planes_left.shape
        x_list, y_list = np.meshgrid(np.arange(cols), np.arange(rows),indexing='xy')  
        self.cost_left = self.plane_match_cost_vectorized(self.planes_left.a, self.planes_left.b, self.planes_left.c, x_list, y_list, 0)
        self.cost_right = self.plane_match_cost_vectorized(self.planes_right.a, self.planes_right.b, self.planes_right.c, x_list, y_list, 1)

    def spatial_propagation(self, cpv, iter_even):
        # 保持原样
        if cpv == 0:
            planes = self.planes_left
            costs = self.cost_left
        else:
            planes = self.planes_right
            costs = self.cost_right
        rows, cols = self.rows, self.cols
        if iter_even:
            for y in range(rows):
                for x in range(cols):
                    if y > 0:
                        neighbor_params = planes.get_at(y-1, x)
                        new_cost = self.plane_match_cost(neighbor_params, x, y, cpv)
                        if new_cost < costs[y, x]:
                            planes.set_at(y, x, neighbor_params)
                            costs[y, x] = new_cost
                    if x > 0:
                        neighbor_params = planes.get_at(y, x-1)
                        new_cost = self.plane_match_cost(neighbor_params, x, y, cpv)
                        if new_cost < costs[y, x]:
                            planes.set_at(y, x, neighbor_params)
                            costs[y, x] = new_cost
        else:
            for y in range(rows-1, -1, -1):
                for x in range(cols-1, -1, -1):
                    if y < rows-1:
                        neighbor_params = planes.get_at(y+1, x)
                        new_cost = self.plane_match_cost(neighbor_params, x, y, cpv)
                        if new_cost < costs[y, x]:
                            planes.set_at(y, x, neighbor_params)
                            costs[y, x] = new_cost
                    if x < cols-1:
                        neighbor_params = planes.get_at(y, x+1)
                        new_cost = self.plane_match_cost(neighbor_params, x, y, cpv)
                        if new_cost < costs[y, x]:
                            planes.set_at(y, x, neighbor_params)
                            costs[y, x] = new_cost
    
    def plane_refinement_vectorized(self, cpv, max_iterations=8):
        # 保持原样
        if cpv == 0:
            planes = self.planes_left
            costs = self.cost_left
        else:
            planes = self.planes_right
            costs = self.cost_right
        rows, cols = self.rows, self.cols
        
        a_curr = planes.a.copy()
        b_curr = planes.b.copy()
        c_curr = planes.c.copy()
        cost_curr = costs.copy()
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        scale_factors = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125] # 稍微减少级数提高速度
        
        for scale_idx, scale in enumerate(scale_factors):
            max_dz = (MAX_DISPARITY / 2) * scale
            max_dn=1.0
            num_trials = 3
            for trial in range(num_trials):
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
                new_nx, new_ny, new_nz = new_nx/norm, new_ny/norm, new_nz/norm
                a_new,b_new,c_new=normal_to_params(nx,ny,nz, z, x_grid, y_grid)
                new_costs = self.plane_match_cost_vectorized(a_new, b_new, c_new, x_grid, y_grid, cpv)
                improved_mask = new_costs < cost_curr
                if np.any(improved_mask):
                    a_curr[improved_mask] = a_new[improved_mask]
                    b_curr[improved_mask] = b_new[improved_mask]
                    c_curr[improved_mask] = c_new[improved_mask]
                    cost_curr[improved_mask] = new_costs[improved_mask]
        planes.a[:, :] = a_curr
        planes.b[:, :] = b_curr
        planes.c[:, :] = c_curr
        costs[:, :] = cost_curr
    
    def view_propagation_vectorized(self, cpv):
        # 保持原样
        rows, cols = self.rows, self.cols
        if cpv == 0:
            src_planes = self.planes_left
            dst_planes = self.planes_right
            dst_costs = self.cost_right
            sign = -1
        else:
            src_planes = self.planes_right
            dst_planes = self.planes_left
            dst_costs = self.cost_left
            sign = 1
        
        a_src, b_src, c_src = src_planes.a, src_planes.b, src_planes.c
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        d = a_src * x_grid + b_src * y_grid + c_src
        target_x = x_grid + sign * d
        target_x_int = np.round(target_x).astype(int)
        valid_mask = (target_x_int >= 0) & (target_x_int < cols)
        
        if not np.any(valid_mask): return
        
        y_valid = y_grid[valid_mask]
        x_valid = x_grid[valid_mask]
        target_x_valid = target_x_int[valid_mask]
        a_dst = a_src[valid_mask]
        b_dst = b_src[valid_mask]
        c_dst = c_src[valid_mask]
        d_dst = a_dst*x_valid+b_dst*y_valid+c_dst
        nx,ny,nz = params_to_normal(a_dst,b_dst,c_dst)
        a_dst,b_dst,c_dst=normal_to_params(nx,ny,nz,d_dst,target_x_valid,y_valid)
        
        a_temp = np.zeros((rows, cols))
        b_temp = np.zeros((rows, cols))
        c_temp = np.zeros((rows, cols))
        cx_temp = np.zeros((rows, cols), dtype=int)
        cy_temp = np.zeros((rows, cols), dtype=int)
        
        a_temp[y_valid, x_valid] = a_dst
        b_temp[y_valid, x_valid] = b_dst
        c_temp[y_valid, x_valid] = c_dst
        cx_temp[y_valid, x_valid] = target_x_valid
        cy_temp[y_valid, x_valid] = y_valid
        
        cost_map = self.plane_match_cost_vectorized(a_temp, b_temp, c_temp, cx_temp, cy_temp, 1-cpv)
        new_costs = cost_map[y_valid, x_valid]
        dst_costs_flat = dst_costs[y_valid, target_x_valid]
        update_mask = new_costs < dst_costs_flat
        
        if not np.any(update_mask): return
        
        update_y = y_valid[update_mask]
        update_x = target_x_valid[update_mask]
        dst_planes.a[update_y, update_x] = a_dst[update_mask]
        dst_planes.b[update_y, update_x] = b_dst[update_mask]
        dst_planes.c[update_y, update_x] = c_dst[update_mask]
        dst_costs[update_y, update_x] = new_costs[update_mask]
    
    def compute_disparity_maps(self):
        # 保持原样
        self.disp_left = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.disp_right = np.zeros((self.rows, self.cols), dtype=np.float32)
        y_coords, x_coords = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing='ij')
        self.disp_left = self.planes_left.a * x_coords + self.planes_left.b * y_coords + self.planes_left.c
        self.disp_right = self.planes_right.a * x_coords + self.planes_right.b * y_coords + self.planes_right.c

    def fill_invalid_pixels(self, y, x, planes, validity):
        # 保持原样
        x_lft, x_rgt = x - 1, x + 1
        while x_lft >= 0 and not validity[y, x_lft]: x_lft -= 1
        while x_rgt < self.cols and not validity[y, x_rgt]: x_rgt += 1
        
        best_plane_x = x
        if x_lft >= 0 and x_rgt < self.cols:
            disp_l = x*planes.a[y][x_lft]+y*planes.b[y][x_lft]+planes.c[y][x_lft]
            disp_r = x*planes.a[y][x_rgt]+y*planes.b[y][x_rgt]+planes.c[y][x_rgt]
            best_plane_x = x_lft if disp_l < disp_r else x_rgt
        elif x_lft >= 0: best_plane_x = x_lft
        elif x_rgt < self.cols: best_plane_x = x_rgt
        planes.set_at(y, x, planes.get_at(y,best_plane_x))
    
    def weighted_median_filter(self, cx, cy, disparity, weights, valid, ws, use_invalid):
        # 保持原样 (需注意 ws 是从类属性传还是参数传，这里保持参数)
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
                if i == 0: disparity[cy, cx] = d
                else: disparity[cy, cx] = (disps_w[i-1][1] + d) / 2.0
                break

    def process(self, iterations=3, coarse_pm=None):
        # 保持原样，ACMM 核心逻辑正确
        print("Processing...")
        if coarse_pm is not None:
            print("  [ACMM] Upsampling planes from coarser scale...")
            self.planes_left = self.upsample_planes(coarse_pm.planes_left, scale_factor=2.0)
            self.planes_right = self.upsample_planes(coarse_pm.planes_right, scale_factor=2.0)
            
            x_list, y_list = np.meshgrid(np.arange(self.cols), np.arange(self.rows), indexing='xy')
            c_init_left = self.plane_match_cost_vectorized(self.planes_left.a, self.planes_left.b, self.planes_left.c, x_list, y_list, 0)
            c_init_right = self.plane_match_cost_vectorized(self.planes_right.a, self.planes_right.b, self.planes_right.c, x_list, y_list, 1)
            
            d_l_coarse = self.planes_left.a * x_list + self.planes_left.b * y_list + self.planes_left.c
            d_r_coarse = self.planes_right.a * x_list + self.planes_right.b * y_list + self.planes_right.c
            self.guidance_disp_left = d_l_coarse
            self.guidance_disp_right = d_r_coarse
            
            detail_thresh = np.mean(c_init_left) * 1.5 
            self.gc_mask_left = (c_init_left < detail_thresh).astype(np.float32)
            detail_thresh_r = np.mean(c_init_right) * 1.5
            self.gc_mask_right = (c_init_right < detail_thresh_r).astype(np.float32)
            print(f"  [ACMM] Detail Restorer: {np.sum(1-self.gc_mask_left)} pixels identified as potential details.")
            self.cost_left = c_init_left
            self.cost_right = c_init_right
        else:
            print("  [ACMM] Coarsest scale detected. Using random initialization.")
        
        for iter in range(iterations):
            print(f"Iteration {iter+1}/{iterations}")
            iter_even = (iter % 2 == 0)           
            for view in range(2):
                self.spatial_propagation(view, iter_even)
                self.plane_refinement_vectorized(view)
                self.view_propagation_vectorized(view)
        print("Computing disparity maps...")
        self.compute_disparity_maps()
    
    def post_process(self):
        # 保持原样
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
                if not lft_validity[y, x]: self.fill_invalid_pixels(y, x, self.planes_left, lft_validity)
                if not rgt_validity[y, x]: self.fill_invalid_pixels(y, x, self.planes_right, rgt_validity)
        self.compute_disparity_maps()
        for x in range(self.cols):
            for y in range(self.rows):
                self.weighted_median_filter(x, y, self.disp_left, self.weight_left, lft_validity, self.window_size, False)
                self.weighted_median_filter(x, y, self.disp_right, self.weight_right, rgt_validity, self.window_size, False)
        return self.disp_left,self.disp_right