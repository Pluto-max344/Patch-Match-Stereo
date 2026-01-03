import numpy as np
import cv2
import random
from plane import Plane

WINDOW_SIZE = 35
MAX_DISPARITY = 60
PLANE_PENALTY = 120

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

def compute_greyscale_gradient(frame, grad):
    scale = 1
    delta = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x_grad = np.zeros_like(gray, dtype=np.float32)
    y_grad = np.zeros_like(gray, dtype=np.float32)
    
    cv2.Sobel(gray, x_grad, cv2.CV_32F, 1, 0, 3, scale, delta, cv2.BORDER_DEFAULT)
    cv2.Sobel(gray, y_grad, cv2.CV_32F, 0, 1, 3, scale, delta, cv2.BORDER_DEFAULT)
    
    x_grad /= 8.0
    y_grad /= 8.0
    
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            grad[y, x] = [x_grad[y, x], y_grad[y, x]]

def inside(x, y, lbx, lby, ubx, uby):
    return lbx <= x < ubx and lby <= y < uby

def disparity(x, y, plane):
    return plane[0] * x + plane[1] * y + plane[2]

def weight(p, q, gamma=10.0):
    return np.exp(-np.linalg.norm(p - q, ord=1) / gamma)

def vec_average(x, y, wx):
    return wx * x + (1 - wx) * y

class PatchMatch:
    def __init__(self, alpha, gamma, tau_c, tau_g):
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
    
    def dissimilarity(self, pp, qq, pg, qg):
        cost_c = np.linalg.norm(pp - qq, ord=1)
        cost_g = np.linalg.norm(pg - qg, ord=1)
        cost_c = min(cost_c, self.tau_c)
        cost_g = min(cost_g, self.tau_g)
        return (1 - self.alpha) * cost_c + self.alpha * cost_g
    
    def plane_match_cost(self, p, cx, cy, ws, cpv):
        sign = -1 + 2 * cpv
        cost = 0.0
        half = ws // 2
        
        f1 = self.views[cpv]
        f2 = self.views[1 - cpv]
        g1 = self.grads[cpv]
        g2 = self.grads[1 - cpv]
        w1 = self.weigs[cpv]
        
        for x in range(cx - half, cx + half + 1):
            for y in range(cy - half, cy + half + 1):
                if not inside(x, y, 0, 0, f1.shape[1], f1.shape[0]):
                    continue
                
                dsp = disparity(x, y, p)
                if dsp < 0 or dsp > MAX_DISPARITY:
                    cost += PLANE_PENALTY
                else:
                    match = x + sign * dsp
                    x_match = int(match)
                    wm = 1 - (match - x_match)
                    
                    x_match = max(0, min(x_match, f1.shape[1] - 2))
                    
                    mcolo = vec_average(f2[y, x_match], f2[y, x_match + 1], wm)
                    mgrad = vec_average(g2[y, x_match], g2[y, x_match + 1], wm)
                    
                    wy = y - cy + half
                    wx = x - cx + half
                    w = w1[cy, cx, wy, wx]
                    cost += w * self.dissimilarity(f1[y, x], mcolo, g1[y, x], mgrad)
        
        return cost
    
    def precompute_pixels_weights(self, frame, weights, ws):
        half = ws // 2
        rows, cols = frame.shape[:2]
        
        for cx in range(cols):
            for cy in range(rows):
                for x in range(cx - half, cx + half + 1):
                    for y in range(cy - half, cy + half + 1):
                        if inside(x, y, 0, 0, cols, rows):
                            wy = y - cy + half
                            wx = x - cx + half
                            weights[cy, cx, wy, wx] = weight(frame[cy, cx], frame[y, x], self.gamma)
    
    def planes_to_disparity(self, planes, disp):
        for x in range(self.cols):
            for y in range(self.rows):
                disp[y, x] = disparity(x, y, planes(y, x))
    
    def initialize_random_planes(self, planes, max_d):
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
        for y in range(self.rows):
            for x in range(self.cols):
                self.costs[cpv][y, x] = self.plane_match_cost(self.planes[cpv](y, x), x, y, WINDOW_SIZE, cpv)
    
    def spatial_propagation(self, x, y, cpv, iter):
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
        self.spatial_propagation(x, y, cpv, iter)
        self.plane_refinement(x, y, cpv, MAX_DISPARITY / 2, 1.0, 0.1)
        self.view_propagation(x, y, cpv)
    
    def fill_invalid_pixels(self, y, x, planes, validity):
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
    
    def get_left_disparity_map(self):
        return self.disps[0]
    
    def get_right_disparity_map(self):
        return self.disps[1]