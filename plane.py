import numpy as np
import cv2

class Plane:
    def __init__(self, point=None, normal=None):
        self.point = np.zeros(3, dtype=np.float32) if point is None else point
        self.normal = np.zeros(3, dtype=np.float32) if normal is None else normal
        self.coeff = np.zeros(3, dtype=np.float32)
        
        if point is not None and normal is not None:
            a = -self.normal[0] / self.normal[2] if self.normal[2] != 0 else 0.0
            b = -self.normal[1] / self.normal[2] if self.normal[2] != 0 else 0.0
            c = (np.sum(self.normal * self.point) / self.normal[2]) if self.normal[2] != 0 else 0.0
            self.coeff = np.array([a, b, c], dtype=np.float32)
    
    def __getitem__(self, idx):
        return self.coeff[idx]
    
    def __call__(self):
        return self.coeff
    
    def get_point(self):
        return self.point
    
    def get_normal(self):
        return self.normal
    
    def get_coeff(self):
        return self.coeff
    
    def view_transform(self, x, y, sign, qx, qy):
        z = self.coeff[0] * x + self.coeff[1] * y + self.coeff[2]
        qx[0] = x + sign * z
        qy[0] = y
        
        p = np.array([qx[0], qy[0], z], dtype=np.float32)
        return Plane(p, self.normal)