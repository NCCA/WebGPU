import nccapy
import math

class FirstPersonCamera :
    def __init__(self,eye,look,up,fov) :
        self.view=nccapy.Mat4()
        self.eye=eye
        self.look=look
        self.world_up=up
        self.front=nccapy.Vec3()
        self.up=nccapy.Vec3()
        self.right=nccapy.Vec3()
        self.yaw=-90.0
        self.pitch=0.0
        self.speed=2.5
        self.sensitivity=0.1
        self.zoom=45.0
        self.near=0.1
        self.far=100.0
        self.aspect=1.2
        self.fov
        self.projection=set_projection(fov,aspect,near,far)
        
    def _update_camera_vectors(self) :
        ...
    def set_projection(self,fov,aspect,near,far) :
        self.fov = math.radians(self.fov)
        f = 1.0 / math.tan(self.fov / 2) 
        self.near=near
        self.far=far
        self.aspect=aspect
        return nccapy.Mat4.from_list(
            [[f / self.aspect, 0,  0,                           0],
            [0,          f,  0,                           0],
            [0,          0,  z_far / (z_near - z_far),   -1],
            [0,          0,  (z_near * z_far) / (z_near - z_far),  0]]
        )




