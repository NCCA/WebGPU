import nccapy
import math

class FirstPersonCamera :
    def __init__(self,eye,look,up,fov) :
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
        self.fov=fov
        self.projection=self.set_projection(self.fov,self.aspect,self.near,self.far)
        self.view=self._look_at(self.eye,self.look,self.world_up)

    def _update_camera_vectors(self) :
        pitch = math.radians(self.pitch)
        yaw = math.radians(self.yaw)
        self.front.x = math.cos(yaw) * math.cos(pitch)
        self.front.m_y = math.sin(pitch)
        self.front.m_z = math.sin(yaw) * math.cos(pitch)
        self.front=self.front.normalize()
        # Also re-calculate the Right and Up vector
        self.right = self.front.cross(self.world_up)  
        self.up    = self.right.cross(self.front)
        # normalize as fast movement can cause issues
        self.right=self.right.normalize()
        self.front=self.front.normalize()
        self.view=self._lookAt(m_eye, m_eye + m_front, m_up)

    def set_projection(self,fov,aspect,near,far) :
        self.fov = math.radians(self.fov)
        f = 1.0 / math.tan(self.fov / 2) 
        self.near=near
        self.far=far
        self.aspect=aspect
        return nccapy.Mat4.from_list(
            [[f / self.aspect, 0,  0,                           0],
            [0,          f,  0,                           0],
            [0,          0,  self.far / (self.near - self.far),   -1],
            [0,          0,  (self.near * self.far) / (self.near - self.far),  0]]
        )


    def move(self,x,y,delta) :
        velocity = self.speed * _deltaTime
        self.eye += self.front * velocity*x
        self.eye += self.right * velocity*y
        self.updateCameraVectors()

    
    
    



    def get_vp(self) :
        return self.projection @ self.view    

    def _look_at(self,eye, look, up):
        """
        Calculate 4x4 matrix for camera lookAt
        """

        n = look - eye
        v = n.cross(up)
        u = v.cross(n)
        n.normalize()
        v.normalize()
        u.normalize()
        result = nccapy.Mat4.identity()
        result.m[0][0] = v.x
        result.m[1][0] = v.y
        result.m[2][0] = v.z
        result.m[0][1] = u.x
        result.m[1][1] = u.y
        result.m[2][1] = u.z
        result.m[0][2] = -n.x
        result.m[1][2] = -n.y
        result.m[2][2] = n.z
        result.m[3][0] = -eye.dot(v)
        result.m[3][1] = -eye.dot(u)
        result.m[3][2] = eye.dot(n)
        return result



