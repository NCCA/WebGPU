import nccapy
import math
import numpy as np


class FirstPersonCamera:
    def __init__(self, eye, look, up, fov):
        self.eye = eye
        self.look = look
        self.world_up = up
        self.front = nccapy.Vec3()
        self.up = nccapy.Vec3()
        self.right = nccapy.Vec3()
        self.front = nccapy.Vec3()
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 2.5
        self.sensitivity = 0.1
        self.zoom = 45.0
        self.near = 0.1
        self.far = 100.0
        self.aspect = 1.2
        self.fov = fov
        self.projection = self.set_projection(self.fov, self.aspect, self.near, self.far)
        self.view = self._look_at(self.eye, self.look, self.world_up)

    def process_mouse_movement(self, diffx, diffy, _constrainPitch=True):
        diffx *= self.sensitivity
        diffy *= self.sensitivity

        self.yaw += diffx
        self.pitch += diffy

        # Make sure that when pitch is out of bounds, screen doesn't get flipped
        if _constrainPitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0

        self._update_camera_vectors()

    def _update_camera_vectors(self):
        pitch = math.radians(self.pitch)
        yaw = math.radians(self.yaw)
        self.front.x = math.cos(yaw) * math.cos(pitch)
        self.front.y = math.sin(pitch)
        self.front.z = math.sin(yaw) * math.cos(pitch)
        self.front.normalize()
        # Also re-calculate the Right and Up vector
        self.right = self.front.cross(self.world_up)
        self.up = self.right.cross(self.front)
        # normalize as fast movement can cause issues
        self.right.normalize()
        self.front.normalize()
        self.view = self._look_at(self.eye, self.eye + self.front, self.up)

    def set_projection(self, fov, aspect, near, far):
        gl_to_web = nccapy.Mat4.from_list(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return gl_to_web @ nccapy.perspective(self.fov, self.aspect, self.near, self.far)

    def move(self, x, y, delta):
        velocity = self.speed * delta
        self.eye += self.front * velocity * x
        self.eye += self.right * velocity * y
        self._update_camera_vectors()

    def get_vp(self):
        return self.projection @ self.view

    def _look_at(self, eye, look, up):
        #        return nccapy.look_at(eye,look,up)
        n = look - eye  # front
        v = n.cross(up)  # side
        u = v.cross(n)  # up
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
        result.m[2][2] = -n.z
        result.m[3][0] = -eye.dot(v)
        result.m[3][1] = -eye.dot(u)
        result.m[3][2] = eye.dot(n)
        return result
