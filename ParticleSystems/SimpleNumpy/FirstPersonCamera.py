import nccapy
import math


class FirstPersonCamera:
    """
    A class representing a first-person camera.

    This class provides functionality for a first-person camera, including movement,
    rotation, and projection matrix calculation.

    Attributes:
        eye (nccapy.Vec3): The position of the camera.
        look (nccapy.Vec3): The point the camera is looking at.
        world_up (nccapy.Vec3): The world's up vector.
        front (nccapy.Vec3): The front direction vector of the camera.
        up (nccapy.Vec3): The up direction vector of the camera.
        right (nccapy.Vec3): The right direction vector of the camera.
        yaw (float): The yaw angle of the camera.
        pitch (float): The pitch angle of the camera.
        speed (float): The movement speed of the camera.
        sensitivity (float): The mouse sensitivity.
        zoom (float): The zoom level of the camera.
        near (float): The near clipping plane.
        far (float): The far clipping plane.
        aspect (float): The aspect ratio.
        fov (float): The field of view.
        projection (nccapy.Mat4): The projection matrix.
        view (nccapy.Mat4): The view matrix.
    """

    def __init__(
        self, eye: nccapy.Vec3, look: nccapy.Vec3, up: nccapy.Vec3, fov: float
    ) -> None:
        """
        Initialize the FirstPersonCamera.

        Args:
            eye (nccapy.Vec3): The position of the camera.
            look (nccapy.Vec3): The point the camera is looking at.
            up (nccapy.Vec3): The world's up vector.
            fov (float): The field of view.
        """
        self.eye: nccapy.Vec3 = eye
        self.look: nccapy.Vec3 = look
        self.world_up: nccapy.Vec3 = up
        self.front: nccapy.Vec3 = nccapy.Vec3()
        self.up: nccapy.Vec3 = nccapy.Vec3()
        self.right: nccapy.Vec3 = nccapy.Vec3()
        self.yaw: float = -90.0
        self.pitch: float = 0.0
        self.speed: float = 2.5
        self.sensitivity: float = 0.1
        self.zoom: float = 45.0
        self.near: float = 0.1
        self.far: float = 100.0
        self.aspect: float = 1.2
        self.fov: float = fov
        self._update_camera_vectors()
        self.projection: nccapy.Mat4 = self.set_projection(
            self.fov, self.aspect, self.near, self.far
        )
        self.view: nccapy.Mat4 = self._look_at(self.eye, self.eye + self.front, self.up)

    def __str__(self) -> str:
        return f"Camera {self.eye} {self.look} {self.world_up} {self.fov}"

    def __repr__(self) -> str:
        return f"Camera {self.eye} {self.look} {self.world_up} {self.fov}"

    def process_mouse_movement(
        self, diffx: float, diffy: float, _constrainPitch: bool = True
    ) -> None:
        """
        Process mouse movement to update the camera's direction vectors.

        Args:
            diffx (float): The difference in the x-coordinate of the mouse movement.
            diffy (float): The difference in the y-coordinate of the mouse movement.
            _constrainPitch (bool, optional): Whether to constrain the pitch angle. Defaults to True.
        """
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

    def _update_camera_vectors(self) -> None:
        """
        Update the camera's direction vectors based on the current yaw and pitch angles.
        """
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

    def set_projection(
        self, fov: float, aspect: float, near: float, far: float
    ) -> nccapy.Mat4:
        """
        Set the projection matrix for the camera.

        Args:
            fov (float): The field of view.
            aspect (float): The aspect ratio.
            near (float): The near clipping plane.
            far (float): The far clipping plane.

        Returns:
            nccapy.Mat4: The projection matrix.
        """
        gl_to_web = nccapy.Mat4.from_list(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return gl_to_web @ nccapy.perspective(fov, aspect, near, far)

    def move(self, x: float, y: float, delta: float) -> None:
        """
        Move the camera based on input directions.

        Args:
            x (float): The movement in the x-direction.
            y (float): The movement in the y-direction.
            delta (float): The amount to move the camera.
        """
        velocity = self.speed * delta
        self.eye += self.front * velocity * x
        self.eye += self.right * velocity * y
        self._update_camera_vectors()

    def get_vp(self) -> nccapy.Mat4:
        """
        Get the view-projection matrix.

        Returns:
            nccapy.Mat4: The view-projection matrix.
        """
        return self.projection @ self.view

    def _look_at(
        self, eye: nccapy.Vec3, look: nccapy.Vec3, up: nccapy.Vec3
    ) -> nccapy.Mat4:
        """
        Create a look-at matrix.

        Args:
            eye (nccapy.Vec3): The position of the camera.
            look (nccapy.Vec3): The point the camera is looking at.
            up (nccapy.Vec3): The up vector.

        Returns:
            nccapy.Mat4: The look-at matrix.
        """
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
