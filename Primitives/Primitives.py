"""
WebGPU primitive generation and drawing functions
In this class we can generate a pipeline for drawing our data for the most part it will be 
x,y,z nx,ny,nz and u,v data in a flat numpy array. 
We need to create the data first which is stored in a map as part of the class, we can then call draw
which will generate a pipeline for this object and draw into the current context.
"""


#!/usr/bin/env python3


import numpy as np
import wgpu


class _primitive:
    def __init__(self):
        self.buffer = None
        self.draw_size = 0
        self.type = None


class Primitives:
    # this is effectively a static class so we can use it to store data
    # and generate pipelines for drawing
    _primitives = {}


    @classmethod
    def _gen_prim(cls, device, data, draw_type):
        vertex_buffer = device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )
        prim = _primitive()
        prim.buffer = vertex_buffer
        prim.draw_size = len(data)//8
        prim.draw_type = draw_type
        return prim

    @classmethod
    def load_default_primitives(cls, device):
        prims=[["cube","PrimData/cube.npy"],
               ["dodecahedron","PrimData/dodecahedron.npy"],
               ["troll","PrimData/troll.npy"],
               ["teapot","PrimData/teapot.npy"],
               ["bunny","PrimData/bunny.npy"] 
               ]
        for p in prims:
            prim_data=np.load(p[1])
            cls._primitives[p[0]] = cls._gen_prim(device, prim_data, "triangle")
        
        # prim_data=np.load("PrimData/cube.npy")
        # cls._primitives["cube"] = cls._gen_prim(device, prim_data, "triangle")
        # prim_data=np.load("PrimData/dodecahedron.npy")
        # cls._primitives["dodecahedron"] = cls._gen_prim(device, prim_data, "triangle")


    @classmethod
    def create_line_grid(cls, name, device, width, depth, steps):
        # Calculate the step size for each grid value
        wstep = width / steps
        ws2 = width / 2.0
        v1 = -ws2

        dstep = depth / steps
        ds2 = depth / 2.0
        v2 = -ds2

        # Create a list to store the vertex data
        data = []

        for i in range(steps + 1):
            # Vertex 1 x, y, z
            data.append([-ws2, 0.0, v1])
            # Vertex 2 x, y, z
            data.append([ws2, 0.0, v1])

            # Vertex 1 x, y, z
            data.append([v2, 0.0, ds2])
            # Vertex 2 x, y, z
            data.append([v2, 0.0, -ds2])

            # Now change our step value
            v1 += wstep
            v2 += dstep

        # Convert the list to a NumPy array
        data_array = np.array(data, dtype=np.float32)

        vertex_buffer = device.create_buffer_with_data(
            data=data_array, usage=wgpu.BufferUsage.VERTEX
        )
        prim = _primitive()
        prim.buffer = vertex_buffer
        prim.draw_size = len(data_array)
        prim.draw_type = "line"
        cls._primitives[name] = prim

    @classmethod
    def draw(cls, render_pass, name):
        try:
            prim = cls._primitives[name]
            render_pass.set_vertex_buffer(0, prim.buffer)
            if prim.draw_type == "line":
                render_pass.draw(prim.draw_size, 1, 0, 0)
            elif prim.draw_type == "triangle":
                render_pass.draw(prim.draw_size, 1, 0, 0)

        except KeyError:
            print(f"Primitive {name} not found")
            return


    @classmethod
    def create_sphere(cls, name, device,radius, precision):
        # Sphere code based on a function Written by Paul Bourke.
        # http://astronomy.swin.edu.au/~pbourke/opengl/sphere/
        # the next part of the code calculates the P,N,UV of the sphere for tri_strips

        theta1 = 0.0
        theta2 = 0.0
        theta3 = 0.0

        # Disallow a negative number for radius.
        if radius < 0.0:
            radius = -radius

        # Disallow a negative number for precision.
        if precision < 4:
            precision = 4

        # Create a numpy array to store our verts
        data = []

        for i in range(precision // 2):
            theta1 = i * 2.0 * np.pi / precision - np.pi / 2.0
            theta2 = (i + 1) * 2.0 * np.pi / precision - np.pi / 2.0

            for j in range(precision + 1):
                theta3 = j * 2.0 * np.pi / precision

                nx = np.cos(theta2) * np.cos(theta3)
                ny = np.sin(theta2)
                nz = np.cos(theta2) * np.sin(theta3)
                x = radius * nx
                y = radius * ny
                z = radius * nz
                u = j / precision
                v = 2.0 * (i + 1) / precision
                data.append([x, y, z, nx, ny, nz, u, v])

                nx = np.cos(theta1) * np.cos(theta3)
                ny = np.sin(theta1)
                nz = np.cos(theta1) * np.sin(theta3)
                x = radius * nx
                y = radius * ny
                z = radius * nz
                u = j / precision
                v = 2.0 * i / precision
                data.append([x, y, z, nx, ny, nz, u, v])

        data_array=np.array(data, dtype=np.float32)

        vertex_buffer = device.create_buffer_with_data(
            data=data_array, usage=wgpu.BufferUsage.VERTEX
        )
        prim = _primitive()
        prim.buffer = vertex_buffer
        prim.draw_size = len(data_array)
        prim.draw_type = "triangle"
        cls._primitives[name] = prim





if __name__ == "__main__":
    print("Primitives")
