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

def _circle_table(n):
    # Determine the angle between samples
    angle = 2.0 * np.pi / (n if n != 0 else 1)
    
    # Allocate list for n samples, plus duplicate of first entry at the end
    cs = np.zeros((n + 1, 2), dtype=np.float32)
    
    # Compute cos and sin around the circle
    cs[0, 0] = 1.0  # cost
    cs[0, 1] = 0.0  # sint

    for i in range(1, n):
        cs[i, 1] = np.sin(angle * i)  # sint
        cs[i, 0] = np.cos(angle * i)  # cost
    
    # Last sample is duplicate of the first
    cs[n, 1] = cs[0, 1]  # sint
    cs[n, 0] = cs[0, 0]  # cost
    
    return cs




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
        prims=np.load("PrimData/Primitives.npz")
        print(prims)
        for p in prims.items():
            prim_data=p[1]
            cls._primitives[p[0]] = cls._gen_prim(device, prim_data, "triangle")
        


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
    def create_sphere(cls, name, device, radius, precision):
        # Sphere code based on a function Written by Paul Bourke.
        # http://astronomy.swin.edu.au/~pbourke/opengl/sphere/
        # the next part of the code calculates the P,N,UV of the sphere for triangles

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

            for j in range(precision):
                theta3 = j * 2.0 * np.pi / precision
                theta4 = (j + 1) * 2.0 * np.pi / precision

                # First triangle
                nx1 = np.cos(theta2) * np.cos(theta3)
                ny1 = np.sin(theta2)
                nz1 = np.cos(theta2) * np.sin(theta3)
                x1 = radius * nx1
                y1 = radius * ny1
                z1 = radius * nz1
                u1 = j / precision
                v1 = 2.0 * (i + 1) / precision
                data.append([x1, y1, z1, nx1, ny1, nz1, u1, v1])

                nx2 = np.cos(theta1) * np.cos(theta3)
                ny2 = np.sin(theta1)
                nz2 = np.cos(theta1) * np.sin(theta3)
                x2 = radius * nx2
                y2 = radius * ny2
                z2 = radius * nz2
                u2 = j / precision
                v2 = 2.0 * i / precision
                data.append([x2, y2, z2, nx2, ny2, nz2, u2, v2])

                nx3 = np.cos(theta1) * np.cos(theta4)
                ny3 = np.sin(theta1)
                nz3 = np.cos(theta1) * np.sin(theta4)
                x3 = radius * nx3
                y3 = radius * ny3
                z3 = radius * nz3
                u3 = (j + 1) / precision
                v3 = 2.0 * i / precision
                data.append([x3, y3, z3, nx3, ny3, nz3, u3, v3])

                # Second triangle
                nx4 = np.cos(theta2) * np.cos(theta4)
                ny4 = np.sin(theta2)
                nz4 = np.cos(theta2) * np.sin(theta4)
                x4 = radius * nx4
                y4 = radius * ny4
                z4 = radius * nz4
                u4 = (j + 1) / precision
                v4 = 2.0 * (i + 1) / precision
                data.append([x4, y4, z4, nx4, ny4, nz4, u4, v4])

                data.append([x1, y1, z1, nx1, ny1, nz1, u1, v1])
                data.append([x3, y3, z3, nx3, ny3, nz3, u3, v3])

        data_array = np.array(data, dtype=np.float32)

        vertex_buffer = device.create_buffer_with_data(
            data=data_array, usage=wgpu.BufferUsage.VERTEX
        )
        prim = _primitive()
        prim.buffer = vertex_buffer
        prim.draw_size = len(data_array)
        prim.draw_type = "triangle"
        cls._primitives[name] = prim

    @classmethod
    def create_cone(cls, name, device, base, height, slices, stacks):
        z_step = height / (stacks if stacks > 0 else 1)
        r_step = base / (stacks if stacks > 0 else 1)

        cosn = height / np.sqrt(height * height + base * base)
        sinn = base / np.sqrt(height * height + base * base)

        cs = _circle_table(slices)

        z0 = 0.0
        z1 = z_step

        r0 = base
        r1 = r0 - r_step

        du = 1.0 / stacks
        dv = 1.0 / slices

        u = 1.0
        v = 1.0

        data = []

        for i in range(stacks):
            for j in range(slices):
                theta1 = j * 2.0 * np.pi / slices
                theta2 = (j + 1) * 2.0 * np.pi / slices

                # First triangle
                d1 = [0] * 8
                d1[6] = u
                d1[7] = v
                d1[3] = cs[j, 0] * cosn  # nx
                d1[4] = cs[j, 1] * sinn  # ny
                d1[5] = sinn             # nz
                d1[0] = cs[j, 0] * r0    # x
                d1[1] = cs[j, 1] * r0    # y
                d1[2] = z0               # z
                data.append(d1)

                d2 = [0] * 8
                d2[6] = u
                d2[7] = v - dv
                d2[3] = cs[j, 0] * cosn  # nx
                d2[4] = cs[j, 1] * sinn  # ny
                d2[5] = sinn             # nz
                d2[0] = cs[j, 0] * r1    # x
                d2[1] = cs[j, 1] * r1    # y
                d2[2] = z1               # z
                data.append(d2)

                d3 = [0] * 8
                d3[6] = u - du
                d3[7] = v - dv
                d3[3] = cs[j + 1, 0] * cosn  # nx
                d3[4] = cs[j + 1, 1] * sinn  # ny
                d3[5] = sinn                 # nz
                d3[0] = cs[j + 1, 0] * r1    # x
                d3[1] = cs[j + 1, 1] * r1    # y
                d3[2] = z1                   # z
                data.append(d3)

                # Second triangle
                d4 = [0] * 8
                d4[6] = u
                d4[7] = v
                d4[3] = cs[j, 0] * cosn  # nx
                d4[4] = cs[j, 1] * sinn  # ny
                d4[5] = sinn             # nz
                d4[0] = cs[j, 0] * r0    # x
                d4[1] = cs[j, 1] * r0    # y
                d4[2] = z0               # z
                data.append(d4)

                d5 = [0] * 8
                d5[6] = u - du
                d5[7] = v - dv
                d5[3] = cs[j + 1, 0] * cosn  # nx
                d5[4] = cs[j + 1, 1] * sinn  # ny
                d5[5] = sinn                 # nz
                d5[0] = cs[j + 1, 0] * r1    # x
                d5[1] = cs[j + 1, 1] * r1    # y
                d5[2] = z1                   # z
                data.append(d5)

                d6 = [0] * 8
                d6[6] = u - du
                d6[7] = v
                d6[3] = cs[j + 1, 0] * cosn  # nx
                d6[4] = cs[j + 1, 1] * sinn  # ny
                d6[5] = sinn                 # nz
                d6[0] = cs[j + 1, 0] * r0    # x
                d6[1] = cs[j + 1, 1] * r0    # y
                d6[2] = z0                   # z
                data.append(d6)

                u -= du

            v -= dv
            u = 1.0
            z0 = z1
            z1 += z_step
            r0 = r1
            r1 -= r_step

        data_array = np.array(data, dtype=np.float32)

        vertex_buffer = device.create_buffer_with_data(
            data=data_array, usage=wgpu.BufferUsage.VERTEX
        )
        prim = _primitive()
        prim.buffer = vertex_buffer
        prim.draw_size = len(data_array) // 8
        prim.draw_type = "triangle"
        cls._primitives[name] = prim




if __name__ == "__main__":
    print("Primitives")
