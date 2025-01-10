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
        self.draw_size=0
        self.type=None


class Primitives:
    # this is effectively a static class so we can use it to store data
    # and generate pipelines for drawing
    _primitives={}

    @classmethod
    def create_line_grid(cls,name,device,width, depth, steps):
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
        prim=_primitive()
        prim.buffer=vertex_buffer
        prim.draw_size=len(data_array)
        prim.draw_type="line"
        cls._primitives[name]=prim
   
    @classmethod
    def draw(cls,render_pass,name) :
        try :
            prim=cls._primitives[name]
            render_pass.set_vertex_buffer(0, prim.buffer)
            if prim.draw_type=="line":
                render_pass.draw(prim.draw_size, 1, 0, 0)

        except KeyError:
            print("Primitive not found")
            return



if __name__ == "__main__":

    print("Primitives")
