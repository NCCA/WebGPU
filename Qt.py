#!/usr/bin/env python3
import sys
import qtpy 
from qtpy.QtCore import QTimer
from qtpy.QtGui import QSurfaceFormat
from qtpy.QtGui import QWindow
from qtpy.QtWidgets import QApplication
import wgpu
from wgpu.gui.qt import WgpuCanvas

# Step 1: Create a custom rendering window using WgpuCanvas
class WebGPUWindow(QWindow):
    def __init__(self):
        super().__init__()
        self.setTitle("WebGPU in PyQt")
        self.setSurfaceType(QWindow.SurfaceType.OpenGLSurface)

        # Initialize the WebGPU canvas
        self.canvas = WgpuCanvas()
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync(required_limits=None)
        # self.device = wgpu.request_adapter_sync(canvas=self.canvas).request_device()
#         swap_chain_format = context.get_preferred_format(adapter)
#         context.configure(
#                     device=device, format=swap_chain_format, usage=wgpu.TextureUsage.RENDER_ATTACHMENT
# )

        
#         self.swap_chain = self.device.configure_swap_chain(
#             canvas=self.canvas,
#             format=wgpu.TextureFormat.bgra8unorm_srgb,
#             usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
#         )
        self.initialize_renderer()

        # Timer for continuous rendering
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.render_frame)
        self.timer.start(16)  # Approx 60 FPS

    def initialize_renderer(self):
        # Create pipeline, buffers, shaders, etc.
        self.render_pipeline = self.device.create_render_pipeline(
            wgpu.RenderPipelineDescriptor(
                label="Render Pipeline",
                layout=None,
                vertex=wgpu.VertexState(
                    module=self.device.create_shader_module(
                        code="""
                        @vertex
                        fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
                            var pos = array<vec2<f32>, 3>(
                                vec2<f32>(0.0, 0.5),
                                vec2<f32>(-0.5, -0.5),
                                vec2<f32>(0.5, -0.5)
                            );
                            return vec4<f32>(pos[vertex_index], 0.0, 1.0);
                        }
                        """
                    ),
                    entry_point="vs_main",
                ),
                fragment=wgpu.FragmentState(
                    module=self.device.create_shader_module(
                        code="""
                        @fragment
                        fn fs_main() -> @location(0) vec4<f32> {
                            return vec4<f32>(0.4, 0.7, 0.9, 1.0);
                        }
                        """
                    ),
                    entry_point="fs_main",
                    targets=[
                        wgpu.ColorTargetState(
                            format=wgpu.TextureFormat.bgra8unorm_srgb,
                            blend=None,
                            write_mask=wgpu.ColorWrites.ALL,
                        )
                    ],
                ),
                primitive=wgpu.PrimitiveState(
                    topology=wgpu.PrimitiveTopology.triangle_list
                ),
            )
        )

    def render_frame(self):
        # Begin rendering
        command_encoder = self.device.create_command_encoder()
        texture_view = self.swap_chain.get_current_texture_view()

        render_pass = command_encoder.begin_render_pass(
            wgpu.RenderPassDescriptor(
                color_attachments=[
                    wgpu.RenderPassColorAttachment(
                        view=texture_view,
                        resolve_target=None,
                        load_value=(0.1, 0.2, 0.3, 1.0),  # Clear color
                        store_op=wgpu.StoreOp.store,
                    )
                ]
            )
        )
        render_pass.set_pipeline(self.render_pipeline)
        render_pass.draw(3, 1, 0, 0)  # Draw a triangle
        render_pass.end()

        self.device.queue.submit([command_encoder.finish()])
        self.canvas.update()  # Ensure the canvas updates

# Step 2: Initialize the PyQt application
def main():
    app = QApplication(sys.argv)
    window = WebGPUWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()