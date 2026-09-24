import os
import sys
import contextlib
from Pom.core.opengl_classes import *
from Pom.core.opengl_classes import _log_str
from OpenGL.platform import PLATFORM as _GL_PLATFORM
import glfw
import warnings
from skimage import measure
from scipy.ndimage import label, find_objects, binary_dilation, gaussian_filter

PIXEL_SCALE = 1024

# TODO: CLI arguments for
# silhouatte alpha & threshold
# render style
# camera pitch and yaw

# PyOpenGL binds to GLX or EGL on first import; tools._choose_gl_platform decides which for the workers
HEADLESS_EGL = type(_GL_PLATFORM).__module__ == 'OpenGL.platform.egl'


class RenderContextError(RuntimeError):
    pass


@contextlib.contextmanager
def _quiet_stderr():
    # Mesa's driver loader writes straight to stderr, ignoring EGL_LOG_LEVEL, for every NVIDIA card
    # it finds and cannot drive. Errors still surface as exceptions.
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def _egl_error_str(e):
    # EGLError's own repr includes object addresses, which would stop identical failures from grouping
    if hasattr(e, 'err'):
        operation = getattr(e.baseOperation, '__name__', e.baseOperation)
        return f"{operation} failed with {e.err}"
    return str(e)


class _HeadlessEGLContext:
    """OpenGL context straight on a GPU through EGL, with no window or display server involved."""

    def __init__(self, device_index=0):
        from OpenGL import EGL
        self.EGL = EGL
        self.display = self._open_display(device_index)
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)

        config_attribs = (EGL.EGLint * 11)(EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
                                           EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
                                           EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8, EGL.EGL_BLUE_SIZE, 8,
                                           EGL.EGL_NONE)
        configs = (EGL.EGLConfig * 1)()
        n_configs = EGL.EGLint()
        EGL.eglChooseConfig(self.display, config_attribs, configs, 1, n_configs)
        if not n_configs.value:
            raise RuntimeError("eglChooseConfig found no OpenGL config with a pbuffer surface")
        config = configs[0]

        context_attribs = (EGL.EGLint * 7)(EGL.EGL_CONTEXT_MAJOR_VERSION, 4,
                                           EGL.EGL_CONTEXT_MINOR_VERSION, 3,
                                           EGL.EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                                           EGL.EGL_NONE)
        self.context = EGL.eglCreateContext(self.display, config, EGL.EGL_NO_CONTEXT, context_attribs)
        if not self.context:
            raise RuntimeError("eglCreateContext failed")

        # everything is drawn into FBOs, so the surface only has to exist
        surface_attribs = (EGL.EGLint * 5)(EGL.EGL_WIDTH, 16, EGL.EGL_HEIGHT, 16, EGL.EGL_NONE)
        self.surface = EGL.eglCreatePbufferSurface(self.display, config, surface_attribs)
        if not self.surface:
            raise RuntimeError("eglCreatePbufferSurface failed")
        if not EGL.eglMakeCurrent(self.display, self.surface, self.surface, self.context):
            raise RuntimeError("eglMakeCurrent failed")

    def _open_display(self, device_index):
        # Pick a GPU explicitly, so that workers spread over all GPUs. Not every listed device can be
        # opened (with glvnd, Mesa also lists the NVIDIA cards it cannot drive), so try them in turn,
        # with Mesa's CPU renderer and then the default display as the last resorts.
        EGL = self.EGL
        candidates = []
        try:
            from OpenGL.EGL.EXT.device_enumeration import eglQueryDevicesEXT
            from OpenGL.EGL.EXT.device_query import eglQueryDeviceStringEXT
            from OpenGL.EGL.EXT.platform_base import eglGetPlatformDisplayEXT
            from OpenGL.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT
            devices = (EGL.EGLDeviceEXT * 32)()
            n_devices = EGL.EGLint()
            eglQueryDevicesEXT(32, devices, n_devices)
            gpus, software = [], []
            for d in devices[:n_devices.value]:
                software_device = b'EGL_MESA_device_software' in (eglQueryDeviceStringEXT(d, EGL.EGL_EXTENSIONS) or b'')
                (software if software_device else gpus).append(d)
            k = device_index % len(gpus) if gpus else 0
            candidates = [lambda d=d: eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, d, None)
                          for d in gpus[k:] + gpus[:k] + software]
        except Exception:
            pass
        candidates.append(lambda: EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY))

        error = "no EGL display found"
        for get_display in candidates:
            try:
                display = get_display()
                if display and EGL.eglInitialize(display, EGL.EGLint(), EGL.EGLint()):
                    return display
            except Exception as e:
                error = _egl_error_str(e)
        raise RuntimeError(error)

    def delete(self):
        EGL = self.EGL
        EGL.eglMakeCurrent(self.display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
        EGL.eglTerminate(self.display)


def _resolve_missing_gl_functions():
    # PyOpenGL looks functions up as libGL exports; older libGL builds leave out the post-3.x ones
    import OpenGL.GL
    from OpenGL.platform import PLATFORM
    from OpenGL.platform.baseplatform import _NullFunctionPointer
    for f in list(vars(OpenGL.GL).values()):
        f = getattr(f, 'wrappedOperation', f)
        if not isinstance(f, _NullFunctionPointer) or f.resolved:
            continue
        pointer = glfw.get_proc_address(f.__name__)
        if not pointer:
            continue
        func = PLATFORM.functionTypeFor(f.DLL)(f.restype, *[PLATFORM.finalArgType(t) for t in f.argtypes])(pointer)
        func = PLATFORM.errorChecking(func, f.DLL, error_checker=f.error_checker)
        type(f).__call__ = staticmethod(func)
        type(f).resolved = True


class Renderer:
    def __init__(self, image_size=1024, device_index=0):
        self.egl_context = None
        if HEADLESS_EGL:
            try:
                with _quiet_stderr():
                    self.egl_context = _HeadlessEGLContext(device_index)
            except Exception as e:
                raise RenderContextError(
                    "Pom could not open an OpenGL context on the GPU through EGL, so nothing was rendered.\n"
                    f"EGL reported: {_egl_error_str(e)}")
        else:
            self._open_glfw_context()

        version = (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION))
        if version < (4, 3):
            description = f"OpenGL {version[0]}.{version[1]} ({_log_str(glGetString(GL_RENDERER))})"
            self.delete()
            raise RenderContextError(
                f"Pom needs OpenGL 4.3 to render, but this session only offers {description}, so nothing was rendered.\n"
                "Run 'pom render' on a machine with a GPU, or with a recent Mesa.")
        glEnable(GL_MULTISAMPLE)

        module_root = os.path.dirname(os.path.abspath(__file__))
        self.surface_model_shader = Shader(os.path.join(module_root, "..", "shaders", "se_surface_model_shader.glsl"))
        self.edge_shader = Shader(os.path.join(module_root, "..", "shaders", "se_depth_edge_detect.glsl"))
        self.depth_mask_shader = Shader(os.path.join(module_root, "..", "shaders", "se_depth_mask_shader.glsl"))
        self.ray_trace_shader = Shader(os.path.join(module_root, "..", "shaders", "raytrace_volume.glsl"))
        self.ndc_img_shader = Shader(os.path.join(module_root, "..", "shaders", "ndc_image.glsl"))

        self.style = 1

        self.width = self.height = image_size
        self.texture3d = glGenTextures(1, GL_TEXTURE_3D)
        glBindTexture(GL_TEXTURE_3D, self.texture3d)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        self.scene_fbo = FrameBuffer(width=self.width, height=self.height, texture_format="rgba32f")
        self.scene_fbo_b = FrameBuffer(width=self.width, height=self.height, texture_format="rgba32f")
        self.depth_fbo_a = FrameBuffer(width=self.width, height=self.height, texture_format="rgba32f")
        self.depth_fbo_b = FrameBuffer(width=self.width, height=self.height, texture_format="rgba32f")
        self.volume_fbo = FrameBuffer(width=self.width, height=self.height, texture_format="rgba32f")
        self.box_va = VertexArray(attribute_format="xyz")
        self.box_va_shape = (0, 0, 0)
        self.ndc_screen_va = VertexArray(attribute_format="xy")
        self.ndc_screen_va.update(VertexBuffer([-1, -1, 1, -1, 1, 1, -1, 1]), IndexBuffer([0, 1, 2, 0, 2, 3]))

        # TODO: make it possible to use style = 0, or 1, or 2, for different styles.
        self.RENDER_SILHOUETTES = True
        self.RENDER_SILHOUETTES_ALPHA = 0.7
        self.RENDER_SILHOUETTES_THRESHOLD = 0.01

        self.camera = Camera3D(self.width, self.height)
        self.camera.on_update()
        self.volume_fbo_active = False
        self.light = Light3D()
        self.ambient_strength = 1.2
        self.background_colour = (1.0, 1.0, 1.0, 0.0)

    def _open_glfw_context(self):
        os.environ.setdefault('EGL_LOG_LEVEL', 'fatal')
        if not glfw.init():
            raise RenderContextError(
                "Pom could not start GLFW, so nothing was rendered.\n"
                "This usually means there is no display ($DISPLAY is unset). Run as 'xvfb-run -a pom render'.")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        self.window = None
        with warnings.catch_warnings(record=True) as glfw_warnings:
            warnings.simplefilter("always")
            for samples, api in ((4, glfw.NATIVE_CONTEXT_API), (0, glfw.NATIVE_CONTEXT_API), (0, glfw.EGL_CONTEXT_API)):
                glfw.window_hint(glfw.SAMPLES, samples)
                glfw.window_hint(glfw.CONTEXT_CREATION_API, api)
                self.window = glfw.create_window(10, 10, "Offscreen Render", None, None)
                if self.window:
                    break
        if not self.window:
            glfw.terminate()
            reasons = "\n".join(f"  {w.message}" for w in glfw_warnings)
            raise RenderContextError(
                "Pom could not open an OpenGL 4.3 context, so nothing was rendered.\n"
                "This usually means the session has no GPU-backed display.\n"
                "Run 'pom render' on a machine with a local display, or as 'xvfb-run -a pom render'.\n"
                f"GLFW reported:\n{reasons}")
        glfw.make_context_current(self.window)
        # PyOpenGL looks the context up via GLX and finds nothing when GLFW made it with EGL
        import OpenGL.platform
        if not OpenGL.platform.GetCurrentContext():
            OpenGL.platform.GetCurrentContext = lambda: 1
        _resolve_missing_gl_functions()

    @staticmethod
    def poll_gl_states():
        # List of capabilities to check
        capabilities = [
            (GL_BLEND, 'GL_BLEND'),
            (GL_CULL_FACE, 'GL_CULL_FACE'),
            (GL_DEPTH_TEST, 'GL_DEPTH_TEST'),
            (GL_DITHER, 'GL_DITHER'),
            (GL_POLYGON_OFFSET_FILL, 'GL_POLYGON_OFFSET_FILL'),
            (GL_SAMPLE_ALPHA_TO_COVERAGE, 'GL_SAMPLE_ALPHA_TO_COVERAGE'),
            (GL_SAMPLE_COVERAGE, 'GL_SAMPLE_COVERAGE'),
            (GL_SCISSOR_TEST, 'GL_SCISSOR_TEST'),
            (GL_STENCIL_TEST, 'GL_STENCIL_TEST'),
            (GL_MULTISAMPLE, 'GL_MULTISAMPLE'),
            # Add more capabilities as needed
        ]
        # Poll and print the state of each capability
        for cap, cap_name in capabilities:
            print(cap, cap_name)
            state = glIsEnabled(cap)
            print(f'{cap_name}: {"Enabled" if state else "Disabled"}')

    def delete(self):
        if self.egl_context:
            self.egl_context.delete()
        else:
            glfw.terminate()

    def set_image_size(self, width, height):
        """Resize the render target FBOs to `width`x`height`. The renderables
        (SurfaceModel VAOs, VolumeModel textures, shaders) are untouched - they live in
        the same GL context. Camera projection is re-emitted to match the new aspect."""
        if (width, height) == (self.width, self.height):
            return
        for fbo in (self.scene_fbo, self.scene_fbo_b, self.depth_fbo_a, self.depth_fbo_b, self.volume_fbo):
            glDeleteTextures(1, [fbo.texture.renderer_id])
            glDeleteTextures(1, [fbo.depth_texture_renderer_id])
            glDeleteFramebuffers(1, [fbo.framebufferObject])
        self.width, self.height = width, height
        self.scene_fbo = FrameBuffer(width=width, height=height, texture_format="rgba32f")
        self.scene_fbo_b = FrameBuffer(width=width, height=height, texture_format="rgba32f")
        self.depth_fbo_a = FrameBuffer(width=width, height=height, texture_format="rgba32f")
        self.depth_fbo_b = FrameBuffer(width=width, height=height, texture_format="rgba32f")
        self.volume_fbo = FrameBuffer(width=width, height=height, texture_format="rgba32f")
        self.camera.set_projection_matrix(width, height)
        self.camera.on_update()

    def render(self, renderables_list):
        # render surface models first, volumes second.
        m_surfaces = [s for s in renderables_list if isinstance(s, SurfaceModel)]
        self.render_surface_models(m_surfaces)

        m_volumes = [v for v in renderables_list if isinstance(v, VolumeModel)]
        if not m_volumes:
            return

        self.render_depth_masks(m_volumes[0].data.shape)
        # depth to start sampling at is now in fbo b
        # depth to stop sampling at is now in fbo a

        # ray trace the volumes one by one.
        self.volume_fbo.bind()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        Z, Y, X = m_volumes[0].data.shape
        self.ray_trace_shader.bind()
        self.ray_trace_shader.uniformmat4("ipMat", self.camera.ipmat)
        self.ray_trace_shader.uniformmat4("ivMat", self.camera.ivmat)
        self.ray_trace_shader.uniformmat4("pMat", self.camera.pmat)
        self.ray_trace_shader.uniform1f("near", self.camera.clip_near)
        self.ray_trace_shader.uniform1f("far", self.camera.clip_far)
        self.ray_trace_shader.uniform2f("viewportSize", (self.width, self.height))
        self.ray_trace_shader.uniform1f("pixelSize", PIXEL_SCALE / m_volumes[0].data.shape[1])
        self.ray_trace_shader.uniform1i("Z", Z)
        self.ray_trace_shader.uniform1i("Y", Y)
        self.ray_trace_shader.uniform1i("X", X)
        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, self.depth_fbo_b.depth_texture_renderer_id)
        glActiveTexture(GL_TEXTURE0 + 2)
        glBindTexture(GL_TEXTURE_2D, self.depth_fbo_a.depth_texture_renderer_id)
        glBindImageTexture(3, self.volume_fbo.texture.renderer_id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)

        for v in m_volumes:
            glActiveTexture(GL_TEXTURE0)
            # Upload each volume's 3D texture once and cache it on the model. render() is called
            # once per frame, so re-uploading here (flatten + glTexImage3D) every call made spin
            # movies pay the full volume transfer cost per frame - by far the dominant cost.
            if getattr(v, 'gl_texture', None) is None:
                v.gl_texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_3D, v.gl_texture)
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, X, Y, Z, 0, GL_RED, GL_FLOAT, v.data.flatten())
            else:
                glBindTexture(GL_TEXTURE_3D, v.gl_texture)
            self.ray_trace_shader.uniform3f("C", v.colour)
            glDispatchCompute((self.width + 31) // 32, (self.height + 31) // 32, 1)
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT)
        self.volume_fbo_active = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

        # combine the isosurface image and the volume images
        self.scene_fbo.bind()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.volume_fbo.texture.renderer_id)
        self.ndc_img_shader.bind()
        self.ndc_screen_va.bind()
        glDrawElements(GL_TRIANGLES, self.ndc_screen_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.ndc_screen_va.unbind()
        self.ndc_img_shader.unbind()

    def render_surface_models(self, surface_models):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.scene_fbo.bind()
        self.surface_model_shader.bind()
        self.surface_model_shader.uniformmat4("vpMat", self.camera.matrix)
        self.surface_model_shader.uniform3f("viewDir", self.camera.get_view_direction())
        self.surface_model_shader.uniform3f("lightDir", self.light.vec)
        self.surface_model_shader.uniform1f("ambientStrength", self.ambient_strength)
        self.surface_model_shader.uniform1f("lightStrength", self.light.strength)
        self.surface_model_shader.uniform3f("lightColour", self.light.colour)
        self.surface_model_shader.uniform1i("style", self.style)
        glEnable(GL_DEPTH_TEST)
        alpha_sorted_surface_models = sorted(surface_models, key=lambda x: x.alpha, reverse=True)
        for s in alpha_sorted_surface_models:
            self.surface_model_shader.uniform4f("color", [*s.colour, s.alpha])
            for blob in s.blobs.values():
                if blob.complete and not blob.hide:
                    blob.va.bind()
                    glDrawElements(GL_TRIANGLES, blob.va.indexBuffer.getCount(), GL_UNSIGNED_INT, None)
                    blob.va.unbind()
        self.surface_model_shader.unbind()
        glDisable(GL_DEPTH_TEST)

        if len(alpha_sorted_surface_models) > 0:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.scene_fbo.framebufferObject)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.scene_fbo_b.framebufferObject)
            glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
            glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo.framebufferObject)
            self.edge_shader.bind()
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.scene_fbo_b.depth_texture_renderer_id)
            self.edge_shader.uniform1f("threshold", self.RENDER_SILHOUETTES_THRESHOLD)
            self.edge_shader.uniform1f("edge_alpha", self.RENDER_SILHOUETTES_ALPHA)
            self.edge_shader.uniform1f("zmin", self.camera.clip_near)
            self.edge_shader.uniform1f("zmax", self.camera.clip_far)
            self.ndc_screen_va.bind()
            glDrawElements(GL_TRIANGLES, self.ndc_screen_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.ndc_screen_va.unbind()
            self.edge_shader.unbind()

    def render_depth_masks(self, vol_size):
        if self.box_va_shape != vol_size:
            self.box_va_shape = vol_size
            render_pixel_size = PIXEL_SCALE / vol_size[1]
            w = vol_size[2] / 2 * render_pixel_size
            h = vol_size[1] / 2 * render_pixel_size
            d = vol_size[0] / 2 * render_pixel_size
            vertices = [-w, h, d,
                        w, h, d,
                        w, -h, d,
                        -w, -h, d,
                        -w, h, -d,
                        w, h, -d,
                        w, -h, -d,
                        -w, -h, -d]
            indices = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 0, 4, 7, 7, 3, 0, 5, 1, 2, 2, 6, 5, 4, 0, 1, 1, 5, 4, 3, 7, 6, 6, 2, 3]
            self.box_va.update(VertexBuffer(vertices), IndexBuffer(indices))


        # read scene depth
        self.scene_fbo_b.bind()
        data = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT)
        scene_depth = np.frombuffer(data, dtype=np.float32).reshape(self.height, self.width)

        # render the provisional stop depth
        self.depth_fbo_a.bind()
        glClearDepth(0.0)
        glClear(GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_GREATER)
        glDepthMask(GL_TRUE)

        self.depth_mask_shader.bind()
        self.depth_mask_shader.uniformmat4("vpMat", self.camera.matrix)
        self.box_va.bind()
        glDrawElements(GL_TRIANGLES, self.box_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.box_va.unbind()
        self.depth_mask_shader.unbind()

        # find the actual stop depth: minimum(scene_depth, stop_depth) and write to fbo a depth texture.
        self.depth_fbo_a.bind()
        data = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT)
        stop_depth = np.frombuffer(data, dtype=np.float32).reshape(self.height, self.width)
        stop_depth = np.minimum(scene_depth, stop_depth)
        glBindTexture(GL_TEXTURE_2D, self.depth_fbo_a.depth_texture_renderer_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.width, self.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, stop_depth.flatten())
        self.depth_fbo_a.unbind((0, 0, self.width, self.height))

        # render the start depth
        self.depth_fbo_b.bind()
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_CULL_FACE)
        glDepthFunc(GL_LESS)
        glClearDepth(1.0)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)


        self.depth_mask_shader.bind()
        self.depth_mask_shader.uniformmat4("vpMat", self.camera.matrix)
        self.box_va.bind()
        glDrawElements(GL_TRIANGLES, self.box_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.box_va.unbind()

        self.depth_mask_shader.unbind()
        glFinish()

        self.depth_fbo_b.unbind((0, 0, self.width, self.height))
        glDepthFunc(GL_LESS)
        glClearDepth(1.0)

    def new_image(self):
        self.scene_fbo.bind()
        glClearColor(*self.background_colour)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.scene_fbo_b.bind()
        glClearColor(*self.background_colour)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.volume_fbo.bind()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def get_image(self):
        self.scene_fbo.bind()
        glBindTexture(GL_TEXTURE_2D, self.scene_fbo.texture.renderer_id)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_FLOAT)
        image = np.frombuffer(data, dtype=np.float32).reshape((self.height, self.width, 4))
        image = np.flip(image, axis=0)[:, :, :3] * 255
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image


class VolumeModel:
    def __init__(self, data, feature_definition, pixel_size):
        self.data = gaussian_filter(data.astype(np.float32), sigma=feature_definition['sigma'] / max(1.0, pixel_size))

        # normalize to roughly 0..1 so the ray-trace shader's fixed thresholds apply (mirrors SurfaceModel)
        if data.dtype == np.int8:
            self.data /= 127.0
        elif data.dtype == np.uint16:
            self.data /= 255.0
        elif data.dtype == np.float32:
            self.data[self.data == 2] = 0  # easymode 3D data hack, see SurfaceModel

        self.data[0, :, :] = 0
        self.data[-1, :, :] = 0
        self.data[:, 0, :] = 0
        self.data[:, -1, :] = 0
        self.data[:, :, 0] = 0
        self.data[:, :, -1] = 0

        self.colour = SurfaceModel.hex_to_rgb(feature_definition['color'])
        self.pixel_size = pixel_size
        self.alpha = 1.0
        self.gl_texture = None  # 3D texture, uploaded lazily and cached on first render

    def delete(self):
        if self.gl_texture is not None:
            glDeleteTextures(1, [self.gl_texture])
            self.gl_texture = None


class SurfaceModel:
    def __init__(self, data, feature_definition, pixel_size):
        self.data = data
        self.data = gaussian_filter(data, sigma=feature_definition['sigma'] / max(1.0, pixel_size))

        self.data[0, :, :] = 0
        self.data[-1, :, :] = 0
        self.data[:, 0, :] = 0
        self.data[:, -1, :] = 0
        self.data[:, :, 0] = 0
        self.data[:, :, -1] = 0

        self.colour = SurfaceModel.hex_to_rgb(feature_definition['color'])

        self.level = feature_definition['threshold']

        if self.data.dtype == np.float32:
            self.data[self.data == 2] = 0 # little hack for easymode 3D data + pom visualization, to be removed eventually
        elif self.data.dtype == np.int8:
            self.level *= 127
        elif self.data.dtype == np.uint16:
            self.level *= 255

        self.dust = feature_definition['dust']
        self.alpha = 1.0

        self.render_pixel_size = PIXEL_SCALE / self.data.shape[1]
        if pixel_size == 1.0:  # Likely AreTomo tomo with no apix set; in Ais this is set to 10.0 A instead, do the same here.
            pixel_size = 10.0
        self.true_pixel_size = pixel_size
        if self.true_pixel_size < 1.0:
            self.dust = 0.0
        self.blobs = dict()

        self.generate_model()

    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def hide_dust(self):
        for i in self.blobs:
            self.blobs[i].hide = self.blobs[i].volume < self.dust

    def generate_model(self):
        data = self.data
        origin = 0.5 * np.array(self.data.shape) * self.render_pixel_size
        new_blobs = dict()

        labels, N = label(data >= self.level)
        voxel_counts = np.bincount(labels.ravel(), minlength=N + 1)
        for l, bbox in enumerate(find_objects(labels), start=1):
            if bbox is None:
                continue
            new_blobs[l] = SurfaceModelBlob(data, self.level, self.render_pixel_size, origin, self.true_pixel_size)
            try:
                new_blobs[l].compute_mesh(labels[bbox] == l, bbox, voxel_counts[l])
            except Exception:
                pass

        for i in self.blobs:
            self.blobs[i].delete()
        self.blobs = new_blobs
        self.hide_dust()

    def delete(self):
        for i in self.blobs:
            self.blobs[i].delete()


class SurfaceModelBlob:
    def __init__(self, data, level, render_pixel_size, origin, true_pixel_size=1.0):
        self.data = data
        self.level = level
        self.render_pixel_size = render_pixel_size
        self.true_pixel_size = true_pixel_size

        self.origin = origin
        self.volume = 0
        self.indices = list()
        self.vertices = list()
        self.normals = list()
        self.vao_data = list()
        self.va = VertexArray(attribute_format="xyznxnynz")
        self.va_requires_update = False
        self.complete = False
        self.hide = False

    def compute_mesh(self, blob_mask, bbox, n_voxels):
        self.volume = n_voxels * self.true_pixel_size**3

        rz, ry, rx = [(s.start, s.stop + 1) for s in bbox]
        box = np.zeros((1 + rz[1]-rz[0] + 1, 1 + ry[1]-ry[0] + 1, 1 + rx[1]-rx[0] + 1), dtype=np.float32)
        box[1:-1, 1:-1, 1:-1] = self.data[rz[0]:rz[1], ry[0]:ry[1], rx[0]:rx[1]]
        mask = np.zeros(box.shape, dtype=bool)
        mask[1:-2, 1:-2, 1:-2] = blob_mask
        mask = binary_dilation(mask, iterations=2)
        box *= mask
        vertices, faces, normals, _ = measure.marching_cubes(box, level=self.level)
        vertices += np.array([rz[0], ry[0], rx[0]])
        self.vertices = vertices[:, [2, 1, 0]]
        self.normals = normals[:, [2, 1, 0]]

        self.vertices *= self.render_pixel_size
        self.vertices -= np.array([self.origin[2], self.origin[1], self.origin[0]])
        self.vao_data = np.hstack((self.vertices, self.normals)).flatten()
        self.indices = faces.flatten()
        self.va_requires_update = True
        self.va.update(VertexBuffer(self.vao_data), IndexBuffer(self.indices, long=True))
        self.va_requires_update = False
        self.complete = True


    def delete(self):
        if self.va.initialized:
            if glIsBuffer(self.va.vertexBuffer.vertexBufferObject):
                glDeleteBuffers(1, [self.va.vertexBuffer.vertexBufferObject])
            if glIsBuffer(self.va.indexBuffer.indexBufferObject):
                glDeleteBuffers(1, [self.va.indexBuffer.indexBufferObject])
            if glIsVertexArray(self.va.vertexArrayObject):
                glDeleteVertexArrays(1, [self.va.vertexArrayObject])
            self.va.initialized = False


class Light3D:
    def __init__(self):
        self.colour = (1.0, 1.0, 1.0)
        self.vec = (0.0, 1.0, 0.0)
        self.yaw = 0.0
        self.pitch = 0.0
        self.strength = 0.8

    def compute_vec(self, dyaw=0, dpitch=0):
        # Calculate the camera forward vector based on pitch and yaw
        cos_pitch = np.cos(np.radians(self.pitch + dpitch))
        sin_pitch = np.sin(np.radians(self.pitch + dpitch))
        cos_yaw = np.cos(np.radians(self.yaw + dyaw))
        sin_yaw = np.sin(np.radians(self.yaw + dyaw))

        forward = np.array([-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw])
        self.vec = forward


class Camera3D:
    def __init__(self, width, height):
        self.view_matrix = np.eye(4)
        self.projection_matrix = np.eye(4)
        self.view_projection_matrix = np.eye(4)
        self.focus = np.zeros(3)
        self.pitch = 0.0
        self.yaw = 180.0
        self.distance = 1120.0
        self.clip_near = 1e2
        self.clip_far = 1e4
        self.projection_width = 1
        self.projection_height = 1
        self.set_projection_matrix(width, height)

    def set_projection_matrix(self, window_width, window_height):
        self.projection_width = window_width
        self.projection_height = window_height
        self.update_projection_matrix()

    def cursor_delta_to_world_delta(self, cursor_delta):
        self.yaw *= -1
        camera_right = np.cross([0, 1, 0], self.get_forward())
        camera_up = np.cross(camera_right, self.get_forward())
        self.yaw *= -1
        return cursor_delta[0] * camera_right + cursor_delta[1] * camera_up

    def get_forward(self):
        # Calculate the camera forward vector based on pitch and yaw
        cos_pitch = np.cos(np.radians(self.pitch))
        sin_pitch = np.sin(np.radians(self.pitch))
        cos_yaw = np.cos(np.radians(self.yaw))
        sin_yaw = np.sin(np.radians(self.yaw))

        forward = np.array([-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw])
        return forward

    @property
    def matrix(self):
        return self.view_projection_matrix

    @property
    def vpmat(self):
        return self.view_projection_matrix

    @property
    def ivpmat(self):
        return np.linalg.inv(self.view_projection_matrix)

    @property
    def pmat(self):
        return self.projection_matrix

    @property
    def vmat(self):
        return self.view_matrix

    @property
    def ipmat(self):
        return np.linalg.inv(self.projection_matrix)

    @property
    def ivmat(self):
        return np.linalg.inv(self.view_matrix)

    def on_update(self):
        self.update_projection_matrix()
        self.update_view_projection_matrix()

    def update_projection_matrix(self):
        aspect_ratio = self.projection_width / self.projection_height
        self.projection_matrix = Camera3D.create_perspective_matrix(60.0, aspect_ratio, self.clip_near, self.clip_far)
        self.update_view_projection_matrix()

    @staticmethod
    def create_perspective_matrix(fov, aspect_ratio, near, far):
        S = 1 / (np.tan(0.5 * fov / 180.0 * np.pi))
        f = far
        n = near

        projection_matrix = np.zeros((4, 4))
        projection_matrix[0, 0] = S / aspect_ratio
        projection_matrix[1, 1] = S
        projection_matrix[2, 2] = -f / (f - n)
        projection_matrix[3, 2] = -1
        projection_matrix[2, 3] = -f * n / (f - n)

        return projection_matrix

    def update_view_projection_matrix(self):
        eye_position = self.calculate_relative_position(self.focus, self.pitch, self.yaw, self.distance)
        self.view_matrix = self.create_look_at_matrix(eye_position, self.focus)
        self.view_projection_matrix = self.projection_matrix @ self.view_matrix

    def get_view_direction(self):
        eye_position = self.calculate_relative_position(self.focus, self.pitch, self.yaw, self.distance)
        focus_position = np.array(self.focus)
        view_dir = eye_position - focus_position
        view_dir /= np.sum(view_dir**2)**0.5
        return view_dir

    @staticmethod
    def calculate_relative_position(base_position, pitch, yaw, distance):
        cos_pitch = np.cos(np.radians(pitch))
        sin_pitch = np.sin(np.radians(pitch))
        cos_yaw = np.cos(np.radians(yaw))
        sin_yaw = np.sin(np.radians(yaw))

        forward = np.array([
            cos_pitch * sin_yaw,
            sin_pitch,
            -cos_pitch * cos_yaw
        ])
        forward = forward / np.linalg.norm(forward)

        relative_position = base_position + forward * distance

        return relative_position

    @staticmethod
    def create_look_at_matrix(eye, position):
        forward = Camera3D.normalize(position - eye)
        right = Camera3D.normalize(np.cross(forward, np.array([0, 1, 0])))
        up = np.cross(right, forward)

        look_at_matrix = np.eye(4)
        look_at_matrix[0, :3] = right
        look_at_matrix[1, :3] = up
        look_at_matrix[2, :3] = -forward
        look_at_matrix[:3, 3] = -np.dot(look_at_matrix[:3, :3], eye)
        return look_at_matrix

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

