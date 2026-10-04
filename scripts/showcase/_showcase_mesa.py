"""VTK generic render window using an explicitly loaded OSMesa context.

Only for hosts without an off-screen OpenGL context (macOS): set
RFX_OSMESA_LIB to a libOSMesa shared library. Unset, install() does nothing
and PyVista uses its own off-screen path (EGL/OSMesa on Linux).
"""
import ctypes as c
import os
from vtkmodules.vtkRenderingOpenGL2 import vtkGenericOpenGLRenderWindow
class MesaWindow(vtkGenericOpenGLRenderWindow):
    def __init__(self):
        super().__init__()
        self._gl=c.CDLL(os.environ['RFX_OSMESA_LIB'])
        self._gl.OSMesaCreateContextAttribs.argtypes=[c.POINTER(c.c_int),c.c_void_p];self._gl.OSMesaCreateContextAttribs.restype=c.c_void_p
        self._gl.OSMesaMakeCurrent.argtypes=[c.c_void_p,c.c_void_p,c.c_uint,c.c_int,c.c_int]
        self._gl.OSMesaGetCurrentContext.restype=c.c_void_p
        self._gl.OSMesaDestroyContext.argtypes=[c.c_void_p]
        attr=(c.c_int*13)(0x22,0x1908,0x30,24,0x31,8,0x33,0x34,0x36,3,0x37,3,0)
        self._ctx=self._gl.OSMesaCreateContextAttribs(attr,None)
        if not self._ctx:raise RuntimeError('OSMesa context creation failed')
        self._buffer=None;self._buffer_size=None
        self.SetOpenGLSymbolLoader2(c.cast(self._gl.OSMesaGetProcAddress,c.c_void_p).value,self._gl._handle)
        self.SetOwnContext(False);self.SetReadyForRendering(True);self.SetSupportsOpenGL(True);self.SetOffScreenRendering(True)
        self.AddObserver('WindowMakeCurrentEvent',self._make_current)
        self.AddObserver('WindowIsCurrentEvent',lambda *_:self.SetIsCurrent(self._gl.OSMesaGetCurrentContext()==self._ctx))
        self.AddObserver('WindowSupportsOpenGLEvent',lambda *_:self.SetSupportsOpenGL(True))
        self._make_current()
    def _make_current(self,*_):
        w,h=self.GetSize();w=max(1,w);h=max(1,h)
        if self._buffer_size!=(w,h):self._buffer=c.create_string_buffer(w*h*4);self._buffer_size=(w,h)
        if not self._gl.OSMesaMakeCurrent(self._ctx,self._buffer,0x1401,w,h):raise RuntimeError('OSMesaMakeCurrent failed')
        self.SetIsCurrent(True)
    def release_cgl(self):
        if self._ctx:self._gl.OSMesaDestroyContext(self._ctx);self._ctx=None

def install():
    if not os.environ.get('RFX_OSMESA_LIB'):
        return
    import pyvista as pv
    import pyvista.plotting.plotter as pm
    import pyvista.plotting.tools as pt
    import pyvista.plotting.utilities.gl_checks as gc
    pm._prepare_offscreen_macos_render_window=lambda _:None
    pt._prepare_offscreen_macos_render_window=lambda _:None
    gc._prepare_offscreen_macos_render_window=lambda _:None
    pv._vtk.vtkRenderWindow=MesaWindow
