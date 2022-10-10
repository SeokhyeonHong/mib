import os
import glfw
import glm
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np

class Camera():
    def __init__(self,
        eye=glm.vec3(-300, 100, -300),
        center=glm.vec3(0,0,0),
        up=glm.vec3(0,1,0),
        zoom_factor=1,
        projection_mode=True,
        z_near=0.01,
        z_far=15000.0,
        fov_y=np.pi / 6.0,
        x_right=1.2
    ):
        self.eye = eye
        self.center = center
        self.up = up
        self.zoom_factor = zoom_factor
        self.projection_mode = projection_mode
        self.z_near = z_near
        self.z_far = z_far
        self.fov_y = fov_y
        self.x_right = x_right

    def get_view_matrix(self):
        return glm.lookAt(self.eye, self.center, self.up)
    
    def get_projection_matrix(self, aspect):
        if self.projection_mode:
            return glm.perspective(self.zoom_factor * self.fov_y, aspect, self.z_near, self.z_far)
        else:
            return self.parallel(self.zoom_factor * self.x_right, aspect, self.z_near, self.z_far)
    
    def parallel(self, r, aspect, n, f):
        l = -r
        width = 2*r
        height = width/ aspect
        t = height / 2
        b = -t
        return glm.ortho(l, r, b, t, n, f)

class Light():
    def __init__(self,
        program,
        position=glm.vec4(10, 10, 10, 1),
        ambient=glm.vec3(.5),
        diffuse=glm.vec3(.5),
        specular=glm.vec3(.5),
        attenuation=glm.vec3(0.01, 0.001, 0.0)
    ):
        self.program = program
        self.position = position
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.attenuation = attenuation

        self.update()

    def update(self):
        glUniform4fv(glGetUniformLocation(self.program, "light.position"), 1, glm.value_ptr(self.position))
        glUniform3fv(glGetUniformLocation(self.program, "light.ambient"), 1, glm.value_ptr(self.ambient))
        glUniform3fv(glGetUniformLocation(self.program, "light.diffuse"), 1, glm.value_ptr(self.diffuse))
        glUniform3fv(glGetUniformLocation(self.program, "light.specular"), 1, glm.value_ptr(self.specular))
        glUniform3fv(glGetUniformLocation(self.program, "light.attenuation"), 1, glm.value_ptr(self.attenuation))
        
class Material():
    def __init__(self,
        program,
        ambient=glm.vec3(0.5),
        diffuse=glm.vec3(0.5),
        specular=glm.vec3(0.0),
        shininess=10.0
    ):
        self.program = program
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess

        self.update()
    
    def update(self):
        glUniform3fv(glGetUniformLocation(self.program, "material.ambient"), 1, glm.value_ptr(self.ambient))
        glUniform3fv(glGetUniformLocation(self.program, "material.diffuse"), 1, glm.value_ptr(self.diffuse))
        glUniform3fv(glGetUniformLocation(self.program, "material.specular"), 1, glm.value_ptr(self.specular))
        glUniform1f(glGetUniformLocation(self.program, "material.shininess"), self.shininess)


def get_shader(filename):
    dirname = os.path.dirname(os.path.abspath(__file__))
    return open(os.path.join(dirname, filename), 'r').read()

def init_glfw(width=1920, height=1080):
    if not glfw.init():
        return None
 
    window = glfw.create_window(width, height, "OpenGL Window", None, None)
    if not window:
        glfw.terminate()
        return None
 
    glfw.make_context_current(window)
    return window

def compile_shader(vertex_shader, fragment_shader, geometry_shader=None):
    shader_list = []
    shader_list.append(OpenGL.GL.shaders.compileShader(get_shader(vertex_shader), GL_VERTEX_SHADER))
    shader_list.append(OpenGL.GL.shaders.compileShader(get_shader(fragment_shader), GL_FRAGMENT_SHADER))
    if geometry_shader != None:
        shader_list.append(OpenGL.GL.shaders.compileShader(get_shader(geometry_shader), GL_GEOMETRY_SHADER))

    program = OpenGL.GL.shaders.compileProgram(*shader_list)
    
    glUseProgram(program)
    return program
    
def key_event(window, key, scancode, action, mods):
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

def bind_buffer(buffer, data, shader, name, size):
    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    location = glGetAttribLocation(shader, name)
    glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(location)

