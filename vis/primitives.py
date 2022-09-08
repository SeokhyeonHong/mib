from distutils.sysconfig import get_config_var
import glm
from OpenGL.GL import *
import numpy as np

from .gl_utils import bind_buffer

def get_color_by_pos(vertices):
    colors = []
    for i in range(len(vertices)//3):
        x, y, z = vertices[3*i], vertices[3*i+1], vertices[3*i+2]
        normalized = glm.normalize(glm.vec3(x, y, z))
        colors.append([normalized.x, normalized.y, normalized.z])
        # colors.append([1, 1, 1])
    return np.array(colors, dtype=np.float32).flatten()

def get_color(vertices, color=[0.5, 0.2, 0.7]):
    colors = []
    for i in range(len(vertices)//3):
        colors.append(color)
    colors = np.array(colors, dtype=np.float32).flatten()
    return colors

class Primitives():
    def __init__(self, program, positions, colors, normals):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(3)
        self.program = program
        self.positions = positions
        self.colors = colors
        self.normals = normals

        bind_buffer(self.vbo[0], self.positions, self.program, "vPosition", 3)
        bind_buffer(self.vbo[1], self.colors, self.program, "vColor", 3)
        bind_buffer(self.vbo[2], self.normals, self.program, "vNormal", 3)

    def get_vertices(self):
        pass

    def draw(self, M, V, P):
        pass

class Cube(Primitives):
    def __init__(self, program, width, height, depth):
        self.width, self.height, self.depth = width, height, depth
        self.positions, self.indices, self.normals = self.get_vertices()
        self.colors = get_color_by_pos(self.positions)

        super().__init__(program, self.positions, self.colors, self.normals)

        self.element_buff = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

    def get_vertices(self):
        width = self.width / 2
        height = self.height / 2
        depth = self.depth / 2
        
        points = [
            [-width, height, depth], [width, height, depth], [-width, height, -depth], [width, height, -depth],
            [-width, -height, depth], [width, -height, depth], [-width, -height, -depth], [width, -height, -depth]
        ]
        positions = [
            points[0], points[4], points[1], points[1], points[4], points[5], # z = 0.5
            points[2], points[3], points[7], points[2], points[7], points[6], # z = -0.5
            points[0], points[1], points[3], points[0], points[3], points[2], # y = 0.5
            points[4], points[6], points[5], points[5], points[6], points[7], # y = -0.5
            points[0], points[2], points[6], points[0], points[6], points[4], # x = 0.5
            points[1], points[5], points[3], points[5], points[7], points[3]  # x = -0.5
        ]
        positions = np.array(positions, dtype=np.float32).flatten()

        indices = [i for i in range(len(positions) // 3)]
        indices = np.array(indices, dtype=np.uint32).flatten()

        normals = [
            [0, 0, 1] * 6, # z = 0.5
            [0, 0, -1] * 6, # z = -0.5
            [0, 1, 0] * 6, # y = 0.5
            [0, -1, 0] * 6, # y = -0.5
            [1, 0, 0] * 6, # x = 0.5
            [-1, 0, 0] * 6 # x = -0.5
        ]
        normals = np.array(normals, dtype=np.float32).flatten()
        return positions, indices, normals

    def draw(self, M, V, P):
        glBindVertexArray(self.vao)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
        glUniformMatrix4fv(1, 1, GL_FALSE, glm.value_ptr(M))
        glUniformMatrix4fv(2, 1, GL_FALSE, glm.value_ptr(V))
        glUniformMatrix4fv(3, 1, GL_FALSE, glm.value_ptr(P))
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

        glDisable(GL_POLYGON_OFFSET_FILL)

class Sphere(Primitives):
    def __init__(self, program, radius, subh, suba):
        self.radius, self.subh, self.suba = radius, subh, suba
        self.positions, self.indices, self.normals = self.get_vertices()
        self.colors = get_color_by_pos(self.positions)
        super().__init__(program, self.positions, self.colors, self.normals)

        self.element_buff = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

    def get_vertices(self):
        y, rst = [], []
        cp, sp = [], []
        for i in range(self.subh+1):
            theta = i * np.pi / self.subh
            y.append(self.radius * np.cos(theta))
            rst.append(self.radius * np.sin(theta))
        for i in range(self.suba+1):
            phi = 2 * np.pi * i / self.suba
            cp.append(np.cos(phi))
            sp.append(np.sin(phi))

        positions, normals = [], []
        for i in range(self.subh):
            for j in range(self.suba):
                vx0, vy0, vz0 = sp[j] * rst[i], y[i], cp[j] * rst[i]
                vx1, vy1, vz1 = sp[j] * rst[i+1], y[i+1], cp[j] * rst[i+1]
                vx2, vy2, vz2 = sp[j+1] * rst[i], y[i], cp[j+1] * rst[i]
                vx3, vy3, vz3 = sp[j+1] * rst[i+1], y[i+1], cp[j+1] * rst[i+1]

                if i < self.subh - 1:
                    positions.append([vx0, vy0, vz0])
                    positions.append([vx1, vy1, vz1])
                    positions.append([vx3, vy3, vz3])

                    normals.append([vx0 / self.radius, vy0 / self.radius, vz0 / self.radius])
                    normals.append([vx1 / self.radius, vy1 / self.radius, vz1 / self.radius])
                    normals.append([vx3 / self.radius, vy3 / self.radius, vz3 / self.radius])
                if i > 0:
                    positions.append([vx3, vy3, vz3])
                    positions.append([vx2, vy2, vz2])
                    positions.append([vx0, vy0, vz0])

                    normals.append([vx3 / self.radius, vy3 / self.radius, vz3 / self.radius])
                    normals.append([vx2 / self.radius, vy2 / self.radius, vz2 / self.radius])
                    normals.append([vx0 / self.radius, vy0 / self.radius, vz0 / self.radius])

        positions = np.array(positions, dtype=np.float32).flatten()
        indices = np.array([i for i in range(len(positions) // 3)], dtype=np.uint32)
        normals = np.array(normals, dtype=np.float32).flatten()
        return positions, indices, normals

    def draw(self, M, V, P):
        glBindVertexArray(self.vao)
        
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
        glUniformMatrix4fv(1, 1, GL_FALSE, glm.value_ptr(M))
        glUniformMatrix4fv(2, 1, GL_FALSE, glm.value_ptr(V))
        glUniformMatrix4fv(3, 1, GL_FALSE, glm.value_ptr(P))
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

        glDisable(GL_POLYGON_OFFSET_FILL)

class Hemisphere(Primitives):
    def __init__(self, program, radius, subh, suba):
        self.radius, self.subh, self.suba = radius, subh, suba
        self.positions, self.indices, self.normals = self.get_vertices()
        self.colors = get_color_by_pos(self.positions)
        super().__init__(program, self.positions, self.colors, self.normals)

        self.element_buff = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
    
    def get_vertices(self):
        y, rst = [], []
        cp, sp = [], []
        for i in range(self.subh+1):
            theta = i * np.pi / self.subh / 2
            y.append(self.radius * np.cos(theta))
            rst.append(self.radius * np.sin(theta))
        for i in range(self.suba + 1):
            phi = 2 * np.pi * i / self.suba
            cp.append(np.cos(phi))
            sp.append(np.sin(phi))

        positions, normals = [], []
        for i in range(self.subh):
            for j in range(self.suba):
                vx0, vy0, vz0 = sp[j] * rst[i], y[i], cp[j] * rst[i]
                vx1, vy1, vz1 = sp[j] * rst[i+1], y[i+1], cp[j] * rst[i+1]
                vx2, vy2, vz2 = sp[j+1] * rst[i], y[i], cp[j+1] * rst[i]
                vx3, vy3, vz3 = sp[j+1] * rst[i+1], y[i+1], cp[j+1] * rst[i+1]

                positions.append([vx0, vy0, vz0])
                positions.append([vx1, vy1, vz1])
                positions.append([vx3, vy3, vz3])

                normals.append([vx0 / self.radius, vy0 / self.radius, vz0 / self.radius])
                normals.append([vx1 / self.radius, vy1 / self.radius, vz1 / self.radius])
                normals.append([vx3 / self.radius, vy3 / self.radius, vz3 / self.radius])

                if i > 0:
                    positions.append([vx3, vy3, vz3])
                    positions.append([vx2, vy2, vz2])
                    positions.append([vx0, vy0, vz0])
                    
                    normals.append([vx3 / self.radius, vy3 / self.radius, vz3 / self.radius])
                    normals.append([vx2 / self.radius, vy2 / self.radius, vz2 / self.radius])
                    normals.append([vx0 / self.radius, vy0 / self.radius, vz0 / self.radius])
        
        positions = np.array(positions, dtype=np.float32).flatten()
        indices = np.array([i for i in range(len(positions) // 3)], dtype=np.uint32)
        normals = np.array(normals, dtype=np.float32).flatten()
        return positions, indices, normals

    def draw(self, M, V, P):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
        glUniformMatrix4fv(1, 1, GL_FALSE, glm.value_ptr(M))
        glUniformMatrix4fv(2, 1, GL_FALSE, glm.value_ptr(V))
        glUniformMatrix4fv(3, 1, GL_FALSE, glm.value_ptr(P))
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

class Cylinder(Primitives):
    def __init__(self, program, radius, height, n, material=None):
        self.radius, self.height, self.n = radius, height, n
        self.positions, self.indices, self.normals = self.get_vertices()
        self.colors = get_color_by_pos(self.positions)
        self.material = material

        super().__init__(program, self.positions, self.colors, self.normals)

        # side - top - bottom
        self.element_buff = glGenBuffers(3)
        for i in range(3):
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff[i])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices[i].nbytes, self.indices[i], GL_STATIC_DRAW)
        
    def get_vertices(self):
        top_positions, side_positions, bottom_positions = [], [], []
        top_normals, side_normals, bottom_normals = [], [], []
        top_indices, side_indices, bottom_indices = [], [], []
        
        top_positions.append([0, self.height / 2, 0])
        top_normals.append([0, 1, 0])
        top_indices.append(0)

        bottom_positions.append([0, -self.height / 2, 0])
        bottom_normals.append([0, -1, 0])
        bottom_indices.append(3 * self.n + 4)

        for i in range(self.n+1):
            theta = 2.0 * np.pi * i / self.n
            x = self.radius * np.sin(theta)
            z = self.radius * np.cos(theta)

            top_positions.append([x, self.height/2, z])
            top_normals.append([0, 1, 0])
            top_indices.append(i+1)
            
            side_positions.append([x, self.height / 2, z])
            side_positions.append([x, -self.height / 2, z])
            side_normals.append([x / self.radius, 0, z / self.radius])
            side_normals.append([x / self.radius, 0, z / self.radius])
            side_indices.append(self.n + 2 + 2 * i)
            side_indices.append(self.n + 3 + 2 * i)

            bottom_positions.append([x, -self.height / 2, z])
            bottom_normals.append([0, -1, 0])
            bottom_indices.append(3 * self.n + 5 + i)

        positions = np.concatenate((top_positions, side_positions, bottom_positions), dtype=np.float32).flatten()        
        top_indices = np.array(top_indices, dtype=np.uint32)
        side_indices = np.array(side_indices, dtype=np.uint32)
        bottom_indices = np.array(bottom_indices, dtype=np.uint32)
        normals = np.concatenate((top_normals, side_normals, bottom_normals), dtype=np.float32).flatten()
        return positions, (top_indices, side_indices, bottom_indices), normals

    def draw(self, M, V, P):
        if self.material is not None:
            self.material.use()
        glBindVertexArray(self.vao)

        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        drawing_mode = [ GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN ]
        for i in range(3):
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff[i])
            glUniformMatrix4fv(1, 1, GL_FALSE, glm.value_ptr(M))
            glUniformMatrix4fv(2, 1, GL_FALSE, glm.value_ptr(V))
            glUniformMatrix4fv(3, 1, GL_FALSE, glm.value_ptr(P))
            glDrawElements(drawing_mode[i], self.indices[i].nbytes, GL_UNSIGNED_INT, None)
        
        glDisable(GL_POLYGON_OFFSET_FILL)

class Grid(Primitives):
    def __init__(self, program, width, height, m, n):
        self.width, self.height, self.m, self.n = width, height, m, n
        self.positions, self.indices, self.normals = self.get_vertices()
        self.colors = get_color_by_pos(self.positions)
        super().__init__(program, self.positions, self.colors, self.normals)

        self.element_buff = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

    def get_vertices(self):
        x0, x1 = -0.5 * self.width, 0.5 * self.width
        z0, z1 = -0.5 * self.height, 0.5 * self.height
        positions, normals = [], []
        for i in range(self.m+1):
            x = x0 + self.width * i / self.m
            positions.append([x, 0, z0])
            positions.append([x, 0, z1])
            normals.append([0, 1, 0])
            normals.append([0, 1, 0])
        for i in range(self.n+1):
            z = z0 + self.height * i / self.n
            positions.append([x0, 0, z])
            positions.append([x1, 0, z])
            normals.append([0, 1, 0])
            normals.append([0, 1, 0])
            
        positions = np.array(positions, dtype=np.float32).flatten()
        indices = np.array([i for i in range(len(positions) // 3)], dtype=np.uint32)
        normals = np.array(normals, dtype=np.float32).flatten()
        return positions, indices, normals

    def draw(self, V, P):
        glBindVertexArray(self.vao)
        glLineWidth(0.5)

        M = glm.mat4(1.0)
        glUniformMatrix4fv(1, 1, GL_FALSE, glm.value_ptr(M))
        glUniformMatrix4fv(2, 1, GL_FALSE, glm.value_ptr(V))
        glUniformMatrix4fv(3, 1, GL_FALSE, glm.value_ptr(P))
        glDrawElements(GL_LINES, len(self.indices), GL_UNSIGNED_INT, None)

class Checkerboard(Primitives):
    def __init__(self, program, width, height, m, n):
        self.width, self.height, self.m, self.n = width, height, m, n
        self.positions, self.indices, self.normals, self.colors = self.get_vertices()
        super().__init__(program, self.positions, self.colors, self.normals)

        self.element_buff = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buff)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

    def get_vertices(self):
        x0, x1 = -0.5 * self.width, 0.5 * self.width
        z0, z1 = -0.5 * self.height, 0.5 * self.height
        positions, normals, colors = [], [], []
        for i in range(self.m):
            for j in range(self.n):
                x_l = x0 + self.width * i / self.m
                x_r = x0 + self.width * (i+1) / self.m
                z_b = z0 + self.height * j / self.n
                z_t = z0 + self.height * (j+1) / self.n
                positions.append([x_l, 0, z_t])
                positions.append([x_l, 0, z_b])
                positions.append([x_r, 0, z_b])
                positions.append([x_r, 0, z_b])
                positions.append([x_r, 0, z_t])
                positions.append([x_l, 0, z_t])
                normals.append([0, 1, 0] * 6)
                if (i + j) % 2 == 0:
                    colors.append([0, 0, 0] * 6)
                else:
                    colors.append([1, 1, 1] * 6)

        positions = np.array(positions, dtype=np.float32).flatten()
        indices = np.array([i for i in range(len(positions) // 3)], dtype=np.uint32)
        normals = np.array(normals, dtype=np.float32).flatten()
        colors = np.array(colors, dtype=np.float32).flatten()
        return positions, indices, normals, colors

    def draw(self, V, P):
        glBindVertexArray(self.vao)
        glLineWidth(0.5)

        M = glm.mat4(1.0)
        glUniform1i(4, 1)
        glUniformMatrix4fv(1, 1, GL_FALSE, glm.value_ptr(M))
        glUniformMatrix4fv(2, 1, GL_FALSE, glm.value_ptr(V))
        glUniformMatrix4fv(3, 1, GL_FALSE, glm.value_ptr(P))
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glUniform1i(4, 0)