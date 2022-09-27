import glfw
from OpenGL.GL import *
import glm
import numpy as np
import imageio

from .utils import *
from .gl_utils import *
from .primitives import *

# global variables

def display(anim, parents, gt=None, targets=None, fps=30, save_gif=False, gif_name="animation.gif", bone_radius=1, eye=glm.vec3(0, 200, 500)):
    window = init_glfw(width=1920, height=1080)
    if window is None:
        return

    program = compile_shader()

    camera = Camera()
    width, height = glfw.get_window_size(window)[0], glfw.get_window_size(window)[1]
    aspect = width / height
    P = camera.get_projection_matrix(aspect)
    
    light = Light(program)
    light.update()

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    
    prev_time = 0.0
    frame = 0

    interval = 1.0 / fps

    images = []
    checkerboard = Checkerboard(program, 10000, 10000, 200, 200)

    material = Material(program, ambient=glm.vec3(0.0, 1.0, 0.0))
    bones = [Cylinder(program, bone_radius, 1, 16, material) for i in range(anim.shape[1])]
    if targets is not None:
        target_material = Material(program, ambient=glm.vec3(1.0, 0.0, 0.0))
        target_bones = []
        for i in range(targets.shape[0]):
            target_bones.append([Cylinder(program, bone_radius, 1, 16, target_material) for i in range(targets.shape[1])])

    if gt is not None:
        gt_material = Material(program, ambient=glm.vec3(0.0, 0.0, 1.0))
        gt_bones = [Cylinder(program, bone_radius, 1, 16, gt_material) for i in range(gt.shape[1])]

    while not glfw.window_should_close(window) and frame < anim.shape[0]:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.5, 0.5, 0.5, 1.0)
        glPolygonOffset(1, 1)

        curr_time = glfw.get_time()
        dt = curr_time - prev_time
        if dt > interval:
            prev_time = curr_time

            # move camera to see the root of the character
            joint_pos = anim[frame][0]
            camera.center = glm.vec3(joint_pos[0], joint_pos[1], joint_pos[2])
            camera.eye = camera.center + eye
            V = camera.get_view_matrix()

            def render_pose(joints, bones, parents):
                for joint_idx, joint_pos in enumerate(joints):
                    if joint_idx == 0:
                        continue
                    parent_pos = joints[parents[joint_idx]]
                    
                    center = (joint_pos + parent_pos) / 2
                    center = glm.vec3(center[0], center[1], center[2])

                    dist = np.linalg.norm(joint_pos - parent_pos)
                    dir = (joint_pos - parent_pos) / (dist + 1e-8)
                    dir = glm.vec3(dir[0], dir[1], dir[2])

                    axis = glm.cross(glm.vec3(0, 1, 0), dir)
                    angle = glm.acos(glm.dot(glm.vec3(0, 1, 0), dir))

                    M = glm.mat4(1.0)
                    M = glm.translate(M, center)
                    M = glm.rotate(M, angle, axis)
                    M = glm.scale(M, glm.vec3(1, dist, 1))

                    bones[joint_idx].draw(M, V, P)
            
            render_pose(anim[frame], bones, parents)

            if gt is not None:
                render_pose(gt[frame], gt_bones, parents)

            if targets is not None:
                for char_idx, t in enumerate(targets):
                    render_pose(t, target_bones[char_idx], parents)

            checkerboard.draw(V, P)

            frame += 1
            
            glFlush()
        
            glfw.swap_buffers(window)

            if save_gif is True and frame > 1:
                glReadBuffer(GL_FRONT)
                data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
                image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                image = np.flip(image, axis=0)
                images.append(image)

        glfw.poll_events()

    if save_gif is True:
        imageio.mimsave(gif_name, images, fps=fps)
 
    glfw.terminate()
