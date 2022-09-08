import glfw
from OpenGL.GL import *
import glm
import numpy as np
import imageio

from .utils import *
from .gl_utils import *
from .primitives import *

def display(anim, parents, gt=None, targets=None, fps=30, save_gif=False, gif_name="animation.gif"):
    """
    :param anim: (#frames, #joints, 3) array of joint positions
    :param parents: (#joints) array of parent indices
    :param targets: (#targets, #joints, 3) array of target poses
    :param fps: frames per second
    :param save_gif: whether to save the animation as a gif
    :param gif_name: name of the gif file
    """
    # GL settings
    window = init_glfw()
    if window is None:
        return
    glfw.swap_interval(0)
    glfw.set_input_mode(window, glfw.STICKY_KEYS, GL_TRUE)
    glfw.set_key_callback(window, key_event)
 
    program = compile_shader()

    camera = Camera()
    light = Light(program)
    light.use()
    
    width, height = glfw.get_window_size(window)[0], glfw.get_window_size(window)[1]
    aspect = width / height
    P = camera.get_projection_matrix(aspect)

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    
    prev = 0.0
    frame = 0

    # Joints
    interval = 1.0 / fps

    images = []
    checkerboard = Checkerboard(program, 10000, 10000, 100,100)

    material = Material(program, ambient=glm.vec3(0.0, 1.0, 0.0))
    bones = [Cylinder(program, 2, 1, 16, material) for i in range(anim.shape[1])]
    if targets is not None:
        target_material = Material(program, ambient=glm.vec3(1.0, 0.0, 0.0))
        target_chars = []
        for i in range(targets.shape[0]):
            target_chars.append([Cylinder(program, 2, 1, 16, target_material) for i in range(targets.shape[1])])

    if gt is not None:
        gt_material = Material(program, ambient=glm.vec3(0.0, 0.0, 1.0))
        gt_bones = [Cylinder(program, 2, 1, 16, gt_material) for i in range(gt.shape[1])]

    while not glfw.window_should_close(window) and frame < anim.shape[0]:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.5, 0.5, 0.5, 1.0)
        glPolygonOffset(1, 1)

        curr = glfw.get_time()
        dt = curr - prev
        if dt > interval:
            prev = curr

            # render settings
            render_frame = anim[frame]

            for joint_idx, joint_pos in enumerate(render_frame):
                if joint_idx == 0:
                    eye = glm.vec3(0, 200, 500)
                    camera.center = glm.vec3(joint_pos[0], joint_pos[1], joint_pos[2])
                    camera.eye = camera.center + eye
                    continue
                parent_idx = parents[joint_idx]
                parent_pos = render_frame[parent_idx]
                
                center = (joint_pos + parent_pos) / 2
                center = glm.vec3(center[0], center[1], center[2])

                dist = np.linalg.norm(joint_pos - parent_pos)
                dir = (joint_pos - parent_pos) / (dist + 1e-5)
                dir = glm.vec3(dir[0], dir[1], dir[2])

                axis = glm.cross(glm.vec3(0, 1, 0), dir)
                angle = glm.acos(glm.dot(glm.vec3(0, 1, 0), dir))

                M = glm.mat4(1.0)
                M = glm.translate(M, center)
                M = glm.rotate(M, angle, axis)
                M = glm.scale(M, glm.vec3(1, dist, 1))

                V = camera.get_view_matrix()
                bones[joint_idx].draw(M, V, P)

            if gt is not None:
                render_frame = gt[frame]

                for joint_idx, joint_pos in enumerate(render_frame):
                    if joint_idx == 0:
                        continue
                    parent_idx = parents[joint_idx]
                    parent_pos = render_frame[parent_idx]
                    
                    center = (joint_pos + parent_pos) / 2
                    center = glm.vec3(center[0], center[1], center[2])

                    dist = np.linalg.norm(joint_pos - parent_pos)
                    dir = (joint_pos - parent_pos) / (dist + 1e-5)
                    dir = glm.vec3(dir[0], dir[1], dir[2])

                    axis = glm.cross(glm.vec3(0, 1, 0), dir)
                    angle = glm.acos(glm.dot(glm.vec3(0, 1, 0), dir))

                    M = glm.mat4(1.0)
                    M = glm.translate(M, center)
                    M = glm.rotate(M, angle, axis)
                    M = glm.scale(M, glm.vec3(1, dist, 1))

                    V = camera.get_view_matrix()
                    gt_bones[joint_idx].draw(M, V, P)


            if targets is not None:
                for char_idx, t in enumerate(targets):
                    for joint_idx, joint_pos in enumerate(t):
                        if joint_idx == 0:
                            continue
                        parent_idx = parents[joint_idx]
                        parent_pos = t[parent_idx]
                        
                        center = (joint_pos + parent_pos) / 2
                        center = glm.vec3(center[0], center[1], center[2])

                        dist = np.linalg.norm(joint_pos - parent_pos)
                        dir = (joint_pos - parent_pos) / (dist + 1e-5)
                        dir = glm.vec3(dir[0], dir[1], dir[2])

                        axis = glm.cross(glm.vec3(0, 1, 0), dir)
                        angle = glm.acos(glm.dot(glm.vec3(0, 1, 0), dir))

                        M = glm.mat4(1.0)
                        M = glm.translate(M, center)
                        M = glm.rotate(M, angle, axis)
                        M = glm.scale(M, glm.vec3(1, dist, 1))

                        V = camera.get_view_matrix()
                        target_chars[char_idx][joint_idx].draw(M, V, P)

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