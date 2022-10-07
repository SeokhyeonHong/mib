import re, os, ntpath
import numpy as np
from . import np_utils as npu

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}

class Anim(object):
    def __init__(self, rotations, positions, offsets, parents, bones, order, fps):
        self.rotations = rotations
        self.positions = positions
        self.offsets = offsets
        self.parents = parents
        self.bones = bones
        self.order = order
        self.fps = fps

def read_bvh(filename, order=None):
    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            fps = int(1. / frametime)
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()
    
    return Anim(rotations, positions, offsets, parents, names, order, fps)

def read_phase(filename):
    if not os.path.exists(filename):
        raise Exception("File not found: %s" % filename)

    f = open(filename, "r")
    lines = f.readlines()
    data = []
    for i in range(1, len(lines)):
        data.append(float(lines[i]))
    f.close()

    return np.array(data)

def get_lafan1_set(bvh_path, actors, window=50, offset=20):
    """
    Extract the same test set as in the article, given the location of the BVH files.
    :param bvh_path: Path to the dataset BVH files
    :param list: actor prefixes to use in set
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps)
    :return: tuple:
        X: local positions
        Q: local quaternions
        parents: list of parent indices defining the bone hierarchy
        contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
        contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
    """
    npast = 10
    subjects = []
    seq_names = []
    X = []
    Q = []
    contacts_l = []
    contacts_r = []

    # Extract
    bvh_files = os.listdir(bvh_path)

    for file in bvh_files:
        if file.endswith('.bvh'):
            seq_name, subject = ntpath.basename(file[:-4]).split('_')

            if subject in actors:
                print('Processing file {}'.format(file))
                seq_path = os.path.join(bvh_path, file)
                anim = read_bvh(seq_path)

                # Sliding windows
                i = 0
                while i+window < anim.pos.shape[0]:
                    q, x = npu.quat_fk(anim.quats[i: i+window], anim.pos[i: i+window], anim.parents)
                    # Extract contacts
                    c_l, c_r = npu.extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.02)
                    X.append(anim.pos[i: i+window])
                    Q.append(anim.quats[i: i+window])
                    seq_names.append(seq_name)
                    subjects.append(subjects)
                    contacts_l.append(c_l)
                    contacts_r.append(c_r)

                    i += offset

    X = np.asarray(X)
    Q = np.asarray(Q)
    contacts_l = np.asarray(contacts_l)
    contacts_r = np.asarray(contacts_r)

    # Sequences around XZ = 0
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # Unify facing on last seed frame
    X, Q = npu.rotate_at_frame(X, Q, anim.parents, n_past=npast)

    return X, Q, anim.parents, contacts_l, contacts_r

def get_data_dict(path, phase=False, target_fps=30):
    seq_features = []

    bvh_files = os.listdir(path)
    for file in bvh_files:
        if file.endswith('.bvh'):
            print('Processing file {}'.format(file))
            seq_path = os.path.join(path, file)
            anim = read_bvh(seq_path)

            # process the data for fps
            if anim.fps % target_fps != 0:
                raise Exception("Target FPS must be a divisor of the BVH FPS")
            step = int(anim.fps / target_fps)
            
            rot = anim.rotations[::step]
            pos = anim.positions[::step]
            if phase:
                phase_path = os.path.join(path, file[:-4] + ".phase")
                phase_data = read_phase(phase_path)[::step]

            # solve FK
            quats = npu.euler_to_quat(np.radians(rot), order=anim.order)
            quats = npu.remove_quat_discontinuities(quats)
            global_q, global_p = npu.quat_fk(quats, pos, anim.parents)
            global_root_p = global_p[:, 0:1]
            global_root_q = global_q[:, 0:1]
            local_p = npu.quat_mul_vec(npu.quat_inv(global_root_q), global_p - global_root_p)

            # NOTE: LaFAN1 foot contact labels: [3, 4], [7, 8]
            # NOTE: PFNN foot contact labels: 
            contacts_l, contacts_r = npu.extract_feet_contacts(global_p, [3, 4], [7, 8], velfactor=0.02 * (anim.fps / target_fps))
            # contacts_l, contacts_r = npu.extract_feet_contacts(global_p, [4, 5], [9, 10], velfactor=0.01 * (anim.fps / target_fps))

            # features
            features = {
                "offset": np.tile(anim.offsets, (global_p.shape[0] - 1, 1, 1)),
                "parents": anim.parents,
                "global_pos": global_p[1:],
                "global_vel": (global_p[1:] - global_p[:-1]) * target_fps,
                "local_pos": local_p[1:],
                "local_vel": (local_p[1:] - local_p[:-1]) * target_fps,
                "local_euler": rot[1:],
                "local_quat": quats[1:],
                "contacts_l": contacts_l,
                "contacts_r": contacts_r,
            }
            if phase:
                features["phase"] = phase_data[1:]

            seq_features.append(features)

    return seq_features
