import torch

################################
# PyTorch utils for quaternion #
################################

def normalize(x, dim=-1, eps=1e-8):
    res = x / (torch.linalg.norm(x, dim=dim, keepdim=True) + eps)
    return res

def quat_normalize(x, eps=1e-8):
    assert x.shape[-1] == 4
    
    res = normalize(x, eps=eps)
    return res

def quat_mul(x, y):
    assert x.shape[-1] == 4 and y.shape[-1] == 4

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = torch.cat([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], dim=-1)

    return res

def quat_mul_vec(q, x):
    assert q.shape[-1] == 4 and x.shape[-1] == 3

    t = 2.0 * torch.cross(q[..., 1:], x)
    res = x + q[..., 0][..., None] * t + torch.cross(q[..., 1:], t)
    return res

def quat_inv(q):
    assert q.shape[-1] == 4

    res = torch.asarray([1, -1, -1, -1], dtype=torch.float32, device=q.device) * q
    return res

def quat_fk(lq, gp_root, offset, parents):
    assert lq.shape[-1] == 4 and gp_root.shape[-1] == 3 and offset.shape[-1] == 3

    gp, gr = [gp_root], [lq[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], offset[..., i:i+1, :]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lq[..., i:i+1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)
    return res
    
def delta_rotate_at_frame(lq, frame):
    assert lq.shape[-1] == 4
    
    key_q = lq[:, frame-1:frame, 0:1, :]
    forward = torch.tensor([1, 0, 1], dtype=torch.float32, device=lq.device)[None, None, None, :]\
            * quat_mul_vec(key_q, torch.tensor([0, 1, 0], dtype=torch.float32, device=lq.device)[None, None, None, :])
    forward = normalize(forward)
    yrot = quat_normalize(quat_between(torch.tensor([1, 0, 0], dtype=torch.float32, device=lq.device)[None, None, None, :], forward))
    return quat_inv(yrot)

def quat_between(x, y):
    assert x.shape[-1] == 3 and y.shape[-1] == 3
    
    res = torch.cat([
        torch.sqrt(torch.sum(x * x, dim=-1) * torch.sum(y * y, dim=-1))[..., None] +
        torch.sum(x * y, dim=-1)[..., None],
        torch.cross(x, y)], dim=-1)
    return res

def euler_to_quat(e, order='zyx'):
    assert e.shape[-1] == 3
    
    axis = {
        'x': torch.asarray([1, 0, 0], dtype=torch.float32, device=e.device),
        'y': torch.asarray([0, 1, 0], dtype=torch.float32, device=e.device),
        'z': torch.asarray([0, 0, 1], dtype=torch.float32, device=e.device)}

    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return quat_mul(q0, quat_mul(q1, q2))

def angle_axis_to_quat(angle, axis):
    assert angle.shape[-1] == 1 and axis.shape[-1] == 3

    c = torch.cos(angle / 2.0)[..., None]
    s = torch.sin(angle / 2.0)[..., None]
    q = torch.cat([c, s * axis], dim=-1)
    return q

def quat_to_6d(q):
    assert q.shape[-1] == 4

    q0, q1, q2, q3 = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    
    r0 = torch.cat([2*(q0*q0 + q1*q1) - 1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)], dim=-1)
    r1 = torch.cat([2*(q1*q2 + q0*q3), 2*(q0*q0 + q2*q2) - 1, 2*(q2*q3 - q0*q1)], dim=-1)
    res = torch.cat([r0, r1], dim=-1)
    return res