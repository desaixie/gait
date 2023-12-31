import pickle

import hydra
import matplotlib
import matplotlib.pyplot as plt
print("trajectory_visualize plt Backend: ", matplotlib.get_backend())
import numpy as np
import quaternion
from PIL import Image
from matplotlib import cm
from pathlib import Path
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import cv2

from habitat_test import HabitatSimGymWrapper, SpaceMapping

# options
# show_img = True
# save_interpolate_video = True
num_inter = 9
use_inter1d = False
use_slerp = True
use_linear_inter = False
use_quat = True
interp_vis = False


def inter1d(val):
    if len(val.shape) != 1 or val.shape[0] < 5:
        raise Exception(f'only interpolate 1d array with minimum length of 5, val shape: {val.shape}')
    x_key = np.linspace(0, 1, val.shape[0])
    tck = interpolate.splrep(x_key, val)
    x = np.linspace(0, 1, (val.shape[0] - 1) * (num_inter + 1) + 1)
    return interpolate.splev(x, tck)

""" [-1, 1] to [-pi, pi]. Positive because yaw is counterclockwise around y, right hand coord"""
def yaw2roty(val):
    return val * np.pi

def roty2yaw(val):
    return val / np.pi

""" [-1, 1] to [-pi/2, pi/2]. Positive because pitch is counterclockwise around x, right hand coord"""
def pitch2rotz(val):
    return 0.5 * val * np.pi

def rotz2pitch(val):
    return 2.0 * val / np.pi

def slerp_rot_mat(theta, phi):
    seq_len = theta.shape[0]
    slerp_seq_len = (seq_len - 1) * (num_inter + 1) + 1

    # interpolate slerp rotation (quaternion)
    euler = np.zeros((seq_len, 2))
    for i in range(seq_len):
        roty = yaw2roty(theta[i])
        rotz = pitch2rotz(phi[i])
        euler[i, :] = [roty, rotz]

    euler_rot = R.from_euler('YZ', euler, degrees=False)
    key_rot_times = np.linspace(0, 1, seq_len)
    slerp = Slerp(key_rot_times, euler_rot)
    frame_rot_times = np.linspace(0, 1, slerp_seq_len)
    if use_quat:
        euler_rot_ = slerp(frame_rot_times)
        euler_rot_quat = euler_rot_.as_quat()  # xyzw
        # convert quaternion from scipy to habitat-sim style: reverse dimensions, then negate last dim
        habitat_quat = euler_rot_quat[:, ::-1]  # wzyx
        habitat_quat[:, -1] = -habitat_quat[:, -1]  # xzy(-x)
        # habitat_quat = np.concatenate([habitat_quat[:, 1:], np.expand_dims(habitat_quat[:, 0], axis=-1)], axis=-1)  # zyxw
        habitat_quat = quaternion.as_quat_array(habitat_quat)  # do this, or we need the above line instead (letting habitat do the conversion, it would move the axis)
        return habitat_quat
    else:
        euler_rot_ = slerp(frame_rot_times).as_euler('XYZ', degrees=False)

        ret_theta, ret_phi = [], []
        for i in range(slerp_seq_len):
            ret_theta.append(roty2yaw(euler_rot_[i, 1]))
            ret_phi.append(rotz2pitch(euler_rot_[i, 0]))
            _roll = euler_rot_[i, 2]
        ret_theta = np.stack(ret_theta)
        ret_phi = np.stack(ret_phi)
        return ret_theta, ret_phi

def save_interpolated_video(env: HabitatSimGymWrapper, x, y, z, u, v, w, theta, phi, save_dir=""):
    # inter1d produces noisy results in theta
    if use_inter1d:
        x, y, z, u, v, w, theta, phi = [inter1d(_a) for _a in [x, y, z, u, v, w, theta, phi]]
        interpolated_poses = np.stack([x, y, z, theta, phi], axis=-1)
        quat = None
    elif use_slerp:
        # inter1d x,y,z. Convert theta, phi to rotation matrix, slerp on it, and convert back.
        _x, _y, _z, _u, _v, _w = [inter1d(_a) for _a in [x, y, z, u, v, w]]
        if interp_vis:
            x, y, z, u, v, w = _x, _y, _z, _u, _v, _w
        if use_quat:
            quat = slerp_rot_mat(theta, phi)
            interpolated_poses = np.stack([_x, _y, _z, _x, _x], axis=-1)
        else:
            theta, phi = slerp_rot_mat(theta, phi)
            interpolated_poses = np.stack([_x, _y, _z, theta, phi], axis=-1)
    elif use_linear_inter:  # take 310 actions of (action/10.), slows down each action by 10 times
        interp_actions = None
        interpolated_poses = None
        quat = None
    
    # video frames
    traj_frames = []
    slerp_after_quat = []
    # time_steps, _ = self.train_env.reset(to_poses=[poses[0]])
    for i in range(len(interpolated_poses)):
        if use_quat:
            _, _, _, aes_obs = env.reset(to_pose=interpolated_poses[i], to_quat=quat[i])
            if i % 10 == 0 or i == len(interpolated_poses) - 1:
                _state = env.agent.get_state()
                slerp_after_quat.append(quaternion.as_float_array(_state.rotation))
        else:
            _, _, _, aes_obs = env.reset(to_pose=interpolated_poses[i])
        traj_frames.append(aes_obs)
    traj_frames = np.stack(traj_frames)
    
    video_name = save_dir + 'trajectory.mp4'
    num_frames, height, width, layers = traj_frames.shape
    print(f"frames: {traj_frames.shape}")
    _fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, _fourcc, 30, (width, height))
    for frame in traj_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()
    print(f"Saved video {video_name}")


# based on https://stackoverflow.com/a/40463831/8667103
def WireframeSphere(centre=[0.,0.,0.], radius=1.,
                    n_meridians=20, n_circles_latitude=None):
    """
    Create the arrays of values to plot the wireframe of a sphere.

    Parameters
    ----------
    centre: array like
        A point, defined as an iterable of three numerical values.
    radius: number
        The radius of the sphere.
    n_meridians: int
        The number of meridians to display (circles that pass on both poles).
    n_circles_latitude: int
        The number of horizontal circles (akin to the Equator) to display.
        Notice this includes one for each pole, and defaults to 4 or half
        of the *n_meridians* if the latter is larger.

    Returns
    -------
    sphere_x, sphere_y, sphere_z: arrays
        The arrays with the coordinates of the points to make the wireframe.
        Their shape is (n_meridians, n_circles_latitude).

    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> sphere = ax.plot_wireframe(*WireframeSphere(), color="r", alpha=0.5)
    >>> fig.show()

    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> frame_xs, frame_ys, frame_zs = WireframeSphere()
    >>> sphere = ax.plot_wireframe(frame_xs, frame_ys, frame_zs, color="r", alpha=0.5)
    >>> fig.show()
    """
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z

# def plot3d_and_save_vid(env: HabitatSimGymWrapper, max_timestep, traj_file="../../../logs/cmaes/3 d s/trajectories/s0_eval2.npz"):
def plot3d_and_save_vid(env: HabitatSimGymWrapper, max_timestep, poses, actions, rewards, excluding_seq, diversity_ratio, smoothness_ratio, merge_plot=False, merge_plot_labels=None, save_fig=False, save_interpolate_video=True, show_img=True, teaser_background=False, fn=0):
    save_dir = Path(f"eval_{fn}")
    save_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["figure.figsize"] = [14.0, 7.0] if show_img else [7.0, 7.0]
    matplotlib.rcParams["figure.dpi"] = 200
    # matplotlib.rcParams["font.sans-serif"] = ["KaiTi"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig = plt.figure()
    fig.set_facecolor('none')
    fig.set_alpha(0.1)
    
    traj_len = max_timestep + 1
    if save_interpolate_video and interp_vis:
        traj_len = (traj_len - 1) * 10 + 1
    viridis32 = cm.get_cmap('viridis', traj_len)
    newcolors = viridis32(np.linspace(0, 1, traj_len))
    
    if merge_plot:
        xs, ys, zs, us, vs, ws, excluding_seqs, diversity_radiusess = [], [], [], [], [], [], [], []
        for i_plot in range(len(poses)):
            temp = np.split(poses[i_plot], 5, axis=-1)
            x, y, z, theta, phi = [_t.squeeze(1) for _t in temp]
            u = np.cos(phi) * np.sin(theta)
            v = np.sin(phi)
            w = np.cos(phi) * np.cos(theta)
            if excluding_seq[i_plot][0] is not None:  # None when diversity == False
                _exc_seq = excluding_seq[i_plot][0]  # for each step, there is the repeating (num_excluding_sequences, 5 or 6 or 7) excluding_seq
                if _exc_seq.shape[1] == 7:
                    diversity_radiuses = _exc_seq[:, -1]
                else:
                    diversity_radiuses = np.ones_like(rewards[i_plot])
                diversity_radiuses *= 0.8  # since the distance from yaw and pitch can't be visualized
                excluding_seqs.append(_exc_seq)
                diversity_radiusess.append(diversity_radiuses)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            us.append(u)
            vs.append(v)
            ws.append(w)

    else:
        temp = np.split(poses, 5, axis=-1)
        x, y, z, theta, phi = [_t.squeeze(1) for _t in temp]
        u = np.cos(phi) * np.sin(theta)
        v = np.sin(phi)
        w = np.cos(phi) * np.cos(theta)
        excluding_seq = excluding_seq[0]  # for each step, there is the repeating (num_excluding_sequences, 5 or 6 or 7) excluding_seq
        if excluding_seq is not None:
            if excluding_seq.shape[1] == 7:
                diversity_radiuses = excluding_seq[:, -1]
            else:
                diversity_radiuses = np.ones_like(rewards)
            diversity_radiuses *= 0.8  # since the distance from yaw and pitch can't be visualized

    if save_interpolate_video:
        save_interpolated_video(env, x, y, z, u, v, w, theta, phi, save_dir=f"{save_dir}/")
    
    if show_img:
        traj_obses = []
        _, _, _, aes_obs = env.reset(to_pose=poses[0])
        _state = env.agent.get_state()
        traj_obses.append(aes_obs)
        for i in range(max_timestep):
            _, _, _, aes_obs = env.step(actions[i + 1])
            _state = env.agent.get_state()
            traj_obses.append(aes_obs)
        
        img_dir = save_dir / "traj_imgs"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(traj_obses)):
            im = Image.fromarray(traj_obses[i])  # has to be uint8 dtype
            im.save(f"{img_dir}/eval_{i}.jpeg")
            plt.subplot2grid((7, 10), (int(i / 5), i % 5), colspan=1)
            plt.imshow(traj_obses[i], extent=[0., 1., 0., 1.])  # have lower left as origin, sides of length 1.
            # step t color legend
            ax = plt.gca()
            ax.spines['bottom'].set_color(newcolors[i, :])
            ax.spines['bottom'].set_linewidth(4.0)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # smoothness and diversity ratio bars
            ax.plot([0, 0], [0, smoothness_ratio[i]], '-', linewidth=6, color="orange", label="smoothness")  # smoothness ratio on the left
            ax.plot([1, 1], [0, diversity_ratio[i]], '-', linewidth=6, color="gold", label="diversity")  # diversity ratio on the right
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, loc='lower right')
    
    if teaser_background:
        teaser_bg_img = Image.open("../../../teaser_bg.jpeg")
        # im_width, im_height = teaser_bg_img.size
        # bbox = plt.gca().get_window_extent()
        # print(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        image_offset_horizon = 0
        image_offset_vertical = 0  # resize the plt window to fit
        # fig.figimage(teaser_bg_img, image_offset_horizon, image_offset_vertical, zorder=1, alpha=0.5)  # not transparent, at the bottom

    # draw positions (using scatter3D), path (using plot), and orientation (using quiver) in 3D plot
    if show_img:
        plt.subplot2grid((1, 2), (0, 1), colspan=1, projection='3d')
    else:
        plt.subplot2grid((1, 1), (0, 0), colspan=1, projection='3d')
    ax = plt.gca()
    c_list = [c for c in newcolors]
    if merge_plot:
        merge_plot_colors = ["red", "lime", "blue"]
        for i_plot in range(len(poses)):
            ax.scatter3D(xs[i_plot][0], zs[i_plot][0], ys[i_plot][0], color=merge_plot_colors[i_plot], s=45)  # bigger beginning dot
            ax.scatter3D(xs[i_plot], zs[i_plot], ys[i_plot], color=merge_plot_colors[i_plot])
            ax.plot(xs[i_plot], zs[i_plot], ys[i_plot], '-', label=merge_plot_labels[i_plot], color=merge_plot_colors[i_plot])
            ax.quiver(xs[i_plot], zs[i_plot], ys[i_plot], us[i_plot], ws[i_plot], vs[i_plot], color=merge_plot_colors[i_plot], length=0.1, normalize=True)

            # excluding spheres
            if excluding_seq[i_plot][0] is not None:
                for i in range(len(excluding_seqs[i_plot])):
                    if excluding_seqs[i_plot][i, 0] == -1.5:
                        continue
                    center = np.array([excluding_seqs[i_plot][i, 0], excluding_seqs[i_plot][i, 2], excluding_seqs[i_plot][i, 1]])  # x, z, y
                    ax.plot_wireframe(*WireframeSphere(center, diversity_radiusess[i_plot][i]), color=merge_plot_colors[i_plot], alpha=.1)
                    ax.scatter(*center, color="orange")
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, loc='lower right')
    elif teaser_background:
        color = "blue"
        ax.scatter3D(x[0], z[0], y[0], color="red", zorder=10)
        ax.scatter3D(x[7], z[7], y[7], color="yellow", zorder=10)
        ax.scatter3D(x[13], z[13], y[13], color="orange", zorder=10)
        ax.quiver(x[0], z[0], y[0], u[0], w[0], v[0], color="red", length=0.1, normalize=True, zorder=10)
        ax.quiver(x[7], z[7], y[7], u[7], w[7], v[7], color="yellow", length=0.1, normalize=True, zorder=10)
        ax.quiver(x[13], z[13], y[13], u[13], w[13], v[13], color="orange", length=0.1, normalize=True, zorder=10)
        ax.plot(x, z, y, '-', zorder=3)
        x = np.delete(x, [0, 7, 13])
        y = np.delete(y, [0, 7, 13])
        z = np.delete(z, [0, 7, 13])
        u = np.delete(u, [0, 7, 13])
        v = np.delete(v, [0, 7, 13])
        w = np.delete(w, [0, 7, 13])
        # x = x.pop(0)
        # x = x.pop(7)
        # x = x.pop(13)
        # y = y.pop(0)
        # y = y.pop(7)
        # y = y.pop(13)
        # z = z.pop(0)
        # z = z.pop(7)
        # z = z.pop(13)
        ax.scatter3D(x, z, y, color=color, zorder=3)
        ax.quiver(x, z, y, u, w, v, color=color, length=0.1, normalize=True, zorder=3)
    else:
        ax.scatter3D(x, z, y, color=c_list)
        ax.plot(x, z, y, '-')
        ax.quiver(x, z, y, u, w, v, color=c_list, length=0.1, normalize=True)
    
        # excluding spheres
        if excluding_seq is not None:
            for i in range(len(excluding_seq)):
                if excluding_seq[i, 0] == -1.5:
                    continue
                center = np.array([excluding_seq[i, 0], excluding_seq[i, 2], excluding_seq[i, 1]])  # x, z, y
                ax.plot_wireframe(*WireframeSphere(center, diversity_radiuses[i]), color="r", alpha=.1)
                ax.scatter(*center, color="orange")

    # labels
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    ax.set_zlim(-1., 1.)
    ax.set_xlabel("x (forward)", color="b")
    ax.invert_xaxis()
    ax.set_zlabel("y (up)", color="b")
    ax.set_ylabel("z (right)", color="b")
    # plt.title("title 1",color ="b")
    plt.grid()
    if teaser_background:
        plt.axis('off')
    
    # plt.suptitle("Overall Title")
    plt.tight_layout()
    plt.subplots_adjust(right=0.965, wspace=0.05, hspace=0.05)
    with open(f'{save_dir}/plot_{fn}.pickle', 'wb') as f:
        pickle.dump(ax, f)
    fig = plt.gcf()
    fig.savefig(f'{save_dir}/plot_{fn}')
    if not save_fig:
        plt.show(block=True)
        plt.interactive(False)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        
        self.cfg = cfg
        self.cfg.gpu_aes_obs = False
        self.cfg.aes_obs_width = 512
        self.cfg.aes_obs_height = 512

        self.num_scenes = self.cfg.num_scenes
        self.position_dim = 5 if self.cfg.use_rotation else 3
        self.step_size = self.cfg.step_size[:self.position_dim]
        
        self.cfg.scene_name = self.cfg.sceneList[self.cfg.scene_index]
        if show_img or save_interpolate_video:
            # self.train_env = AestheticTourDMCWrapper(self.cfg)
            self.space_mapper = SpaceMapping(cfg.sceneList[cfg.scene_index])
            self.train_env = HabitatSimGymWrapper(self.cfg, space_mapper=self.space_mapper)

    def plot(self):
        traj_file = "../../../logs/drqv2_habitat/25 3 rand exc pose/trajectories/1497300_s0_eval3.npz"
        traj_file = np.load(traj_file)
        poses = traj_file["pose"]
        actions = traj_file["action"]
        rewards = traj_file["reward"]
        excluding_seq = traj_file["excluding_seq"]
        diversity_ratio = traj_file["diversity_ratio"]
        smoothness_ratio = traj_file["smoothness_ratio"]
        plot3d_and_save_vid(self.train_env, self.cfg.max_timestep, poses, actions, rewards, excluding_seq, diversity_ratio, smoothness_ratio)



@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from trajectory_visualize import Workspace as W
    workspace = W(cfg)
    workspace.plot()


if __name__ == '__main__':
    main()
