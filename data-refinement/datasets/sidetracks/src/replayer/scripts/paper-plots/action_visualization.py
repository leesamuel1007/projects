import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import mesh_utils
import os
import mayavi.mlab as mlab

def viz_pts_and_eef_o3d(
    pts_pcd,
    eef_pos_list,
    eef_quat_list,
    heatmap_labels=None,
    save_path=None,
    frame="world",
    draw_frame=False,
    cam_frame_x_front=False,
    highlight_top_k=None,
    pcd_rgb=None
):
    """
    Plot eef in o3d visualization, with point cloud, at positions and
    orientations specified in eef_pos_list and eef_quat_list
    pts_pcd, eef_pos_list, and eef_quat_list need to be in same frame
    """
    print('o3ding...')
    # Get line_set for drawing eef in o3d
    line_set_list = get_eef_line_set_for_o3d_viz(
        eef_pos_list, eef_quat_list, highlight_top_k=highlight_top_k,
    )
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for line_set in line_set_list:
        vis.add_geometry(line_set)
    # Draw ref frame
    if draw_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis.add_geometry(mesh_frame)
    # Move camera
    if frame == "camera":
        # If visualizing in camera frame, view pcd from scene view
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        fov = ctr.get_field_of_view()
        H = np.eye(4)
        if cam_frame_x_front:
            R = Rotation.from_euler("XYZ", [90, 0, 90], degrees=True).as_matrix()
            H[:3, :3] = R
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)
    else:
        # If world frame, place camera accordingly to face object front
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        fov = ctr.get_field_of_view()
        H = np.eye(4)
        H[2, -1] = 1
        R = Rotation.from_euler("XYZ", [90, 0, 90], degrees=True).as_matrix()
        H[:3, :3] = R
        param.extrinsic = H
        ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    vis.update_renderer()
    if save_path is None:
        vis.run()
    else:
        vis.capture_screen_image(
            save_path,
            do_render=True,
        )
    vis.destroy_window()


def get_eef_line_set_for_o3d_viz(eef_pos_list, eef_quat_list, highlight_top_k=None):
    # Get base gripper points
    g_opening = 0.07
    gripper = mesh_utils.create_gripper("panda", root_folder=os.path.abspath(os.path.dirname(__file__)))
    gripper_control_points = gripper.get_control_point_tensor(
        1, False, convex_hull=False
    ).squeeze()
    mid_point = 0.5 * (gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array(
        [
            np.zeros((3,)),
            mid_point,
            gripper_control_points[1],
            gripper_control_points[3],
            gripper_control_points[1],
            gripper_control_points[2],
            gripper_control_points[4],
        ]
    )
    gripper_control_points_base = grasp_line_plot.copy()
    gripper_control_points_base[2:, 0] = np.sign(grasp_line_plot[2:, 0]) * g_opening / 2
    # Need to rotate base points, our gripper frame is different
    # ContactGraspNet
    r = Rotation.from_euler("z", 90, degrees=True)
    gripper_control_points_base = r.apply(gripper_control_points_base)
    # Compute gripper viz pts based on eef_pos and eef_quat
    line_set_list = []
    for i in range(len(eef_pos_list)):
        eef_pos = eef_pos_list[i]
        eef_quat = eef_quat_list[i]
        gripper_control_points = gripper_control_points_base.copy()
        g = np.zeros((4, 4))
        rot = Rotation.from_quat(eef_quat).as_matrix()
        g[:3, :3] = rot
        g[:3, 3] = eef_pos.T
        g[3, 3] = 1
        z = gripper_control_points[-1, -1]
        gripper_control_points[:, -1] -= z
        gripper_control_points[[1], -1] -= 0.02
        pts = np.matmul(gripper_control_points, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        lines = [[0, 1], [2, 3], [1, 4], [1, 5], [5, 6]]
        if highlight_top_k is not None:
            if i < highlight_top_k:
                # Draw grasp in green
                colors = [[0,1,0] for i in range(len(lines))]
            else:
                colors = [[0,0,0] for i in range(len(lines))]
        else:
            colors = [[0,0,0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set_list.append(line_set)
    return line_set_list

def draw_grasps_ours(grasps, cam_pose, gripper_openings, color=(0,1.,0), colors=None, show_gripper_mesh=False, tube_radius=0.0008):
    """
    Draws wireframe grasps for robotiq schematic from given camera pose and with given gripper openings

    Arguments:
        grasps {np.ndarray} -- Nx4x4 grasp pose transformations
        cam_pose {np.ndarray} -- 4x4 camera pose transformation
        gripper_openings {np.ndarray} -- Nx1 gripper openings

    Keyword Arguments:
        color {tuple} -- color of all grasps (default: {(0,1.,0)})
        colors {np.ndarray} -- Nx3 color of each grasp (default: {None})
        tube_radius {float} -- Radius of the grasp wireframes (default: {0.0008})
        show_gripper_mesh {bool} -- Renders the gripper mesh for one of the grasp poses (default: {False})
    """

    gripper = mesh_utils.create_gripper('panda', root_folder=os.path.abspath(os.path.dirname(__file__)))
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    mid_point[2] -= 0.02
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], gripper_control_points[1], mid_point, gripper_control_points[2], gripper_control_points[4]])

    # if show_gripper_mesh and len(grasps) > 0:
    #     plot_mesh(gripper.hand, cam_pose, grasps[0])
        
    all_pts = []
    connections = []
    index = 0
    N = 8
    for i,(g,g_opening) in enumerate(zip(grasps, gripper_openings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        finger_idx = [2,3,4,6,7]
        gripper_control_points_closed[finger_idx,0] = np.sign(grasp_line_plot[finger_idx,0]) * g_opening/2
        
        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((8, 1))),axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:,:3]
        
        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N

        if colors is not None:
            # Draw grasps individually
            color = colors[i]

            mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=color, tube_radius=tube_radius, opacity=1.0)

    if colors is None:
        # speeds up plot3d because only one vtk object
        all_pts = np.vstack(all_pts)
        connections = np.vstack(connections)
        src = mlab.pipeline.scalar_scatter(all_pts[:,0], all_pts[:,1], all_pts[:,2])
        src.mlab_source.dataset.lines = connections
        src.update()
        lines =mlab.pipeline.tube(src, tube_radius=tube_radius, tube_sides=12)
        mlab.pipeline.surface(lines, color=color, opacity=1.0)


grasp_transformation = np.eye(4)
cam_pose = np.eye(4)
gripper_openings = np.array([0.07])


eef_pos_list = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
eef_quat_list = np.array([[0, 0, 0, 1], [0, 0, 0, 1]])

draw_grasps_ours(
    grasps=grasp_transformation.reshape(1, 4, 4),
    cam_pose=cam_pose,
    gripper_openings=gripper_openings,

)





