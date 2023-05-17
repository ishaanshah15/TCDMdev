from jutils import mesh_utils, image_utils, plot_utils
# if you have depth instead of pc
depth_obj = mesh_utils.depth_to_pc(depth_obj, cameras=cameras)
# convert points to mesh, which are essentially tiny(eps) cubes for each points
# assume points are torch.Tensor in shape of (N, P, 3)
depth_obj = plot_utils.pc_to_cubic_meshes(pc=depth_obj, eps=5e-2)

# save as .obj
mesh_utils.dump_meshes(osp.join(logger.log_dir, 'meshes/%08d_obj' % it), depth_obj)

# merge two meshes with label
depth_hoi = mesh_utils.join_scene_w_labels([depth_hand, depth_obj], 3)
# merge two meshes without label
depth_hoi = mesh_utils.join_scene([depth_hand, depth_obj])
# Given one pytorch3d camera -> move the camera around world origin -> return a list of images[(N, 3, out_size, out_size)]
image_list = mesh_utils.render_geom_rot(depth_hoi, 'circle', cameras=cameras, view_centric=True, out_size=512)
# when you don't have camera, scale the scene such that all verts are visible -> rotate the scene -> return a list of images[(N, 3, H, W)]
image_list = mesh_utils.render_geom_rot(depth_hoi, scale_geom=True, out_size=512)
# save the gif to file system
image_utils.save_gif(image_list, 'hoi/hoi_depth_pointcloud')