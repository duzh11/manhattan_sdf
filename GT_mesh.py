from lib.config import args, cfg
import os
import trimesh

from lib.datasets import make_data_loader
from lib.utils.mesh_utils import extract_mesh, refuse, transform
import open3d as o3d

data_loader = make_data_loader(cfg, is_train=False)
base_dir='./evaluation/neuris/'
scene_name='scene0616_00.ply'
output_mesh='scene0616_refuse.ply'

mesh=trimesh.load(os.path.join(base_dir,scene_name))

mesh = refuse(mesh, data_loader, refuse_GT=True,scale=cfg.test_dataset.scale, offset=cfg.test_dataset.offset)

output_dir=os.path.join(base_dir,output_mesh)
o3d.io.write_triangle_mesh(output_dir, mesh)
# mesh.export(output_dir)


