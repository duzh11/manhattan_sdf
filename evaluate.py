from lib.config import args, cfg
import os    
from lib.datasets import make_data_loader
from lib.evaluators import make_evaluator
from lib.networks import make_network
from lib.utils.mesh_utils import extract_mesh, refuse, transform
from lib.utils.net_utils import load_network
import open3d as o3d
import trimesh

def print_result(result_dict):
    for k, v in result_dict.items():
        print(f'{k:7s}: {v:1.3f}')

def save_result(result_dict, file_dir):
    with open(file_dir,'w') as f_log:
        for k, v in result_dict.items():
            f_log.writelines(f'{k:7s}: {v:1.3f}\n')

name_baseline='neus'
evaluator = make_evaluator(cfg)
scene_list=['0050_00','0084_00','0580_00','0616_00']
output_dir='evaluation/'+name_baseline+'/'

for scene in scene_list:
    if name_baseline=='neus':
        mesh = o3d.io.read_triangle_mesh(output_dir+'scene'+scene+'_clean_bbox_faces_mask.ply')
    else:
        mesh = o3d.io.read_triangle_mesh(output_dir+'scene'+scene+'.ply')
    mesh_gt = o3d.io.read_triangle_mesh('./evaluation/GT/scene'+scene+'_vh_clean_2.ply')
    evaluate_result = evaluator.evaluate(mesh, mesh_gt)
    print_result(evaluate_result)
    file_dir=os.path.join(output_dir,scene+'.txt')
    save_result(evaluate_result,file_dir)
    print('save result to '+file_dir)