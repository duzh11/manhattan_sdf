from lib.config import args, cfg
import os
import trimesh
import torch
import numpy as np
import random

def run_mesh_extract():
    from lib.datasets import make_data_loader
    from lib.networks import make_network
    from lib.utils.mesh_utils import extract_mesh, refuse, transform
    from lib.utils.net_utils import load_network
    import open3d as o3d

    network = make_network(cfg).cuda()
    load_network(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch
    )
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)

    mesh = extract_mesh(network.model.sdf_net)
    # mesh = trimesh.load('evaluation/neus/'+cfg.output_mesh)
    # mesh = refuse(mesh, data_loader)
    # mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)

    assert cfg.output_mesh != ''
    output_dir=os.path.join(cfg.record_dir,cfg.output_mesh)
    # o3d.io.write_triangle_mesh(output_dir, mesh)
    mesh.export(output_dir)

def print_result(result_dict):
    for k, v in result_dict.items():
        print(f'{k:7s}: {v:1.3f}')

def save_result(result_dict, file_dir):
    with open(file_dir,'w') as f_log:
        for k, v in result_dict.items():
            f_log.writelines(f'{k:7s}: {v:1.3f}\n')

def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    from lib.networks import make_network
    from lib.utils.mesh_utils import extract_mesh, refuse, transform
    from lib.utils.net_utils import load_network
    import open3d as o3d

    network = make_network(cfg).cuda()
    load_network(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch
    )
    network.eval()
    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)

    mesh = extract_mesh(network.model.sdf_net)
    # mesh = trimesh.load('evaluation/neus/'+cfg.output_mesh)

    # mesh = refuse(mesh, data_loader)
    mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)

    assert cfg.output_mesh != ''
    output_dir=os.path.join(cfg.record_dir,cfg.output_mesh)
    # o3d.io.write_triangle_mesh(output_dir, mesh)
    mesh.export(output_dir)

    mesh_gt = o3d.io.read_triangle_mesh(f'{cfg.test_dataset.data_root}/{cfg.test_dataset.scene}/gt.obj')
    evaluate_result = evaluator.evaluate(mesh, mesh_gt)
    print_result(evaluate_result)
    file_dir=os.path.join(cfg.record_dir,cfg.exp_name+'.txt')
    save_result(evaluate_result,file_dir)
    print('save result to '+file_dir)

if __name__ == '__main__':
    args.type='mesh_extract'
    globals()['run_' + args.type]()
