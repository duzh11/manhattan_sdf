import time
import datetime
import torch
import os
import numpy as np
import open3d as o3d
import cv2
import trimesh
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import lib.train.trainers.utils_colour as utils_colour
from lib.config import cfg
from lib.utils.data_utils import to_cuda
from lib.utils.mesh_utils import extract_mesh, refuse, transform

class Trainer(object):
    def __init__(self, network):
        print('GPU ID: ', cfg.local_rank)
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
            network = DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                # find_unused_parameters=True
           )
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                #batch[k] = [b.cuda() for b in batch[k]]
                batch[k] = [b.to(self.device) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
            else:
                # batch[k] = batch[k].cuda()
                batch[k] = batch[k].to(self.device)
        return batch
    
    def get_loss_weights(self, epoch):
        loss_weights = dict()

        loss_weights['rgb'] = cfg.loss.rgb_weight

        loss_weights['depth'] = cfg.loss.depth_weight
        for decay_epoch in cfg.loss.depth_weight_decay_epochs:
            if epoch >= decay_epoch:
                loss_weights['depth'] *= cfg.loss.depth_weight_decay
        if epoch >= cfg.loss.depth_loss_clamp_epoch:
            loss_weights['depth_loss_clamp'] = cfg.loss.depth_loss_clamp
        
        loss_weights['joint_start'] = epoch >= cfg.loss.joint_start
        loss_weights['joint'] = cfg.loss.joint_weight

        loss_weights['ce_cls'] = torch.tensor([cfg.loss.non_plane_weight, 1.0, 1.0])
        loss_weights['ce_cls'] = to_cuda(loss_weights['ce_cls'])

        loss_weights['ce'] = cfg.loss.ce_weight
        for decay_epoch in cfg.loss.ce_weight_decay_epochs:
            if epoch >= decay_epoch:
                loss_weights['ce'] *= cfg.loss.ce_weight_decay
        
        loss_weights['eikonal'] = cfg.loss.eikonal_weight

        return loss_weights

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)                                                         
        self.network.train()
        end = time.time()
        #loss权重
        loss_weights = self.get_loss_weights(epoch)
        semantic_class = cfg.model.semantic.semantic_class
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = to_cuda(batch, self.device)
            batch['loss_weights'] = loss_weights
            batch['semantic_class'] = semantic_class
            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            
            torch.use_deterministic_algorithms(False)
            loss.backward()
            torch.use_deterministic_algorithms(True)
            
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, 
            epoch, 
            save_mesh=True, 
            evaluate_mesh=False, 
            data_loader=None, 
            evaluator=None, 
            recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        # mesh
        mesh, sem_mesh, surface_label = extract_mesh(self.network.net.model.sdf_net,
                                                    semantic_net = self.network.net.model.semantic_net,
                                                    seg_mesh = True)
        
        if save_mesh and not evaluate_mesh:
            mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)
            sem_mesh = transform(sem_mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)
            
            os.makedirs(f'{cfg.result_dir}/', exist_ok=True)
            mesh.export(f'{cfg.result_dir}/{epoch}.ply')
            sem_mesh.export(f'{cfg.result_dir}/{epoch}_sem.ply')
            np.savez(f'{cfg.result_dir}/semantic_surface.npz', surface_label)
        
        if evaluate_mesh:
            assert data_loader is not None
            assert evaluator is not None
            mesh = refuse(mesh, data_loader)
            mesh = transform(mesh, cfg.test_dataset.scale, cfg.test_dataset.offset)
            mesh_gt = o3d.io.read_triangle_mesh(f'{cfg.test_dataset.data_root}/{cfg.test_dataset.scene}/gt.obj')
            evaluate_result = evaluator.evaluate(mesh, mesh_gt)
            print(evaluate_result)
    
    def val_from_mesh(self, 
                    epoch,
                    from_mesh_dir=None):
        self.network.eval()
        torch.cuda.empty_cache()
        # mesh
        if from_mesh_dir is None:
            from_mesh_dir = glob(f'{cfg.result_dir}/*_TSDF.ply')[-1]
        from_mesh = trimesh.load(from_mesh_dir, process=False)
        from_mesh_vertices = from_mesh.vertices
        vertices = (from_mesh_vertices-cfg.test_dataset.offset)*cfg.test_dataset.scale

        ### copy from mesh_utils.py
        chunk=100000
        surface_label = []
        sdf_net = self.network.net.model.sdf_net
        semantic_net = self.network.net.model.semantic_net

        vertices_tensor = torch.FloatTensor(vertices.copy()).reshape([-1, 3]).cuda()
        for i in tqdm(range(0, vertices_tensor.shape[0], chunk), desc='Querying Surface Label of exsiting mesh'):
            sdf_i, nablas_i, geometry_feature_i = sdf_net.forward_with_nablas(vertices_tensor[i:i+chunk])
            semantics = semantic_net.forward(vertices_tensor[i:i+chunk], geometry_feature_i)
            surface_label.append(semantics.argmax(axis=1).cpu().numpy())
        surface_label = np.concatenate(surface_label, axis=0)

        semantic_class=cfg.model.semantic.semantic_class
        if semantic_class==3:
            colour_map_np = utils_colour.nyu3_colour_code
        elif semantic_class==40 or semantic_class==41:
            colour_map_np = utils_colour.nyu40_colour_code
            surface_label = surface_label + 1
        
        surface_labels_vis = colour_map_np[(surface_label)].astype(np.uint8)
        tomesh = trimesh.Trimesh(vertices=from_mesh_vertices, 
                                faces=from_mesh.faces, 
                                vertex_colors = surface_labels_vis,
                                process=False)
        
        tomesh.export(f'{cfg.result_dir}/{epoch}_TSDF_sem.ply')
        np.savez(f'{cfg.result_dir}/semantic_surface_TSDF.npz', surface_label)
    
    def render(self, epoch, data_loader):
        self.network.eval()
        torch.cuda.empty_cache()
        semantic_class=cfg.model.semantic.semantic_class
        if semantic_class==3:
            colour_map_np = utils_colour.nyu3_colour_code
        elif semantic_class==40 or semantic_class==41:
            colour_map_np = utils_colour.nyu40_colour_code

        imgs_render={}
        # render
        for iteration, batch in enumerate(data_loader):
            print(f'render image: {iteration}')
            for key in ['rgb', 'depth', 'semantic', 'surface_normals', 'volume_normals']:
                imgs_render[key] = []
            # render
            rays=batch['rays'][0]
            rays_batches=rays.split(1024)
            for ray_batch in rays_batches:
                ray_batch = to_cuda(ray_batch, self.device)
                render_output = self.network(ray_batch)
                for key in imgs_render:
                    imgs_render[key].append(render_output[key].detach().cpu().numpy())
                del render_output
            
            # save image
            H, W=480, 640
            for key in imgs_render:
                os.makedirs(f'{cfg.result_dir}/{key}', exist_ok=True)
                imgs_render[key] = np.concatenate(imgs_render[key], axis=0)

                if key=='rgb':  # rgb
                    imgs_render[key] = imgs_render[key].reshape([H, W, 3])
                    img_tmp=imgs_render[key][...,::-1]*255

                elif key=='depth':  # depth
                    imgs_render[key] = imgs_render[key].reshape([H, W])
                    render_depth=imgs_render[key]/cfg.test_dataset.scale*50
                    render_depth_map = cv2.convertScaleAbs(render_depth)
                    render_depth_map_jet = cv2.applyColorMap(render_depth_map, cv2.COLORMAP_JET)
                    img_tmp=render_depth_map_jet

                elif key=='semantic':  # rgb
                    imgs_render[key] = imgs_render[key].reshape([H, W, semantic_class])
                    semantic_fine=(imgs_render[key].argmax(axis=2))
                    if semantic_class!=3:
                        semantic_fine=semantic_fine+1
                    img_tmp=semantic_fine
                    vis_label=colour_map_np[(semantic_fine).astype(np.uint8)]
                    cv2.imwrite(f'{cfg.result_dir}/{key}/{iteration}_vis.png', vis_label[...,::-1])
                
                elif key=='volume_normals' or key=='surface_normals':
                    normals = imgs_render[key].reshape([H, W, 3])
                    norm_normals = np.linalg.norm(normals, axis=-1, ord=2,keepdims=True)
                    img_tmp = (((normals/norm_normals + 1) * 0.5).clip(0,1) * 255).astype(np.uint8)
                    img_tmp = img_tmp[...,::-1]

                cv2.imwrite(f'{cfg.result_dir}/{key}/{iteration}.png', img_tmp)
 
