import os
import subprocess
import time

lis_name_scenes = ['scene0378_00', 'scene0435_02', 'scene0050_00', 'scene0084_00']
lis_name_scenes += ['scene0616_00', 'scene0426_00']
lis_name_scenes += ['scene0025_00', 'scene0169_00']

### NeuRIS
method_name_lis = ['deeplab3sigmoid', 'deeplab40retrain', 'deeplab40detach', \
                   'Mask2Formera3sigmoid', 'Mask2Formera40retrain', 'Mask2Formera40detach']
class_lis=[3, 40, 40, 3, 40, 40]

exp_dir = './exps/result'

i=0
for method_name in method_name_lis:
    semclass=class_lis[i]
    i+=1
    for scene in lis_name_scenes:
        method_scene_name = f'{method_name}_{scene[-7:-3]}'
        
        path_conf = f'configs/scannet/{scene[-7:-3]}.yaml'
        
        with open(path_conf, 'r') as file:
            lines = file.readlines()
        lines[1] = f'exp_name: {method_scene_name}\n'
        lines[34] = f'        semantic_class: {semclass}\n'
        with open(path_conf, 'w') as file:
            file.writelines(lines)

        command = (f'python train_net.py --test --cfg_file {path_conf} gpus [0]')
        subprocess.run(command, shell=True, text=True)
        time.sleep(5)
