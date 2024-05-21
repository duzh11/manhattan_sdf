sleep 0
echo "Train Start!!!"

python train_net.py --test --cfg_file configs/scannet/0025.yaml gpus [0]
python train_net.py --test --cfg_file configs/scannet/0050.yaml gpus [0]
python train_net.py --test --cfg_file configs/scannet/0084.yaml gpus [0]
python train_net.py --test --cfg_file configs/scannet/0169.yaml gpus [0]
# python train_net.py --test --cfg_file configs/scannet/0378.yaml gpus [0]
# python train_net.py --test --cfg_file configs/scannet/0426.yaml gpus [0]
# python train_net.py --test --cfg_file configs/scannet/0435.yaml gpus [0]
# python train_net.py --test --cfg_file configs/scannet/0616.yaml gpus [0]
