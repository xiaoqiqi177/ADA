CUDA_VISIBLE_DEVICES=$1 python3 train_optha_part.py --datasetname='trainval' --ratio-name='721' --task-name=$2 --resume=$3
