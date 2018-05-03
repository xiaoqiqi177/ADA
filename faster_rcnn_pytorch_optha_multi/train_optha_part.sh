CUDA_VISIBLE_DEVICES=$1 KERAS_BACKEND=theano MKL_THREADING_LAYER=GNU python train_optha_part.py --datasetname='trainval' --ratio-name='721' --task-name=$2 --ymlname=$3 --resume=$4
#CUDA_VISIBLE_DEVICES=$1 python train_optha_part.py --datasetname='trainval' --ratio-name='721' --task-name=$2 --ymlname=$3
#--resume=$3
