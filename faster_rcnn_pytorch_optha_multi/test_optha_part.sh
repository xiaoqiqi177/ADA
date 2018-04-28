CUDA_VISIBLE_DEVICES=$1 python test_optha_part.py --datasetname=$4 --ratio-name='721' --task-name=$2 --epochs=$3 --ymlname=$7 --trained-model=$5 --thresh=$6
python test_accuracy.py --datasetname=$4 --ratio-name='721' --task-name=$2 --epochs=$3 --thresh=$6
