#CUDA_VISIBLE_DEVICES=$1 python test_optha_part_all.py --datasetname=$4 --ratio-name='721' --task-name=$2 --method-name=$3 --ymlname=$6 --trained-model=$5 
python test_accuracy_all.py --datasetname=$4 --ratio-name='721' --task-name=$2 --method-name=$3
#>> log_$4_$3
#python plot_precision_recall_line.py --datasetname=$4 --ratio-name='721' --task-name=$2 --method-name=$3
