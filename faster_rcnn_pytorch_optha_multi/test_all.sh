for thresh in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.
do
    CUDA_VISIBLE_DEVICES=$1 python test_optha_part.py --datasetname=$4 --ratio-name='721' --task-name=$2 --epochs=$3 --ymlname=$7 --trained-model=$5 --thresh=$thresh >> log_half_finetune
    python test_accuracy.py --datasetname=$4 --ratio-name='721' --task-name=$2 --epochs=$3 --thresh=$thresh >> log_half_finetune
done
