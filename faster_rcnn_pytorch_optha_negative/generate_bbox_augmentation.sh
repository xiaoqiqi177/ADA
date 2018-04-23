DOUBLE='ten_healthy_noaug'
TIME=10

#first step, not skip
python generate_bbox_augmentation.py --task='ma_'$DOUBLE --iffake --ifimage --ifhealthy --ratio-name='721' --size-times=$TIME --aug-steps=16

#second step, skip
#python generate_bbox_augmentation.py --task='ma_'$DOUBLE'2' --ifskip --iffake --ifimage --ratio-name='721' --size-times=$TIME --aug-steps=16

#third step, skip and no fake
#python generate_bbox_augmentation.py --task='ma_'$DOUBLE'3' --ifskip --ifimage --ratio-name='721' --size-times=$TIME --aug-steps=16

#additional
#--ifdeug for visualization, --ifaug for training data
