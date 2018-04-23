DOUBLE='double_healthy_negative'
TIME=2

python generate.py --task='ma_'$DOUBLE --ifimage --ifhealthy --ratio-name='721' --size-times=$TIME --aug-steps=16

