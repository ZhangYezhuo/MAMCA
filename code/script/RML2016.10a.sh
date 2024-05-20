# for RML2016.10a TRAIN Mamba
CUDA_VISIBLE_DEVICES=2 python RML2016.10a.py -a MAMCA -e 150 --seed 114514 -l ./logs/baseline/Mamba/ -ph train\
 -lentrain 128 -lentest 128\
 --hdf5_file "/mnt/DATA/RadioML2016/RML2016.10a_dict.pkl" \
 --inner_train true --cru_name "RML2016.10a" --warm_iter 10 --warm_epoch 10\
 --lr 0.03  -b 400