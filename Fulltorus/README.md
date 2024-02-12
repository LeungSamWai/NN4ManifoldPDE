# 3D case

## scripts
The scripts can be found in **scripts.py**
## run the code
To run the data with grid length 25, follow the command:
```
python train.py --gpu-id 0  --function fc_activate --activation raf --layers 4 --m 50 --optim adam --lr 0.01 --iters 1 --epoch 2000 --left -6 --right 6 --wd 0 --width 25 --penalty 0.001 --checkpoint N25_1 --trial 1
```
where `--m` represents the node size of NN and `--penalty` represents the l2 regularization for output.

For *width=283*, we use the SGD-alike training, i.e., we randomly sample some rows from *L* and conduct the residual training. To run the data with grid length 283, follow the command:
```
python train_batch.py --gpu-id 0  --function fc_activate --activation raf --layers 4 --m 564 --optim adam --lr 0.01 --iters 100 --epoch 80 --left -6 --right 6 --wd 0 --width 283 --penalty 0.01 --checkpoint N283_1 --trial 1
```
