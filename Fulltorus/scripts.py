import os
import random
#
for gpu in ['2', '4', '5', '6', '7', '8']:
    random_num = random.randint(1, 1000000)
    os.system('python train.py --manualSeed '+str(random_num)+' --gpu-id '+gpu+'  --function fc_activate --activation raf --layers 4 --m 50 --optim adam --lr 0.01 --iters 1 --epoch 2000 --left -6 --right 6 --wd 0 --width 25 --penalty 0.001 --checkpoint N25_raf_repeat_'+gpu+' --trial '+gpu)
    random_num = random.randint(1, 1000000)
    os.system('python train.py --manualSeed '+str(random_num)+' --gpu-id '+gpu+'  --function fc_activate --activation raf --layers 4 --m 71 --optim adam --lr 0.01 --iters 1 --epoch 3000 --left -6 --right 6 --wd 0 --width 35 --penalty 0.001 --checkpoint N35_raf_repeat_'+gpu+' --trial '+gpu)
    random_num = random.randint(1, 1000000)
    os.system('python train.py --manualSeed '+str(random_num)+' --gpu-id '+gpu+'  --function fc_activate --activation raf --layers 4 --m 100 --optim adam --lr 0.01 --iters 1 --epoch 4000 --left -6 --right 6 --wd 0 --width 50 --penalty 0.001 --checkpoint N50_raf_repeat_'+gpu+' --trial '+gpu)
    random_num = random.randint(1, 1000000)
    os.system('python train.py --manualSeed '+str(random_num)+' --gpu-id '+gpu+'  --function fc_activate --activation raf --layers 4 --m 141 --optim adam --lr 0.01 --iters 1 --epoch 4000 --left -6 --right 6 --wd 0 --width 71 --penalty 0.001 --checkpoint N71_raf_repeat_'+gpu+' --trial '+gpu)
    random_num = random.randint(1, 1000000)
    os.system('python train.py --manualSeed '+str(random_num)+' --gpu-id '+gpu+'  --function fc_activate --activation raf --layers 4 --m 200 --optim adam --lr 0.01 --iters 1 --epoch 4000 --left -6 --right 6 --wd 0 --width 100 --penalty 0.001 --checkpoint N100_raf_repeat_'+gpu+' --trial '+gpu)
    random_num = random.randint(1, 1000000)
    os.system('python train.py --manualSeed '+str(random_num)+' --gpu-id '+gpu+'  --function fc_activate --activation raf --layers 4 --m 282 --optim adam --lr 0.01 --iters 1 --epoch 4000 --left -6 --right 6 --wd 0 --width 141 --penalty 0.002 --checkpoint N141_raf_repeat_'+gpu+' --trial '+gpu)
    random_num = random.randint(1, 1000000)
    os.system('python train.py --manualSeed '+str(random_num)+' --gpu-id '+gpu+'  --function fc_activate --activation raf --layers 4 --m 400 --optim adam --lr 0.01 --iters 1 --epoch 4000 --left -6 --right 6 --wd 0 --width 200 --penalty 0.005 --checkpoint N200_raf_repeat_'+gpu+' --trial '+gpu)
for gpu in ['2', '4', '5', '6', '7', '8']:
    random_num = random.randint(1, 1000000)
    os.system('python train_batch.py --manualSeed '+str(random_num)+' --gpu-id '+gpu+' --function fc_activate --activation raf --layers 4 --m 564 --optim adam --lr 0.01 --iters 100 --epoch 80 --left -6 --right 6 --wd 0 --width 283 --penalty 0.01 --checkpoint N283_raf_repeat_'+gpu+' --trial '+gpu)