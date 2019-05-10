import argparse
import numpy as np
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from model import get_model,Schedule
from generator import LR_HR_generator

def get_args():
    args = argparse.ArgumentParser(description="Super-Resolution_args",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument('--image_dir',type=str,required=True,help='train image dir')
    args.add_argument('--test_dir',type=str,required=True,help='test image dir')

    args.add_argument('--de_num',type=int,default=4,help='low image')
    args.add_argument('--model',type=str,default='SRCNN')

    args.add_argument('--image_size',type=int,default=80)
    args.add_argument('--batch_size',type=int,default=16)
  
    args.add_argument('--epochs',type=int,default=100)
    args.add_argument('--steps',type=int,default=100)
    args.add_argument('--lr',type=float,default=0.01,help='learning rate')


    args.add_argument('--outpath',type=str,default='result')
    args = args.parse_args()
    return args

def main():
    args = get_args()
    output_path = Path(__file__).resolve().parent.joinpath(args.outpath)
    
    inplace = args.model in ['SRResNet'] 
    train_generator = LR_HR_generator(args.image_dir,args.batch_size,args.image_size,args.de_num,inplace)
    test_generator = LR_HR_generator(args.test_dir,args.batch_size,args.image_size,args.de_num,inplace)
    model = get_model(args.model,args.lr) 
    
    callbacks = []
    callbacks.append(LearningRateScheduler(schedule=Schedule(args.epochs,args.lr)))
    callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_PSNR:.5f}.hdf5",
                                     monitor="val_PSNR",
                                     verbose=1,
                                     mode="max",
                                     save_best_only=True))

    hist = model.fit_generator(generator=train_generator,epochs=args.epochs,steps_per_epoch=args.steps,validation_data=test_generator,verbose=1,
                    callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)

if __name__ == "__main__":
    main()