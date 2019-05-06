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

    args.add_argument('--image_size',type=int,default=128)
    args.add_argument('--batch_size',type=int,default=16)
  
    args.add_argument('--epochs',type=int,default=100)
    args.add_argument('--steps',type=int,default=100)
    args.add_argument('--lr',type=float,default=0.01,help='learning rate')
    args.add_argument('--loss',type=str,default='mse')

    args.add_argument('--out_path',type=str,default=None)
    args = args.parse_args()
    return args

def main():
    args = get_args()
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    
    train_generator = LR_HR_generator(args.image_dir,args.batch_size,args.image_size,args.de_num)
    test_generator = LR_HR_generator(args.test_dir,args.batch_size,args.image_size,args.de_num)
    model = get_model(args.model,args.lr,args.loss) 
    
    callbacks = []
    callbacks.append(LearningRateScheduler(schedule=Schedule(args.epochs,args.lr)))
    callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_PSNR:.5f}.hdf5",
                                     monitor="val_PSNR",
                                     verbose=1,
                                     mode="max",
                                     save_best_only=True))

    hist = model.fit_generator(train_generator,epochs=args.epochs,steps_per_epoch=args.steps,
                    callbacks=callbacks,validation_data=test_generator)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)

if __name__ == "__main__":
    main()