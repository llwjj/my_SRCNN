from keras.utils import Sequence
from pathlib import Path
import random
import numpy as np

from image_process import imread,resize,LR_image

class LR_HR_generator(Sequence):
    def __init__(self,image_dir,batch_size=16,image_size=128,de_num=4,inplace=False):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.batch_size = batch_size
        self.image_size = image_size
        self.de_num = de_num
        self.inplace =inplace

        self.images_num = len(self.image_paths)
        if self.images_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))


    def __len__(self):
        return self.images_num//self.batch_size

    def __getitem__(self, idex):
        batch_size = self.batch_size
        image_size = self.image_size
        images_num = self.images_num
        inputs = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        outputs = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        
        path_index = idex * batch_size
        num = 0
        while True:
            if path_index > images_num:
                path_index = path_index % images_num
            path = str(self.image_paths[path_index])
            img = imread(path)

            h,w,_ = img.shape
            mini = min(h,w)
            if mini < image_size:
                img = resize(img,image_size//mini+1)
                h,w,_ = img.shape

            x = random.randint(0,w-image_size)
            y = random.randint(0,h-image_size)
            img = img[y:y+image_size,x:x+image_size]

            img_LR = LR_image(img,self.de_num,self.inplace)

            inputs[num] = img_LR
            outputs[num] = img
            
            num += 1
            path_index += 1

            if num >=batch_size:
                break
        return inputs,outputs

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.image_paths = LR_HR_generator.choices(self.image_paths)

    @staticmethod
    def choices(p,num=None):
        '''
        return a random num_size list from p,not repeat element
        '''
        if num is None:
            num = len(p)

        a = np.zeros(num,dtype=int)
        for i in range(num):
            a[i] = random.randint(0,len(p)-i-1)
        for i in range(num-1,-1,-1):
            v = a[i]
            for j in range(i+1,num):
                if v<=a[j]:
                    a[j]+=1
        p = np.array(p)[a]

        return list(p)

if __name__ == "__main__":
    generator = LR_HR_generator('../dataset/291',batch_size=4)
    generator.on_epoch_end()
    x,y = generator.__getitem__(0)
    from image_process import imshow
    imshow(x[0])
    imshow(y[0])