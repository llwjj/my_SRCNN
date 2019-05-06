import keras.backend as K
from keras.layers import Input,Conv2D
from keras.models import Model
from keras.optimizers import Adam

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125
        
def tf_log10(x):
    numerator = K.log(x)
    denominator = K.log(K.constant(10, dtype=numerator.dtype))
    return numerator / denominator
def mse(y_true, y_pred):
    return K.mean(K.square(y_true-y_pred))

def PSNR(y_true, y_pred):
    max_pixel = 255.0
    y_pred = K.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def SRCNN(input_size=(None,None,3)):
    inputs = Input(input_size)
    x = Conv2D(128,(9,9),padding='same')(inputs)
    x = Conv2D(64,(1,1),padding='same')(x)
    outputs = Conv2D(input_size[-1],(5,5),padding='same')(x)

    model = Model(inputs=inputs,outputs=outputs)
    return model

def get_model(model='SRCNN',lr=0.01,*args,**kw):
    func = None
    if model == 'SRCNN':
        func = SRCNN
    model = func(*args,**kw)
    opt = Adam(lr)
    model.compile(optimizer=opt, loss=mse, metrics=[PSNR])
    return model

if __name__ == "__main__":
    model = get_model()
    model.summary()