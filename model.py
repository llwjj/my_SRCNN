import keras.backend as K
from keras.layers import Input,Conv2D,LeakyReLU,UpSampling2D,BatchNormalization,add
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

def SRResNet(input_size=(None,None,3)):
    class _Residul_Block:
        def __init__(self):
            self.conv1 = Conv2D(64,3,padding='same')
            self.in1 = BatchNormalization()
            self.relu = LeakyReLU(0.2)
            self.conv2 = Conv2D(64,3,padding='same')
            self.in2 = BatchNormalization()
        def __call__(self,x):
            outputs = self.relu(self.in1( self.conv1(x)))
            outputs = self.in2(self.conv2(outputs))
            outputs = add([x,outputs])
            return outputs
    def _Residul(x,num=1):
        outputs = _Residul_Block()(x)
        for _ in range(num-1):
            outputs = _Residul_Block()(outputs)
        return outputs

    inputs = Input(input_size)

    layer_mid = Conv2D(64,9,padding='same')(inputs)
    res1 = LeakyReLU(0.2)(layer_mid)
    
    residual = _Residul(res1,16)

    layer_mid = Conv2D(64,3,padding='same')(residual)
    layer_mid = LeakyReLU(0.2)(layer_mid)

    layer_mid = add([layer_mid,res1])

    layer_mid = Conv2D(256,3,padding='same')(layer_mid)
    layer_mid = UpSampling2D(2)(layer_mid)
    layer_mid = Conv2D(64,1,padding='same')(layer_mid)
    layer_mid = LeakyReLU(0.2)(layer_mid)
    layer_mid = Conv2D(256,3,padding='same')(layer_mid)
    layer_mid = UpSampling2D(2)(layer_mid)
    layer_mid = Conv2D(64,1,padding='same')(layer_mid)
    layer_mid = LeakyReLU(0.2)(layer_mid)

    outputs = Conv2D(3,9,padding='same')(layer_mid)

    model = Model(inputs,outputs)

    return model


def get_model(model='SRCNN',lr=0.01,loss='mse',*args,**kw):
    func = None
    if model == 'SRCNN':
        func = SRCNN
    elif model == 'SRResNet':
        func = SRResNet
    model = func(*args,**kw)
    opt = Adam(lr)
    model.compile(optimizer=opt, loss=mse, metrics=[PSNR])
    return model

if __name__ == "__main__":
    model = get_model('SRResNet',lr=0.01,loss='mse',input_size=(64,64,3))
    model.summary()