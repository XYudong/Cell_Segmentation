from data_utils import *
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras import backend as K

tf.keras.backend.set_image_data_format('channels_last')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# crop this smaller square portions from the original image(s), which are resized to this value (68 pixels by 68 pixels)
S_SIZE = 68
# size (pixels) for padded (symmetric padding) result of smaller square portions
L_SIZE = 128


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def decoder_block(num_filers, conv1, conv2):
    up = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv1), conv2], axis=-1)
    # to reduce checkerboard effect

    conv = layers.Conv2D(num_filers, (3, 3), padding='same',
                         kernel_initializer=VarianceScaling())(up)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)

    conv = layers.Conv2D(num_filers, (3, 3), padding='same',
                         kernel_initializer=VarianceScaling())(conv)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)
    return conv


def new_Unet(model_flag='vUnet'):
    MARGIN = 30
    model = VGG16(include_top=False, input_shape=(L_SIZE, L_SIZE, 3), weights='imagenet')
    conv5 = model.get_layer('block5_conv3').output
    conv4 = model.get_layer('block4_conv3').output
    conv3 = model.get_layer('block3_conv3').output
    conv2 = model.get_layer('block2_conv2').output
    conv1 = model.get_layer('block1_conv2').output

    conv6 = decoder_block(512, conv5, conv4)
    conv7 = decoder_block(256, conv6, conv3)
    conv8 = decoder_block(128, conv7, conv2)

    conv91 = decoder_block(64, conv8, conv1)
    conv92 = decoder_block(64, conv8, conv1)

    out1 = layers.Cropping2D(cropping=((MARGIN, MARGIN), (MARGIN, MARGIN)))(conv91)      # for mask
    out1 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='out1')(out1)

    if model_flag == 'vUnet':
        out2 = layers.Cropping2D(cropping=((MARGIN, MARGIN), (MARGIN, MARGIN)))(conv92)      # for edge
        out2 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='out2')(out2)

        model = tf.keras.Model(inputs=model.inputs, outputs=[out1, out2])
    else:
        model = tf.keras.Model(inputs=model.inputs, outputs=out1)

    return model


def train_Unet(dataset, lr, epochs):
    im_path = './DataSet_label/' + dataset + '/train'
    mask_path = './DataSet_label/' + dataset + '/train_mask'
    batch_size = 32

    data_getter = DataPreparer(im_path, mask_path, batch_size=batch_size)
    train_generator, val_generator = data_getter.main()
    num_train = data_getter.num_train
    num_val = data_getter.num_val

    # build model
    model_flag = 'vUnet'
    model = new_Unet(model_flag)
    for i in range(18):  # first 18 layers of pre-trained model
        model.layers[i].trainable = False

    model.summary()
    print('dataset: ', dataset)
    print('num of training data: ', num_train)
    print('num of validation data: ', num_val)

    # compile
    model.compile(optimizer=Adam(lr),
                  loss=['binary_crossentropy', 'binary_crossentropy'],      # mask, edge
                  metrics=[dice_coef],
                  loss_weights=[0.999, 0.001])
    # callbacks
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=3, verbose=1, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint('results/model/vUnet_' + dataset + '_01.hdf5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       verbose=1)
    tensorboard = TensorBoard(log_dir='./log/' + dataset + '/01',
                              write_graph=False,
                              write_grads=False,
                              histogram_freq=10,
                              batch_size=batch_size)
    # train
    hist = model.fit_generator(train_generator,
                               epochs=epochs,
                               steps_per_epoch=num_train // batch_size,
                               verbose=2,
                               callbacks=[reduce_lr, tensorboard, model_checkpoint],
                               validation_data=val_generator,
                               validation_steps=num_val // batch_size)
    K.clear_session()

    return


if __name__ == '__main__':
    dataset = 'FAK_N4'
    train_Unet(dataset, lr=2e-3, epochs=26)
