from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.01,
        height_shift_range=0.01,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        zca_whitening=True,
        fill_mode='nearest')
for dir in os.listdir("image/original"):
    for file in os.listdir("image/original/"+dir):
        if file.endswith(".jpg"):
            img = load_img('image/original/'+dir+'/'+file)  # this is a PIL image
            print('image/original/'+dir+'/'+file);
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            # the .flow() command below generates batches of randomly transformed images
            # and saves the results to the `preview/` directory
            i = 0
            for batch in datagen.flow(x, batch_size=1,
            save_to_dir="image/generate/"+dir+'/', save_prefix=file[:-4], save_format='jpg'):
                i += 1
                if i > 6:
                    break  # otherwise the generator would loop indefinitely
