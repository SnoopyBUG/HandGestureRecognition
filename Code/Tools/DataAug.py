from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
dirs = os.listdir("9")   ##the route of your origin images(need to edit) 
print(len(dirs))
for filename in dirs:
    img = load_img("9//{}".format(filename))  ##the route of your origin images(need to edit) 
    x = img_to_array(img)
    ##print(x.shape)
    x = x.reshape((1,) + x.shape)
    datagen.fit(x)
    prefix = filename.split('.')[0]
    print(prefix)
    counter = 0
    for batch in datagen.flow(x, batch_size=4 , save_to_dir='g9', save_prefix=prefix, save_format='png'):  
    ## parame:  save_to_dir  the route to save the new images(need to edit)
    
        counter += 1
        if counter > 10:
            break  # 否则生成器会退出循环

