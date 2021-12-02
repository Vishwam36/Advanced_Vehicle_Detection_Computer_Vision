from keras.preprocessing import image as image_utils
import PIL.ImageOps
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model

def giveIndex(img):
    if img.size[0] >= img.size[1]:
        read_image = np.asarray(img)
        edges = cv2.Canny(read_image, 150, 300)
        shape = np.shape(edges)
        left = np.sum(edges[0:shape[0] // 2, 0:shape[1] // 2])
        right = np.sum(edges[0:shape[0] // 2, shape[1] // 2:])

        # More edges = Building
        # Less edges = Sky
        if right > left:
            sky_side = 0
        else:
            sky_side = 1

        # Resizing image to a particular size
        base_height = 400
        wpercent = (base_height / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(wpercent)))
        img = img.resize((wsize, base_height), Image.ANTIALIAS)

        #Cropping sky area from the image
        if sky_side == 0:
            img = img.crop((0, 0, base_height, img.size[1])) 
        else: 
            img = img.crop((img.size[0]-base_height, 0, img.size[0], img.size[1]))

        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(r'redundant\op.jpg', open_cv_image) 

    else:
        base_width = 400
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))

        # Resizing the image
        img = img.resize((base_width, hsize), Image.ANTIALIAS)

        # Cropping the image
        img = img.crop((0, 0, img.size[0], 400))

        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(r'redundant\op.jpg', open_cv_image) 
    
    # predict
    train_data = []

    img = image_utils.load_img(r'redundant\op.jpg', target_size = (100,100))
    img = PIL.ImageOps.invert(img)
    img = image_utils.img_to_array(img)

    # Appending array to the list
    train_data.append(img)
    np.save(r'redundant\train_data' + ".npy", np.array(train_data))

    model = load_model(r'trainedModelE10.h5')
    test_data = np.load(r'redundant\train_data.npy')
    test_data = test_data / 255.0
    y = model.predict(test_data, verbose=0)
    print(y[0])
    print(np.argmax(y[0]))
    
    return np.argmax(y[0])

def detect(img):
    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #  converted image
    pil_image = Image.fromarray(color_coverted)
    return giveIndex(pil_image)

if __name__ == "__main__":
    img = cv2.imread(r'dataset\images\ft2.jpg')
    i = detect(img)
    
    if i==0:
        print('Rainy')
    elif i==1 or i==2:
        print('normal')
    elif i==3:
        print('snowy')
    else:
        print('Foggy')

    cv2.imshow('img', img)
    cv2.waitKey(0)