import cv2 as cv
import numpy as np

# Function to generate drops
def generate_random_lines(imshape,slant,drop_length):
    drops=[]
    numDrops = (imshape[0] * imshape[1]) // 100 ## If You want heavy rain, try reducing denominator

    for i in range(numDrops):                 
        x= np.random.randint(0,imshape[1]-slant)        
        y= np.random.randint(0,imshape[0]-drop_length)        
        drops.append((x,y))    
    return drops
    

# Function to add rain
def add_rain(image):        
    imshape = image.shape    
    slant_extreme=8   
    slant= np.random.randint(-slant_extreme,slant_extreme)
    
    drop_length=4
    drop_width=1
    drop_color=(200,200,200) ## a shade of gray

    rain_drops= generate_random_lines(imshape,slant,drop_length)        
    for rain_drop in rain_drops:
        cv.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)    
        # image= cv2.blur(image,(3,3)) ## rainy view are blurry        
        # brightness_coefficient = 0.7 ## rainy days are usually shady     
        # image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS    
        # image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)    
        # image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB    
    return image


if __name__ == "__main__":
    img = cv.imread(r"dataset\images\img4.jpg")
    cv.imshow('Original Image',img)
    
    # Adding rain
    rainy = add_rain(img)
    cv.imshow('Rainy Image',rainy)
    cv.waitKey(0)