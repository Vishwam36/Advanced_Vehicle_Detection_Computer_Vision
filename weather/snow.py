import cv2 as cv
import numpy as np

# Function generating random snow
def generate_random_balls(imshape,ball_size):
    balls=[]
    numBalls = (imshape[0] * imshape[1]) // 120

    for i in range(numBalls): ## If You want heavy fall, try increasing this                  
        x= np.random.randint(0,imshape[1])        
        y= np.random.randint(0,imshape[0])        
        balls.append((x,y))    
    return balls

# Function adding snow
def add_snow(image):        
    imshape = image.shape

    ball_size = 3
    ball_color = (255, 255, 255)

    balls = generate_random_balls(imshape, ball_size)

    for ball in balls:
        cv.circle(image, (ball[0], ball[1]), 1, ball_color, thickness=-1)
    
    return image

if __name__ == "__main__":
    img = cv.imread(r"dataset\images\img4.jpg")
    cv.imshow('Original Image',img)
    cv.waitKey(0)

    # Adding snow
    snowy = add_snow(img)
    cv.imshow('Snowy Image',snowy)

    cv.waitKey(0)