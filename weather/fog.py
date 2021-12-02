from PIL import Image
import cv2
import numpy as np

def addfog(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img2= img.convert('L')
    img = img.convert("RGBA")
    kopi=img.copy()

    pixdata = img.load()
    ddata=img2.load()
    dup=kopi.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
                pixdata[x, y] = (pixdata[x,y][0],pixdata[x,y][1],pixdata[x,y][2],115-max(ddata[x,y]-100,0))
                dup[x,y]=(200,224,224,255)
    
    #img.save("oup71.png", "PNG")
    img = Image.alpha_composite(kopi, img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

if __name__ == "__main__":
    img = cv2.imread(r'dataset\images\img4.jpg')
    op = addfog(img)

    cv2.imshow("original", img)
    cv2.waitKey(0)

    cv2.imshow("output", op)
    cv2.waitKey(0)