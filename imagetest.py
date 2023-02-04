import os
import cv2
import imgaug.augmenters as ia
import glob


#loading images from folder(multiple)

images = []
images_path=glob.glob("C:/image_aug/images/*.jpg")
#print(images_path)
#print(os.getcwd())

for i in images_path:
    img = cv2.imread(i)
    images.append(img)


print(images[133])
    
print(len(images))

#2. Image Augmentation

aug = ia.Sequential([

    #a.flip
    ia.Fliplr(0.5),
    ia.Flipud(0.5),

    #b.affine
    ia.Affine(translate_percent={"x":(-0.2,0.2), "y":(-0.2,0.2)},
    rotate=(-27,25),
    scale=(0.5,1.5)),

    #c.multiply
    ia.Multiply((0.8,1.5)),

    #d.Linear contrast
    ia.LinearContrast((0.6,1.4)),

    #e To Sometimes apply blur 

    ia.Sometimes(0.5,

  

    ia.GaussianBlur((0,3)))
    
])

aug_images = aug(images=images)

#3. Show Images
while True:
    aug_images = aug(images=images)
    for i in aug_images:
        cv2.imshow("Image",i)
        cv2.waitKey(0)


#print("new length\n")
#print(len(aug_images))

