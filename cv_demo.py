import cv2
import numpy as np
import matplotlib.pyplot as plt

# default size of figures
plt.rcParams['figure.figsize'] = [6, 6]

#imgstr = 'img/fabel.jpg'

def bgr_rgb(imgstr):
    # read input image 
    image = cv2.imread(imgstr)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # plot
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('BGR')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.title('RGB')
    plt.axis('off')
    plt.show()

    return image

def show_channels(image):
    # obtain individual channels
    channels = cv2.split(image)
    colors = ['blue', 'green', 'red']

    # show channels
    with plt.ioff():
        fig = plt.figure(figsize=(12, 4))
        for i, (color, channel) in enumerate(zip(colors,channels)):
            ax = fig.add_subplot(1, 3, i+1)
            ax.imshow(channel, cmap='gray')   # cmap because OpenCV uses BGR
            ax.set_xlabel(f'channel {i}, i.e. {colors[i]}', size=12)
    plt.show()

    return channels

def grayscale_comparison(image, channels):
    # manual grayscale
    grayscale_manual = 0.299 * channels[2] + 0.587 * channels[1] + 0.114 * channels[0]
    grayscale_manual = grayscale_manual.astype(np.uint8)
    
    # built-in OpenCV grayscale 
    grayscale_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # displaying images
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image (RGB)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(grayscale_manual, cmap='gray')
    plt.title('Manual Grayscale')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(grayscale_cv2, cmap='gray')
    plt.title('CV Grayscale')
    plt.axis('off')

    plt.show()

    # difference map
    difference = cv2.absdiff(grayscale_manual, grayscale_cv2)

    plt.figure(figsize=(4, 4))
    plt.imshow(difference, cmap='hot')
    plt.title('Difference Map')
    plt.axis('off')
    plt.show()

    return grayscale_cv2

def corner_detection(image, grayscale):
    # Shi-Tomasi Detector
    shi_image = image.copy()
    shi_grayscale = grayscale.copy()

    shi_corners = cv2.goodFeaturesToTrack(shi_grayscale, 100, 0.01, 10)
    shi_corners = np.intp(shi_corners)

    for i in shi_corners:
        x,y = i.ravel()
        cv2.circle(shi_image,(x,y),3,255,-1)
    
    # Harris Detector
    harris_image = image.copy()
    harris_grayscale = grayscale.copy()

    harris_grayscale_float = np.float32(harris_grayscale)
    harris_corners = cv2.cornerHarris(harris_grayscale_float, 2, 3, 0.04)

    harris_corners_dilated = cv2.dilate(harris_corners, None)
    harris_image[harris_corners_dilated > 0.01 * harris_corners_dilated.max()] = [0, 0, 255]

    # Compare
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(shi_image, cv2.COLOR_BGR2RGB))
    plt.title("Shi-Tomasi Corners")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(harris_image, cv2.COLOR_BGR2RGB))
    plt.title("Harris Corners")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def face_detection(img_path):
    image = cv2.imread(img_path)
    #rgb version
    rgb_image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # grayscale version
    grayscale2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # default face detector
    face_box_color1 = (255,0,0)

    face_cascade1 = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    faces1 = face_cascade1.detectMultiScale(grayscale2, 1.3, 5)
    img_default = image.copy()
    for (x,y,w,h) in faces1:
        cv2.rectangle(img_default,(x,y),(x+w,y+h),face_box_color1,2)

    # alternative face detector (DNN)
    dnn_image = image.copy()

    modelFile = "mymodels/dnn_model.caffemodel"
    configFile = "mymodels/deploy.prototxt"

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    (h, w) = dnn_image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(dnn_image, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))



    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(dnn_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # comparing visually
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_default, cv2.COLOR_BGR2RGB))
    plt.title('Default')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(dnn_image, cv2.COLOR_BGR2RGB))
    plt.title('Alternative (DNN)')
    plt.axis('off')

    plt.show()

def upperbody_detection(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # default upper body detector
    upperbody_box_color1 = (255,0,0)
    img1 = image.copy()

    upperbody_cascade1 = cv2.CascadeClassifier('mymodels/haarcascade_upperbody.xml')
    upperbodys1 = upperbody_cascade1.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in upperbodys1:
        cv2.rectangle(img1,(x,y),(x+w,y+h),upperbody_box_color1, 2)

    # alternative upper body detector (MCS)
    upperbody_box_color2 = (0,0,255)
    img2 = image.copy()
    
    upperbody_cascade2 = cv2.CascadeClassifier('mymodels/haarcascade_mcs_upperbody.xml')
    upperbodys2 = upperbody_cascade2.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in upperbodys2:
        cv2.rectangle(img2,(x,y),(x+w,y+h),upperbody_box_color2, 2)
        

    # comparing visually
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Default")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title("MCS")

    plt.show()

def main():
    img1 = 'img/fabel.jpg'
    img2 = 'img/Fest_gemischt.jpg'
    img3 = 'img/Personen_im_Park.jpg'

    image = bgr_rgb(img1)
    channels = show_channels(image)
    grayscale = grayscale_comparison(image, channels)

    corner_detection(image, grayscale)

    face_detection(img2)

    upperbody_detection(img3)


if __name__ == "__main__":
    main()