import numpy as np
import cv2
# from opencv.video import create_capture

# Use a classifier to detect a face in the video frame

def get_cam_frame(cam):
    ret, img = cam.read()
    # smaller frame size - things run a lot smoother than a full screen img
    img = cv2.resize(img, (1000, 800))
    return img

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def main():
    # Camera 0 is usually the built in webcam camera... also most people only have 1 webcam on their laptop
    cam = cv2.VideoCapture(0)

    # replace bear.png with any image in src directory
    picture = cv2.imread("bear.png")
    picture = cv2.resize(picture, (150,150))

    cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_default.xml")

    while True:
        img = get_cam_frame(cam)
        final = img.copy()

        # classifier wants things in black and white
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.equalizeHist(bw)

        rects = detect(bw, cascade)

        # Mostly useful for debugging
        for x1, y1, x2, y2 in rects:
            picture = cv2.resize(picture,(x2-x1,y2-y1))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)
            final[y1:y2, x1:x2] = picture

        cv2.imshow('face detect', final)

        # Esc key quits
        if 0xFF & cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
