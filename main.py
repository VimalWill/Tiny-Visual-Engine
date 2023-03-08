from infere_src import load_model, infer
import cv2
from driver import MotorDriver
import threading

md = MotorDriver(37, 38, 3, 5, 7, 8)

if __name__ == "__main__":
    #load the model 
    infer_se = load_model("/home/vimal/Project/codes/models/yolop-640-640.onnx")
    cap = cv2.VideoCapture("/dev/video0")
    while(cap.isOpened()):
        ret, frame = cap.read()
        img_det, conf = infer(frame, infer_se)
        if not len(conf) == 0:
            if(conf[-1] >= 70):
                md.run(True)
            else:
                md.run(False)

        cv2.imshow("",img_det)
        cv2.waitKey(1)

    