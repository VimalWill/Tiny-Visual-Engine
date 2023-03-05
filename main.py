from infere_src import load_model, infer
import cv2
import threading


if __name__ == "__main__":
    #load the model 
    infer_se = load_model("/home/vimal/Project/models/yolop-640-640.onnx")
    cap = cv2.VideoCapture("/dev/device0")
    while(cap.isOpened()):
        ret, frame = cap.read()
        img_det, conf = infer(frame, infer_se)
        

    