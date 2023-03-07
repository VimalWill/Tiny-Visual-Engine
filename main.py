from infere_src import load_model, infer
import cv2
import threading


if __name__ == "__main__":
    #load the model 
    infer_se = load_model("/home/vimal/Project/models/model_quant.onnx")
    cap = cv2.VideoCapture("/dev/video0")
    while(cap.isOpened()):
        ret, frame = cap.read()
        img_det, conf = infer(frame, infer_se)
        cv2.imshow("",img_det)
        cv2.waitKey(1)

    