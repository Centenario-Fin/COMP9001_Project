import sys
import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import torch

# resolve the root path
SCRIPT_DIR = Path(__file__).resolve().parent

# get the model path ###!!! note the school's linux do not allow me to train on the system
# If you down load the file you can train it on your laptop
if(len(sys.argv)>1):
    MODEL_WEIGHTS_PATH = SCRIPT_DIR / 'runs' / 'Try' / sys.argv[1] / 'weights' / 'best.pt'
else:
    MODEL_WEIGHTS_PATH = SCRIPT_DIR / 'runs' / 'Try' / 'Yolov8n' / 'weights' / 'best.pt'
VIDEO_DIR = SCRIPT_DIR.parent/'OriData'

if not MODEL_WEIGHTS_PATH.exists():
    print(f"Error: Model weights not found: {MODEL_WEIGHTS_PATH}")
    exit()

CONFIDENCE_THRESHOLD = 0.5 #!! important hyper parameter if the model think the obj's probability is higher than it the model will draw a bbox.

# get the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_WEIGHTS_PATH)# load the weights
CLASS_NAMES = getattr(model, 'names', None)

# Colors for drawing
BBOX_COLOR = (0, 255, 0) # use green box to make the frame
TEXT_COLOR = (0, 0, 0) # white text

def draw_predictions(frame, results):
    boxes = results[0].boxes
    if boxes is not None:# boxes means the model detect the obj and frame it with a box. No box means no obj in the frame
        for i, box in enumerate(boxes):
            coor = box.xyxy[0].cpu().numpy().astype(int)# if the model is on cuda, the data should be processed on cuda, but cv always on cpu, so change data to cpu
            # if your laptop dosen't has cuda it stil okay
            # and the coordinate in the pixel digital frame must me integer.
            conf = box.conf[0].cpu().numpy() # check the confidence of the model out put. How many percentage the model think the obj in the frame is a "Gate"
            cls_id = int(box.cls[0].cpu().numpy()) # class ID which gate the box belong to
            if conf < CONFIDENCE_THRESHOLD:# if the confidence is too low, we do not think the bbox is real so go next bbox
                continue
            # check if the bbox has a class name, if have class name use the class name, else just show ID
            label = f"{CLASS_NAMES[cls_id] if CLASS_NAMES else f'ID:{cls_id}'}: {conf:.5f}"
            # define the text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (coor[0], coor[1]), (coor[2], coor[3]), BBOX_COLOR, 2)# frame of obj left upper piont vs right low point 
            # draw a white box fill with black text show detail about the bbox
            cv2.rectangle(frame, (coor[0], coor[1] - text_height - baseline), (coor[0] + text_width, coor[1]), (225,225,225), -1) 
            # input the text
            cv2.putText(frame, label, (coor[0], coor[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    return frame

def process_video_file(video_path):
    cap = cv2.VideoCapture(str(video_path))
    #naming format use the oriname + predicted.orisuffix
    output_path = video_path.parent / f"{video_path.stem}_predicted{video_path.suffix}" # naming format 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')# use the package to process vid
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)
        #print(results[0]) 
        annotated = draw_predictions(frame, results)
        out.write(annotated)# display the frame on the video frame
    cap.release() # when finish the video detect close the opencv process to release the resource
    out.release() # same as before
    print(f"Saved: {output_path}")

def main():
    # Search for video files in the script directory
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f for f in VIDEO_DIR.iterdir() if f.suffix.lower() in video_exts]
    if not videos:
        print("cant find video file")
        return
    for video_path in videos:
        process_video_file(video_path) #  process all vid in the dir

if __name__ == "__main__":
    main()
