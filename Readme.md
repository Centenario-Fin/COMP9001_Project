# dependency
ultralytics
numpy
pyyaml
pytorch 
torchvision
opencv-python

# check all file
OriData -- the video file, is used to test the result 
dataset -- train and validation, make sure the file path dosen't has .cashe file( which may generate through ED WRONGLY!!!)
Utils -- the python file, run train then predict get the model result by training your model
runs/Try/*anyfile* -- generate by YOLO, show the statictical result of the label and the model and the *Weight* store the model

# pretrain model
IF YOUR LAPTOP CAN NOT TRAIN MODEL ON GPU(WHICH WILL BE SO SLOW), YOU CAN USE MY MODEL
MY MODEL IS IN "Utils/runs/Try/Yolov8n/weights/best.pt"
THE DEFAULT PREDICT PYTHON FILE PATH IS TOWARDS TO MY MODEL!
IF you wnat to just use my pretrained model, just go to the project file and run: python predit_yolo.python
But if you want to use your model, check the model you train in Utils/runs and get the folder name(in pretrained model is 'Yolov8n')
and run python predit_yolo.py <your_folder>
