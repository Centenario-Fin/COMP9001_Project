import os
import yaml
from ultralytics import YOLO
from pathlib import Path
# data set setting
NUM_CLASSES = 3
CLASS_NAMES = ['Gate1', 'Gate2', 'Gate3'] # mark ID matching
NUM_KEYPOINTS = 4
KEYPOINT_DIM = 3 #pisition and visiabl=ility

# model setting 
MODEL_NAME = 'yolov8n-pose.yaml'
EPOCHS = 500
BATCH_SIZE = 10
IMG_SIZE = 640
PROJECT_NAME = 'Try'
RUN_NAME = 'Yolov8n' 

# path setting 
ROOT_DIR = Path(__file__).resolve().parent 
print(str(ROOT_DIR))
DATASET_DIR = ROOT_DIR.parent / 'dataset'  
print(str(DATASET_DIR))      
YAML_FILE_PATH = ROOT_DIR / 'dataset.yaml' 

#dataset yaml which used for training 
dataset_yaml_content = {
    'path': str(DATASET_DIR),             
    'train': str(Path('images') / 'train'),        
    'val': str(Path('images') / 'val'),            
    'kpt_shape': [NUM_KEYPOINTS, KEYPOINT_DIM],
    'nc': NUM_CLASSES,
    'names': {i: name for i, name in enumerate(CLASS_NAMES)}
}

# after change the yaml file, then close it in order to void some random change 
with open(YAML_FILE_PATH, 'w') as f: # use write model open the yaml file. write the new status in it 
    # change the python file method to the 
    yaml.dump(dataset_yaml_content, f, sort_keys=False, default_flow_style=None)



# training method
def main():
    try:
        model = YOLO(MODEL_NAME) 
        results = model.train(
            data=str(YAML_FILE_PATH.resolve()), 
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=str(ROOT_DIR / 'runs' / PROJECT_NAME), 
            name=RUN_NAME,
            exist_ok=False, 
            optimizer = 'auto',
        )
    except Exception as e:
        import traceback
        print(f"Wrong: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
