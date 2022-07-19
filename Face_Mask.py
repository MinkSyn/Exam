from yolov5 import detect
import os

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync


def main():
	detect.run(
		weights=curr_dict + '/Model_Parameter/YOLOv5/weight_train.pt',
		source=curr_dict + '/Images', # file/dir/URL/glob, 0 for webcam
		data=curr_dict + '/yolov5/data/facemask.yaml',
		imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,
        view_img=False,  # maximum detections per image
		project=curr_dict + '/Detect',  # save results to project/name
        name='Face_Mask',  # save results to project/name
		)

if __name__ == '__main__':
	curr_dict = os.getcwd()
	main()