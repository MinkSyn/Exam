from yolov5 import detect
import os

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