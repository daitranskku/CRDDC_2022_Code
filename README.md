# IEEE BigData 2022
# Crowdsensing-based Road Damage Detection Challenge (CRDDC2022)
## Team name: T22_033_Dai_Quoc_Tran
## 1. Prepare data
In [prepare_data](./prepare_data), using these notebooks to convert data format into mmdetection and yolov5 format.

## 2. Train model
### MMDet
Clone [mmdetection](https://github.com/open-mmlab/mmdetection), follow tutorial for installing. Using [vfnet_config](./mmdet_based/configs/vfnet_train_all.py) to train.

### Yolov5
Clone [Yolov5](https://github.com/ultralytics/yolov5), follow tutorial for installing. Using [road_crack_coco.yaml](./yolov5_based/road_crack_coco.yaml) to train.

## 3. Inference
### Trained weights
Download [train weights](https://o365skku-my.sharepoint.com/:f:/g/personal/daitran_o365_skku_edu/EtLUrPZsX_FCoaO6G6yOJ-QB2r7G0dxPhowcfiB6pfcOjw?e=pytn0f)

### MMDet
Norway, with its high-resolution rectangular image size, works well with VFNet. Using [inference_norway](./mmdet_based/inference_norway.ipynb) to visualize and prepare for submission. 
### Yolov5
Other countries with low-resolution images and square image sizes work well with Yolov5. Using [inference_all](./yolov5_based/inference_all.ipynb) to inference other countries and ensemble for overall countries submission.

### 4. Ensemble and submission
We can adjust the hyper-parameter for ensemble_boxes in the inference files to improve accuracy.  

