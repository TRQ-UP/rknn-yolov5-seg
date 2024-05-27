# rknn-yolov5-seg

# 1.文件目录

- ├── images: image_path
  
- ├── model: rknn_model_path
  
- │   └── yolov5x_seg.rknn
  
- ├── python: rknn model inference
  
- │   ├── coco_utils.py
  
- │   ├── result:inference result
  
- │   └── yolov5_seg_image.py
  
- └── trans_model
  
- ├── coco_subset_20.txt: qat_images_path
  
- ├── convert.py: rknn model convert
  
- ├── subset: qat images
  
- ├── yolov5s-seg.onnx:5s_onnx
  
- └── yolov5x-seg.onnx:5x_onnx
  

2.模型转换

- cd trans_model
  
- python3 convert.py model.onnx rk3588/rk3568
  

    rknn模型转换在model目录下

3.模型推理

- python3 yolov5_seg_image.py
  
  结果保存在当前文件夹中的result.png
