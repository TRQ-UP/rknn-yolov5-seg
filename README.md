# rknn-yolov5-seg
# 文件目录
###├── images: image_path
###├── model: rknn_model_path
###│   └── yolov5x_seg.rknn
###├── python: rknn model inference 
###│   ├── coco_utils.py
###│   ├── result:inference result
###│   └── yolov5_seg_image.py
###└── trans_model
    ###├── coco_subset_20.txt: qat_images_path
    ###├── convert.py: rknn model convert
    ###├── subset: qat images
    ###├── yolov5s-seg.onnx:5s_onnx
    ###└── yolov5x-seg.onnx:5x_onnx
