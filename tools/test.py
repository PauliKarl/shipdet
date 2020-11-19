import shipdet.transforms.transforms as T
from shipdet.datasets import parse_params_xmlfile_test, BatchSizeDataset, simple_obb_xml_dump
from shipdet.transforms import collate_fn
from shipdet.datasets import get_bandmax_min,get_subimg
from tools.maskrcnn import get_model_object_detection
from tools.engine import testBatchSize
import tools.all_nms as all_nms

import os
import cv2
import gdal
import torch
import time
import numpy as np


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensorTest())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.ComposeTest(transforms)

def __save_object_crop(data,bandmax,bandmin, Obj_img_dir, objects):
        width = data.RasterXSize
        height = data.RasterYSize
        for num, object_struct in enumerate(objects):
            [xmin, ymin, xmax, ymax] = object_struct['bbox']
            xmin = max(0, min(width, int(xmin)))
            xmax = max(0, min(width, int(xmax)))
            ymin = max(0, min(height, int(ymin)))
            ymax = max(0, min(height, int(ymax)))
            _,img = get_subimg(data,bandmax,bandmin,(xmin,ymin),(xmax,ymax))
            Obj_img_fns = os.path.join(Obj_img_dir, "ObjectID_{}.jpg").format(num+1)
            cv2.imencode('.jpg', img)[1].tofile(Obj_img_fns)


def split_image(data,bandmax,bandmin,subsize=1024, gap=200, mode='keep_all'):
    subimages = dict()
    img_width = data.RasterXSize
    img_height = data.RasterYSize
    #img_height, img_height = img.shape[0], img.shape[1]
    '''
    if img_height<1024 or img_width<1024:
        img=cv2.resize(img,(1024,1024))
        subimages[(0,0)]=img
        return subimages
    '''
    start_xs = np.arange(0, img_width, subsize - gap)
    if mode == 'keep_all':
        start_xs[-1] = img_width - subsize if img_width - start_xs[-1] <= subsize else start_xs[-1]
    elif mode == 'drop_boundary':
        if img_width - start_xs[-1] < subsize - gap:
            start_xs = np.delete(start_xs, -1)
    start_xs[-1] = np.maximum(start_xs[-1], 0)

    start_ys = np.arange(0, img_height, subsize - gap)
    if mode == 'keep_all':
        start_ys[-1] = img_height - subsize if img_height - start_ys[-1] <= subsize else start_ys[-1]
    elif mode == 'drop_boundary':
        if img_height - start_ys[-1] < subsize - gap:
            start_ys = np.delete(start_ys, -1)
    start_ys[-1] = np.maximum(start_ys[-1], 0)


    for start_x in start_xs:
        for start_y in start_ys:
            end_x = np.minimum(start_x + subsize, img_width)
            end_y = np.minimum(start_y + subsize, img_height)
            #subimage = img[start_y:end_y, start_x:end_x, ...]
            subimage,_ = get_subimg(data,bandmax,bandmin,(start_x,start_y),(end_x,end_y))
            '''
            cv2.imshow('image',subimage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            coordinate = (start_x, start_y)
            subimages[coordinate] = subimage
    return subimages

##定义全局标量来存储所有batch检测的结果
global all_objects
all_objects = []

##可检测整景影像
def test_batch_size(img_file, 
                    model_file, 
                    result_dir, 
                    score_th=0.3, 
                    pretrained_model_dir=None,
                    classFile=None, 
                    flag='AUTO',
                    paramsPath=None):
    #测试单景遥感影像
    '''
    #args：
        img_file:遥感影像文件路径
        model_file: 模型文件
        result_dir: 结果文件存放路径
        score_th: 置信度分数阈值
        pretrained_model_dir: 预训练模型路径
        flag: 选择gpu/cpu、禁用CUDNN
    '''
    if img_file is None:
        #print("Please input a test image file")
        print("请选择待检测影像")
        return
    if model_file is None:
        #print("Please input a model file (*.pth)")
        print("请输入模型文件")
        return
    if result_dir is None:
        #print("Please select a dir to save result file")
        print("请选择结果文件保存路径")
        return
    if classFile is None:
        #print("Please input a class name file")
        print("请检测类别文件是否输入正确")
        return
    
    ##建立标签字典
    cls_map = {}
    #key 是字符label，value是数字label
    idxi = 1
    with open(classFile, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip("\n")
            cls_map[line] = idxi
            idxi = idxi +1
    # num_classes = 目标类数+背景
    num_classes = len(cls_map) + 1
    

    ##读取测试参数
    if os.path.exists(paramsPath):
        parameters_dict = parse_params_xmlfile_test(paramsPath)

        batch_size = parameters_dict['batchsize'] 
        num_workers = parameters_dict['num_workers']
        gap = parameters_dict['gap'] #在测试影像上裁切子图（1024x1024）时的重叠区域大小
        backbone = parameters_dict['backbone'] #模型骨架，跟训练保持一致，all=['resnet50',resnet101','resnet152','resnext50_32x4d','resnext101_32x8d']
        mask_threshold = parameters_dict['mask_threshold'] #输出检测结果时的mask二值化阈值
        nms_name = parameters_dict['nms_name'] #将所有裁切的子图检测结果合并后，选择全局nms剔除重复目标，all=['nms','soft_nms','pd_nms','roRect_nms']
        iou_threshold = parameters_dict['iou_threshold'] #所选择的nms操作的IOU阈值
        score_threshold = parameters_dict['score_threshold'] #soft_nms的阈值参数

        rpn_nms_thresh = parameters_dict['rpn_nms_thresh']
        rpn_fg_iou_thresh = parameters_dict['rpn_fg_iou_thresh']
        rpn_bg_iou_thresh = parameters_dict['rpn_bg_iou_thresh']

        box_nms_thresh = parameters_dict['box_nms_thresh']
        box_fg_iou_thresh = parameters_dict['box_fg_iou_thresh']
        box_bg_iou_thresh = parameters_dict['box_bg_iou_thresh']

    else:
        #当参数文件不存在时，设置默认的测试参数
        batch_size = 2
        num_workers = 1
        gap = 300
        backbone = 'resnet50'
        mask_threshold = 0.3
        nms_name = 'pd_nms'
        iou_threshold=0.5
        score_threshold = 0.001

        #rpn参数
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        #box参数
        box_nms_thresh = 0.5
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5



    device_set = ['AUTO', 'ONLY_CPU', 'NO_CUDNN']

    #获取影像文件名
    
    (filepath, tempfilename) = os.path.split(img_file)
    #img_name = tempfilename
    (filename, extension) = os.path.splitext(tempfilename)
    #imgFormat = extension[1:]
    
    img_name = img_file.split('/')[-1]
    imgFormat = img_file.split('.')[-1]
    filename = img_name.split('.'+imgFormat)[0]
    # filename:为不带后缀的文件名
    # img_name:为带后缀的文件名
    # 设置检测子图大小
    subsize=1024
    if "JB16" in filename or "jb16" in filename:
        subsize=2048


    #检测结果存放路径
    Obj_img_dir = result_dir + '/' + filename
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(Obj_img_dir):
        os.makedirs(Obj_img_dir)
    
    # 创建目标切片存放路径
    if not os.path.exists(os.path.join(Obj_img_dir, filename)):
        os.makedirs(os.path.join(Obj_img_dir, filename))

    #print("Start test")
    print("开始检测")
    ###通用版本
    if flag == device_set[0]:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if device == torch.device('cpu'):
            #print("CUDA is Unavailable! Turn to CPU!")
            print("CUDA 不可用， 使用CPU!")
            map_location = "cpu"
        else:
            #print("Test on GPU!")
            print("GPU检测!")
            map_location = None
    ##仅cpu版本
    if flag == device_set[1]:
        device = torch.device('cpu')
        #print("Test on CPU!")
        print("CPU检测!")
        map_location = "cpu"
    if flag == device_set[2]:
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = False
        print("GPU检测, 禁用CUDNN!")
        #print("Test on GPU, CUDNN has been disabled!")
        map_location = None
    
    #print("Loading model from %s " % model_file)
    print("正在加载模型 %s " % model_file)
    model = get_model_object_detection( num_classes, 
                                        model_dir=pretrained_model_dir, 
                                        backbone_name=backbone,
                                        rpn_nms_thresh=rpn_nms_thresh,
                                        rpn_fg_iou_thresh=rpn_fg_iou_thresh,
                                        rpn_bg_iou_thresh=rpn_bg_iou_thresh,
                                        box_nms_thresh=box_nms_thresh,
                                        box_fg_iou_thresh=box_fg_iou_thresh,
                                        box_bg_iou_thresh=box_bg_iou_thresh)




    # move model to the right device
    model.to(device)


    model.load_state_dict(torch.load(model_file, map_location=map_location))

    print("模型加载完成")
    #print("Model loaded")

    model.eval()
    #print("Model eval()")
    #   读取测试图片：
    #print("Loading %s" % img_file)
    print("正在加载图像 %s" % img_file)
    '''
    img_rgb, img_bgr = read_gaofen(img_file=img_file, imgFormat=imgFormat)
    
    if img_rgb is None:
        #print("Please input correct image file!")
        print("请输入正确的图像文件！")
        return
    else:
        img_height, img_width = img_rgb.shape[0], img_rgb.shape[1]
        print("图像加载完成, 高度: %d, 宽度: %d" % (img_height, img_width))
        #print("Image loaded, height: %d, width: %d" % (img_height, img_width))
    '''
    # 将大图拆分成1024x1024的子图
    data_img = gdal.Open(img_file)
    bandmax,bandmin=get_bandmax_min(data_img)

    subimages = split_image(data_img,bandmax,bandmin,subsize=subsize, gap=gap)

    subimage_coordinates = list(subimages.keys())
    number_sub_img = len(subimage_coordinates)
    print("{} subimages need to be detected!".format(number_sub_img))
    #根据裁切后的子图字典变量subimages创建dataloader

    dataset_test = BatchSizeDataset(subimages, get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                                   batch_size=batch_size, 
                                                   shuffle=False, 
                                                   num_workers=num_workers, 
                                                   collate_fn=collate_fn)

    #开始检测
    testBatchSize(model, data_loader_test, all_objects, cls_map, mask_threshold, device=device)
    '''
    with torch.no_grad():
        predictions = model(im)
    '''
    
    ##剔除重复的目标舰船

    print("postprocessing...")
    postprocess_time = time.time()
    final_objects = all_nms.__dict__[nms_name](
        all_objects=all_objects,
        iou_threshold = iou_threshold,
        score_threshold=score_threshold,
    )
    postprocess_time = time.time()-postprocess_time

    print("postprocess time: {:.4f} s".format(postprocess_time))
    print("目标总个数: %d " % len(final_objects))
    
    #保存目标切片
    save_crop_dir = Obj_img_dir + '/' + filename
    print("正在保存目标切片至路径 %s" % (Obj_img_dir + '/' + filename))
    #print("cropping all objects to %s" % (Obj_img_dir + '/' + "object_crop"))
    __save_object_crop(data=data_img,bandmax=bandmax,bandmin=bandmin, Obj_img_dir=save_crop_dir, objects=final_objects)
    #save_object_crop(img=img_bgr, Obj_img_dir=save_crop_dir, objects=final_objects)

    ##检测结果写入xml文件
    simple_obb_xml_dump(final_objects, filename, Obj_img_dir)
    '''
    #保存大图的斜框结果png
    bboxes_curr,pbboxes,all_scores,all_labels=[],[],[],[]
    for object_struct in final_objects:
        bbox = object_struct['bbox']
        pointobb = object_struct['pointobbs']
        label = cls_map[object_struct['label']]
        scores = object_struct['score']

        bboxes_curr.append(bbox)
        pbboxes.append(pointobb)
        all_scores.append(scores)
        all_labels.append(label)


    # 斜框可视化结果保存路径
    out_img_robbox = Obj_img_dir + '/' + filename + "_angle.png"
    print("检测可视化结果保存为 %s" % out_img_robbox)
    #print("Save 'robbox' to %s" % out_img_robbox)
    img_ro = img_bgr.copy()
    test_utils.imshow_rbboxes(img_or_path=img_ro, 
                              rbboxes=pbboxes, 
                              labels=all_labels, 
                              scores=all_scores,
                              cls_map=cls_map,
                              show_label=False,
                              show_score=True, 
                              show=False, 
                              out_file=out_img_robbox)
    '''



if __name__=='__main__':
    img_file ='F:/data/demo/GF1_PMS1_E113.9_N22.2_20181002_L1A0003492330.tif'#'F:/data/demo/testimg_01.png' # "F:/data/demo/GF1_PMS1_E113.9_N22.2_20181002_L1A0003492330.tif" #'F:/data/demo/small.png'
    model_file = "F:/qt_test/work_dir/gf2/11.pth"
    result_dir = "F:/data/demo/results"
    pretrained_model_dir= "F:/qt_test/pre_trained"
    classFile = "F:/qt_test/classesName/classesname.txt"

    parasPath = os.path.join(pretrained_model_dir, 'parameters.xml')
    
    test_batch_size(img_file, model_file, result_dir, 0.3, pretrained_model_dir, classFile, flag="NO_CUDNN", paramsPath=parasPath)
    