from shipdet import GaoFenDataset, parse_params_xmlfile, Logger


import shipdet.transforms.transforms as T
from shipdet.transforms import collate_fn
from tools.engine import train_one_epoch, evaluate
from tools.maskrcnn import get_model_object_detection
import os
import sys
import torch
import math
import time


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


#############训练算法################
def train(root_dir, 
          model_save_dir, 
          pretrained_model_dir=None, 
          epochs_num = 24, 
          classFile=None, 
          flag = "NO_CUDNN",
          learning_rate = 0.01,
          paramsPath=None):

    '''
    args:
        root_dir: 训练数据文件夹，包括labelxmls,images
        model_save_dir: 保存训练模型文件
        pretrained_model_dir：与训练模型保存的位置
        epochs_num：训练的次数
        learning_rate：学习率设置
    future add:
        batch_size
        num_classes
        opt+lr_scheduler:
            momentum
            weight_decay
            milestones
            gamma
    python objectDetection.py --operation train --root_dir f:/data/gf3_v2/trainval --model_save_dir f:/data/gf3_v2/work_dir --epoch_num 12 --pretrained_model_dir g:/503/MB1/resources/algorithm/pre_trained --device_flag NO_CUDNN
    '''

    ##建立标签字典
    cls_map = {}
    #key 是字符label，value是数字label
    idxi = 1
    with open(classFile, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip("\n")
            cls_map[line] = idxi
            idxi = idxi +1
    num_classes = len(cls_map) + 1

    #读取参数文件中的训练参数，如果文件不存在，则设置成默认值
    if os.path.exists(paramsPath):
        parameters_dict = parse_params_xmlfile(paramsPath)

        batch_size = parameters_dict['batchsize']
        num_workers = parameters_dict['num_workers']
        learning_rate = parameters_dict['learning_rate']
        test_ratio = parameters_dict['test_ratio']
        test_dir = parameters_dict['test_dir']
        resume_from = parameters_dict['resume_from']
        backbone = parameters_dict['backbone']
        save_log = parameters_dict['save_log_path']

        rpn_nms_thresh = parameters_dict['rpn_nms_thresh']
        rpn_fg_iou_thresh = parameters_dict['rpn_fg_iou_thresh']
        rpn_bg_iou_thresh = parameters_dict['rpn_bg_iou_thresh']
        box_nms_thresh = parameters_dict['box_nms_thresh']
        box_fg_iou_thresh = parameters_dict['box_fg_iou_thresh']
        box_bg_iou_thresh = parameters_dict['box_bg_iou_thresh']
    else:
        batch_size = 2
        num_workers = 1
        learning_rate = 0.01
        test_ratio = 0.2
        test_dir = None
        resume_from = None
        backbone = 'resnet50'
        save_log = None

        #rpn参数
        rpn_nms_thresh = 0.7
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        #box参数
        box_nms_thresh = 0.5
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
    

    #训练时，每个epoch的输出结果写在record.log文件中
    recordlog_path = model_save_dir + "/record.log"
    
    #设备模式选择，默认为AUTO
    device_set = ['GPU', 'NO_CUDNN']

    if root_dir is None:
        print("Please input a test image file")
        #print("请输入训练数据集")
        return
    if model_save_dir is None:
        print("Please input a dir to save the model file")
        #print("请选择模型保存路径")
        return

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # train on the GPU or on the CPU, if a GPU is not available
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    device = torch.device('cuda') 

    if flag == device_set[1]:
        torch.backends.cudnn.enabled = False
        print("Train on GPU, CUDNN has been disabled!")
        #print("GPU训练，禁用CUDNN!")
    # our dataset has two classes only - background and person

    dataset = GaoFenDataset(root_dir, get_transform(train=True), cls_map=cls_map)
    ##测试数据集，如果没有单独设置test数据集文件夹，则按照test_ratio的比率从train数据中选取test样本
    if test_dir is not None:
        no_test=False
        dataset_test = GaoFenDataset(test_dir, get_transform(train=False), cls_map=cls_map)
    elif test_ratio is not None:
        no_test=False
        #dataset = ShipWudanDataset(root_dir, get_transform(train=True), cls_map=cls_map)
        #dataset_test = ShipWudanDataset(test_dir, get_transform(train=False), cls_map=cls_map)
        test_num = 0
        # use our dataset and defined transformations
        
        dataset_test = GaoFenDataset(root_dir, get_transform(train=False), cls_map=cls_map)

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        
        test_num=int(len(dataset)*test_ratio)

        dataset = torch.utils.data.Subset(dataset, indices[:-test_num])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_num:])
    else:
        no_test=True
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              num_workers=num_workers, 
                                              collate_fn=collate_fn)
    if not no_test:
        data_loader_test = torch.utils.data.DataLoader(dataset_test, 
                                                    batch_size=batch_size, 
                                                    shuffle=False, 
                                                    num_workers=num_workers, 
                                                    collate_fn=collate_fn)

    # get the model using our helper function
    model = get_model_object_detection( num_classes, 
                                        model_dir=pretrained_model_dir, 
                                        backbone_name=backbone,
                                        rpn_nms_thresh=rpn_nms_thresh,
                                        rpn_fg_iou_thresh=rpn_fg_iou_thresh,
                                        rpn_bg_iou_thresh=rpn_bg_iou_thresh,
                                        box_nms_thresh=box_nms_thresh,
                                        box_fg_iou_thresh=box_fg_iou_thresh,
                                        box_bg_iou_thresh=box_bg_iou_thresh)

    multi_GPUs=False
    #multi-GPU
    if torch.cuda.device_count()>1:
        multi_GPUs=True
        model = torch.nn.DataParallel(model)

    # move model to the right device
    model.to(device)



    # construct an optimizer
    if multi_GPUs:
        params = [p for p in model.module.parameters() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(params, 
                                lr=learning_rate,
                                momentum=0.9, 
                                weight_decay=0.0001)
    # and a learning rate scheduler
    '''
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    '''
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[8,11],
                                                        gamma=0.1)    

    #True则在resume_from的基础上继续训练，反之从零开始
    if resume_from!=None:
        model.load_state_dict(torch.load(resume_from))
        (_, modelfilename) = os.path.split(resume_from)
        (tempfilename, _) = os.path.splitext(modelfilename)
        start_epochs = int(tempfilename)+1
    else:
        start_epochs = 0
    
    #save_log为None时，不保存训练过程的输出结果
    if save_log is not None:
        #'record.log'
        sys.stdout = Logger(recordlog_path, sys.stdout)
    
    # 开始训练
    num_epochs = epochs_num
    print("Successfully loaded train parameters, start...")
    for epoch in range(start_epochs, num_epochs):
        
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        if multi_GPUs:
            torch.save(model.module.state_dict(), model_save_dir + '/{}.pth'.format(epoch))
        else:
            torch.save(model.state_dict(), model_save_dir + '/{}.pth'.format(epoch))

        # evaluate on the test dataset
        if not no_test:
            evaluate(model, data_loader_test, device=device)
    print("That's it!")
    #print("训练完成")





if __name__=="__main__":
    import multiprocessing 
    multiprocessing.freeze_support()
    root_dir="f:/data/gf2_v2/trainval"
    model_save_dir="f:/data/gf2_v2/model"
    pretrained_model_dir= "f:/qt_test/pre_trained"
    classFile='f:/qt_test/classesName/classesname.txt'
    parafile= "f:/qt_test/pre_trained/parameters.xml"
    train(root_dir=root_dir,model_save_dir=model_save_dir,pretrained_model_dir=pretrained_model_dir, classFile=classFile, flag="NO_CUDNN",paramsPath=parafile)    



