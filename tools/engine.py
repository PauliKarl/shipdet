import math
import sys
import time
import torch

import shipdet.model.mask_rcnn

from tools.coco_utils import get_coco_api_from_dataset
from tools.coco_eval import CocoEvaluator
from shipdet.transforms import collate_fn, MetricLogger, warmup_lr_scheduler, reduce_dict, SmoothedValue

from pktool.datasets.box_convert import mask2rbbox
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, shipdet.model.MaskRCNN):
        iou_types.append("segm")
    '''
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    '''
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 10, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

@torch.no_grad()
def testBatchSize(  model,
                    data_loader,
                    objs, 
                    cls_map,
                    mask_threshold,
                    device,
                    print_freq=10):
    '''
    利用dataloader处理大图的切片检测
    args:
        model:
        data_loader:
        objs: 全局变量用于存储整个大图的检测结果
        cls_map: 类别标签的字典
    '''
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Detection:'

    for image, subimage_coordinates in metric_logger.log_every(data_loader, print_freq, header):
        image = list(img.to(device) for img in image)

        torch.cuda.synchronize()
        model_time = time.time()
        '''
        with torch.no_grad():
            predictions = model(image)
        '''
        predictions = model(image)
        model_time = time.time() - model_time

        evaluator_time = time.time()

        for prediction,subimage_coordinate in zip(predictions,subimage_coordinates):
            boxes = prediction['boxes'].to('cpu')
            label = prediction['labels'].to('cpu')
            score = prediction['scores'].to('cpu')

            masks = prediction['masks'].to('cpu')

            bboxes = boxes.detach().numpy()
            labels = label.detach().numpy()
            scores = score.detach().numpy()
            masks = masks.detach().numpy()
            for idx, mask in enumerate(masks):
                object_struct = {}
                if score[idx] < mask_threshold:
                    continue
                thetaobb, pointobb = mask2rbbox(mask[0])
                bbox = pointobb2bbox(pointobb)

                #还原目标在大图中的目标位置

                bbox = [bbox[0]+subimage_coordinate[0], bbox[1]+subimage_coordinate[1],
                        bbox[2]+subimage_coordinate[0], bbox[3]+subimage_coordinate[1]]
                x_tmp = thetaobb[0]
                y_tmp = thetaobb[1]
                thetaobb[0] = thetaobb[0] + subimage_coordinate[0]
                thetaobb[1] = thetaobb[1] + subimage_coordinate[1]
                pointobb = [pointobb[0]+subimage_coordinate[0],
                            pointobb[1]+subimage_coordinate[1],
                            pointobb[2]+subimage_coordinate[0],
                            pointobb[3]+subimage_coordinate[1],
                            pointobb[4]+subimage_coordinate[0],
                            pointobb[5]+subimage_coordinate[1],
                            pointobb[6]+subimage_coordinate[0],
                            pointobb[7]+subimage_coordinate[1]]
                
                #bbox = [xmin, ymin, xmax, ymax]
                object_struct['bbox'] = bbox
                object_struct['rbbox'] = thetaobb
                object_struct['pointobbs'] = pointobb
                object_struct['label'] = get_key(cls_map, labels[idx])
                object_struct['score'] = scores[idx]

                objs.append(object_struct)

        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    torch.set_num_threads(n_threads)
    return 

def get_key(dict, value):
    return [k for k,v in dict.items() if v==value][0]


def pointobb2bbox(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
        return [xmin, ymin, xmax, ymax]
    """
    xmin = min(pointobb[0::2])
    ymin = min(pointobb[1::2])
    xmax = max(pointobb[0::2])
    ymax = max(pointobb[1::2])
    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox

