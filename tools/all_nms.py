import numpy as np
import cv2
from shapely.geometry import Polygon

__all__=[
    'nms', 'soft_nms', 'pd_nms', 'roRect_nms', 'pd_nms_old', 'roRect_soft_nms'
]

def roRect_nms(all_objects, iou_threshold=0.5, **kwargs):
    '''
    argvs
    input:
        all_objects: 全局目标结果list[object_struct]
                                object_struct['bbox'] = bbox
                                object_struct['rbbox'] = thetaobb
                                object_struct['pointobbs'] = pointobb
                                object_struct['label'] = get_key(cls_map, labels[idx])
                                object_struct['score'] = scores[idx]
        iou_threshold: nms阈值
    output:
        tmp_objects: nms处理后的结果list
    '''
    tmp_objects = [] # 输出结果
    re_idx=[] #存放被判定为应该剔除的检测结果的索引
    num = len(all_objects)
    for idx, obj in enumerate(all_objects):
        if idx in re_idx:
            continue
        pointobb1 = obj['pointobbs']
        score1 = obj['score']
        thetaobb1=obj['rbbox']
        for idx_c in range(idx+1, num):
            if idx_c in re_idx:
                continue
            obj_c = all_objects[idx_c]
            pointobb2 = obj_c['pointobbs']
            score2 = obj_c['score']
            thetaobb2=obj['rbbox']
            if abs(thetaobb1[0]-thetaobb2[0])>256 or abs(thetaobb1[1]-thetaobb2[1])>256:
                continue
            iou, inter, area1, area2= theta_iou(pointobb1, pointobb2)
            id_m = None
            ##判断剔除条件
            if iou >iou_threshold:
                if score1==score2: #置信度分数相等时，保留面积大的实例；否则，保留置信度较大的实例
                    id_m = idx if area1<area2 else idx_c
                else: 
                    id_m = idx if score1<score2 else idx_c
            elif inter>0.8*area1: # 当出现包含关系时，保留面积大的实例
                id_m = idx 
            elif inter>0.8*area2: # 当出现包含关系时，保留面积大的实例
                id_m = idx_c
            
            re_idx.append(id_m) ##id_m代表需要被剔除的目标实例

        if idx not in re_idx:
            tmp_objects.append(obj)
    return tmp_objects

def pd_nms_old( all_objects, iou_threshold=0.5,  **kwargs):
    tmp_objects = []
    re_idx=[]
    num = len(all_objects)
    for idx, obj in enumerate(all_objects):
        if idx in re_idx:
            continue
        bbox1 = obj['bbox']
        score1 = obj['score']
        for idx_c in range(idx+1, num):
            if idx_c in re_idx:
                continue
            obj_c = all_objects[idx_c]
            bbox2 = obj_c['bbox']
            score2 = obj_c['score']
            iou, inter, area1, area2= box_iou(bbox1, bbox2)
            if iou >iou_threshold:
                id_m = idx if score1<score2 else idx_c
                re_idx.append(id_m)
            elif inter>0.9*area1 or inter>0.9*area2:
                id_m = idx if score1<score2 else idx_c
                re_idx.append(id_m)
        if idx not in re_idx:
            tmp_objects.append(obj)
    return tmp_objects

def nms( all_objects, iou_threshold=0.5, **kwargs):
    """non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU)

    Arguments:
        boxes {np.array} -- [N * 4]
        scores {np.array} -- [N * 1]
        iou_threshold {float} -- threshold for IoU
    """
    boxes_list = []
    scores_list = []
    temp_objects = []
    for obj in all_objects:
        bbox = obj['bbox']
        score = obj['score']
        boxes_list.append(bbox)
        scores_list.append(score)

    boxes = np.array(boxes_list)
    scores = np.array(scores_list)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    order = scores.argsort()[::-1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep = []
    while order.size > 0:
        best_box = order[0]
        keep.append(best_box)

        inter_x1 = np.maximum(x1[order[1:]], x1[best_box])
        inter_y1 = np.maximum(y1[order[1:]], y1[best_box])
        inter_x2 = np.minimum(x2[order[1:]], x2[best_box])
        inter_y2 = np.minimum(y2[order[1:]], y2[best_box])

        inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0.0)
        inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0.0)

        inter = inter_w * inter_h

        iou = inter / (areas[best_box] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]

        order = order[inds + 1]

    #return keep
    for idx in keep:
        temp_objects.append(all_objects[idx])
    return temp_objects

def soft_nms( all_objects, iou_threshold=0.5, score_threshold=0.001, **kwargs):
    """soft non-maximum suppression (soft-NMS) on the boxes according to their intersection-over-union (IoU)

    Arguments:
        boxes {np.array} -- [N * 4]
        scores {np.array} -- [N * 1]
        iou_threshold {float} -- threshold for IoU
        score_threshold {float} -- threshold for score
    """
    boxes_list = []
    scores_list = []
    temp_objects = []
    for obj in all_objects:
        bbox = obj['bbox']
        score = obj['score']
        boxes_list.append(bbox)
        scores_list.append(score)

    boxes = np.array(boxes_list)
    scores = np.array(scores_list)


    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    order = scores.argsort()[::-1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep = []
    while order.size > 0:
        best_box = order[0]
        keep.append(best_box)

        inter_x1 = np.maximum(x1[order[1:]], x1[best_box])
        inter_y1 = np.maximum(y1[order[1:]], y1[best_box])
        inter_x2 = np.minimum(x2[order[1:]], x2[best_box])
        inter_y2 = np.minimum(y2[order[1:]], y2[best_box])

        inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0.0)
        inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0.0)

        inter = inter_w * inter_h

        iou = inter / (areas[best_box] + areas[order[1:]] - inter)

        weights = np.ones(iou.shape) - iou

        scores[order[1:]] = weights * scores[order[1:]]

        inds = np.where(scores[order[1:]] > score_threshold)[0]

        order = order[inds + 1]

    #return keep
    for idx in keep:
        temp_objects.append(all_objects[idx])
    return temp_objects

def roRect_soft_nms(all_objects, iou_threshold=0.5, score_threshold=0.001, **kwargs):
    """rotation non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU)
    Arguments:
        rboxes {np.array} -- [N * 5] (cx, cy, w, h, theta (rad/s))
        scores {np.array} -- [N * 1]
        iou_threshold {float} -- threshold for IoU
    """
    rboxes_list = []
    scores_list = []
    temp_objects = []
    for obj in all_objects:
        rbbox = obj['rbbox']
        score = obj['score']
        rboxes_list.append(rbbox)
        scores_list.append(score)

    rbboxes = np.array(rboxes_list)
    scores = np.array(scores_list)
    cx = rbboxes[:, 0]
    cy = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    theta = rbboxes[:, 4] * 180.0 / np.pi

    order = scores.argsort()[::-1]

    areas = w * h
    
    keep = []
    while order.size > 0:
        best_rbox_idx = order[0]
        keep.append(best_rbox_idx)

        best_rbbox = np.array([cx[best_rbox_idx], 
                               cy[best_rbox_idx], 
                               w[best_rbox_idx], 
                               h[best_rbox_idx], 
                               theta[best_rbox_idx]])
        remain_rbboxes = np.hstack((cx[order[1:]].reshape(1, -1).T, 
                                    cy[order[1:]].reshape(1,-1).T, 
                                    w[order[1:]].reshape(1,-1).T, 
                                    h[order[1:]].reshape(1,-1).T, 
                                    theta[order[1:]].reshape(1,-1).T))

        inters = []
        for remain_rbbox in remain_rbboxes:
            rbbox1 = ((best_rbbox[0], best_rbbox[1]), (best_rbbox[2], best_rbbox[3]), best_rbbox[4])
            rbbox2 = ((remain_rbbox[0], remain_rbbox[1]), (remain_rbbox[2], remain_rbbox[3]), remain_rbbox[4])
            inter = cv2.rotatedRectangleIntersection(rbbox1, rbbox2)[1]
            if inter is not None:
                inter_pts = cv2.convexHull(inter, returnPoints=True)
                inter = cv2.contourArea(inter_pts)
                inters.append(inter)
            else:
                inters.append(0)

        inters = np.array(inters)

        iou = inters / (areas[best_rbox_idx] + areas[order[1:]] - inters)

        weights = np.ones(iou.shape) - iou

        scores[order[1:]] = weights * scores[order[1:]]

        inds = np.where(scores[order[1:]] > score_threshold)[0]
        
        order = order[inds + 1]

    #return keep
    for idx in keep:
        temp_objects.append(all_objects[idx])
    return temp_objects

def pd_nms( all_objects, iou_threshold=0.4,  **kwargs):
    tmp_objects = []
    re_idx=[]
    num = len(all_objects)
    for idx, obj in enumerate(all_objects):
        if idx in re_idx:
            continue
        bbox1 = obj['bbox']
        score1 = obj['score']
        thetaobb1=obj['rbbox']
        for idx_c in range(idx+1, num):
            if idx_c in re_idx:
                continue
            obj_c = all_objects[idx_c]
            bbox2 = obj_c['bbox']
            score2 = obj_c['score']
            thetaobb2=obj_c['rbbox']
            if abs(thetaobb1[0]-thetaobb2[0])>256 or abs(thetaobb1[1]-thetaobb2[1])>256:
                continue
            iou, inter, area1, area2= box_iou(bbox1, bbox2)
            id_m = None
            if iou >iou_threshold:
                if score1==score2: #置信度分数相等时，保留面积大的实例；否则，保留置信度较大的实例
                    id_m = idx if area1<area2 else idx_c
                else: 
                    id_m = idx if score1<score2 else idx_c
            elif inter>0.9*area1: # 当出现包含关系时，保留面积大的实例
                id_m = idx 
            elif inter>0.9*area2: # 当出现包含关系时，保留面积大的实例
                id_m = idx_c
            re_idx.append(id_m)

        if idx not in re_idx:
            tmp_objects.append(obj)
    
    return tmp_objects

def box_iou(bbox1, bbox2):

    area1 = box_area(bbox1)
    area2 = box_area(bbox2)

    lt_x, lt_y = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    rb_x, rb_y = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
    if rb_x-lt_x>0 and rb_y-lt_y>0:
        inter = (lt_x-rb_x)*(lt_y-rb_y)
    else:
        inter = 0
    iou = inter / (area1 + area2 - inter)
    return iou, inter, area1, area2

def box_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def theta_iou(pointobb1, pointobb2):

    """
    docstring here
        :param pointobb1, pointobb2 : list, [x1, y1, x2, y2, x3, y3, x4, y4]
        return: iou, inter, area1, area2
    """
    pointobb1=np.asarray(pointobb1)
    pointobb2=np.asarray(pointobb2)
    pointobb1 = Polygon(pointobb1[:8].reshape((4, 2)))
    pointobb2 = Polygon(pointobb2[:8].reshape((4, 2)))
    if not pointobb1.is_valid or not pointobb2.is_valid:
        return 0, 0, 0, 0
    inter = Polygon(pointobb1).intersection(Polygon(pointobb2)).area
    union = pointobb1.area + pointobb2.area - inter
    if union == 0:
        return 0, inter, pointobb1.area, pointobb2.area
    else:
        return inter/union, inter, pointobb1.area, pointobb2.area


def rotation_iou(rboxe1, rboxe2):
    """IoU of rboxe1 and rboxe2
    
    Arguments:
        rboxe1, rboxe2 {list} -- [cx, cy, w, h, theat(rad/s)]
    """

    rbbox1 = ((rboxe1[0], rboxe1[1]), (rboxe1[2], rboxe1[3]), rboxe1[4] * 180.0 / np.pi)
    rbbox2 = ((rboxe2[0], rboxe2[1]), (rboxe2[2], rboxe2[3]), rboxe2[4] * 180.0 / np.pi)
    inter = cv2.rotatedRectangleIntersection(rbbox1, rbbox2)[1]
    if inter is not None:
        inter_pts = cv2.convexHull(inter, returnPoints=True)
        inter = cv2.contourArea(inter_pts)
    else:
        inter = 0

    area1 = (rboxe1[2] * rboxe1[3])
    area2 = (rboxe2[2] * rboxe2[3])
    iou = inter / ( area1+ area2 - inter)

    return iou, inter, area1, area2