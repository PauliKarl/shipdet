import random
import torch
import numpy as np
import cv2
import math
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def thetaobb2pointobb(thetaobb):
    """
    docstring here
        :param self: 
        :param thetaobb: list, [x, y, w, h, theta]
    """
    box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4]*180.0/np.pi))
    box = np.reshape(box, [-1, ]).tolist()
    pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

    return pointobb
def rotate_pointobb(pointobb, theta, anchor=None):
    """rotate pointobb around anchor
    
    Arguments:
        pointobb {list or numpy.ndarray, [1x8]} -- vertices of obb region
        theta {int, rad} -- angle in radian measure
    
    Keyword Arguments:
        anchor {list or tuple} -- fixed position during rotation (default: {None}, use left-top vertice as the anchor)
    
    Returns:
        numpy.ndarray, [1x8] -- rotated pointobb
    """
    if type(pointobb) == list:
        pointobb = np.array(pointobb)
    if type(anchor) == list:
        anchor = np.array(anchor).reshape(2, 1)
    v = pointobb.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:,:1]

    rotate_mat = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    res = np.dot(rotate_mat, v - anchor)
    
    return (res + anchor).T.reshape(-1)
def pointobb2thetaobb(pointobb):
    """convert pointobb to thetaobb
    Input:
        pointobb (list[1x8]): [x1, y1, x2, y2, x3, y3, x4, y4]
    Output:
        thetaobb (list[1x5])
    """
    pointobb = np.int0(np.array(pointobb))
    pointobb.resize(4, 2)
    rect = cv2.minAreaRect(pointobb)
    x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    theta = theta / 180.0 * np.pi
    thetaobb = [x, y, w, h, theta]
    return thetaobb
def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             border_value=0,
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple): Center of the rotation in the source image, by default
            it is the center of the image.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[-2::]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)
    return rotated
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
class RotateAngle(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, image, target):
        if random.random() < 0.5:
            boxes=target["boxes"]
            masks=target["masks"]
            thetaobbs=target["thetaobbes"]
            labels = target["labels"]
            bboxes = boxes.detach().numpy()
            labels = labels.detach().numpy()
            masks = masks.detach().numpy()

            pointobbs = [thetaobb2pointobb(thetaobb) for thetaobb in thetaobbs]
            #img = generate_image(1024, 1024)
            img_origin = np.transpose(image.numpy(),(2,1,0))
            #imshow_rbboxes(img, thetaobbs, win_name='origin')

            rotation_anchor = [image.shape[0]//2, image.shape[1]//2]
            
            rotated_img = img_origin.copy()
            rotated_img = imrotate(img_origin, self.angle)

            rotated_pointobbs = [rotate_pointobb(pointobb, self.angle*np.pi/180, rotation_anchor) for pointobb in pointobbs]

            rotated_thetaobbs = [pointobb2thetaobb(rotated_pointobb) for rotated_pointobb in rotated_pointobbs]

            rotated_thetaobbs_ = np.array([obb for obb in rotated_thetaobbs])

            cx_bool = np.logical_and(rotated_thetaobbs_[:, 0] >= 0, rotated_thetaobbs_[:, 0] < 1024)
            cy_bool = np.logical_and(rotated_thetaobbs_[:, 1] >= 0, rotated_thetaobbs_[:, 1] < 1024)

            rotation_thetaobbs = rotated_thetaobbs_[np.logical_and(cx_bool, cy_bool)]
            mm = np.logical_and(cx_bool, cy_bool)
            rotation_labels = []
            for idx, flag in enumerate(mm):
                if flag:
                    rotation_labels.append(labels[idx])

            #imshow_rbboxes(rotated_img, rotated_thetaobbs, win_name='rotated')
            bb = rotation_thetaobbs.tolist()
            
            robboxes = [pointobb2bbox(thetaobb) for thetaobb in bb]
            romask = []
            for mask in enumerate(masks):
                romask.append(imrotate(mask, self.angle))
            
            target["boxes"] = torch.as_tensor(robboxes, dtype=torch.float32)
            target["thetaobbes"]=torch.as_tensor(bb, dtype=torch.float32)
            target["area"] = torch.as_tensor(bb[:, 2]*bb[:,3],dtype=torch.float32)
            target["labels"]=torch.as_tensor(rotation_labels, dtype=torch.int64)
            target["masks"]=torch.as_tensor(masks, dtype=torch.uint8)
        return rotated_img, target

        '''
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        '''
#######用于处理BatchSizeDataset的操作
class ToTensorTest(object):
    def __call__(self, image, subimage_coordinate):
        image = F.to_tensor(image)
        return image, subimage_coordinate

class ComposeTest(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, subimage_coordinate):
        for t in self.transforms:
            image, subimage_coordinate = t(image, subimage_coordinate)
        return image, subimage_coordinate