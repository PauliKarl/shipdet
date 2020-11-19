import os
import xml.etree.ElementTree as ET
import gdal
import cv2
import numpy as np
from pycocotools import mask as maskUtils
import torch

class GaoFenDataset(object):
    def __init__(self, root, transforms, cls_map):
        self.root = root
        self.transforms = transforms
        self.cls_map=cls_map
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labelxmls"))))


    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labelxmls", self.labels[idx])
        #img_cv=cv2.imread(img_path)
        #b,g,r = cv2.split(img_cv)
        #img = cv2.merge([r, g, b])
        #img=img[...,::-1]
        #img = Image.open(img_path).convert("RGB")
        img, _ = self.read_gaofen(img_path)

        tree = ET.parse(label_path)
        root = tree.getroot()
        objects = []

        imgsize=root.find('size')
        width = int(imgsize.find('width').text)
        height = int(imgsize.find('height').text)

        boxes = []
        thetaobbes = []
        pointobbes = []
        masks = []
        labels = []
        for single_object in root.findall('object'):
            robndbox = single_object.find('robndbox')
            cx = float(robndbox.find('cx').text)
            cy = float(robndbox.find('cy').text)
            w  = float(robndbox.find('w').text)
            h  = float(robndbox.find('h').text)
            theta = float(robndbox.find('angle').text)
            thetaobb = [cx,cy,w,h,theta]

            classname = single_object.find('name').text
            label = self.cls_map[classname]

            labels.append(label)
            
            box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4] * 180.0 / np.pi))
            box = np.reshape(box, [-1, ]).tolist()
            pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]
            pointobb = [i if i<1024 else 1024 for i in pointobb]
            pointobb = [i if i>0 else 0 for i in pointobb]

            xmin = max(0,min(pointobb[0::2]))
            ymin = max(0,min(pointobb[1::2]))
            xmax = min(max(pointobb[0::2]),width)
            ymax = min(max(pointobb[1::2]),height)
            bbox = [xmin, ymin, xmax, ymax]
            
            reference_bbox = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            reference_bbox = np.array(reference_bbox)
            normalize = np.array([1.0, 1.0] * 4)
            combinate = [np.roll(pointobb, 0), np.roll(pointobb, 2), np.roll(pointobb, 4), np.roll(pointobb, 6)]
            distances = np.array([np.sum(((coord - reference_bbox) / normalize)**2) for coord in combinate])
            sorted = distances.argsort()
            pointobb = combinate[sorted[0]].tolist()

            thetaobbes.append(thetaobb)
            pointobbes.append(pointobb)
            boxes.append(bbox)

            segm = [pointobb]            
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
            mask = maskUtils.decode(rle)

            masks.append(mask)

        
        num_objs = len(masks)
        thetaobbes = torch.as_tensor(thetaobbes, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        #labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)


        image_id = torch.tensor([idx])
        area = thetaobbes[:, 2]*thetaobbes[:,3]
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["thetaobbes"]=thetaobbes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)
    

    def read_gaofen(self,img_file):
        # 返回的图像uint8类型，[r,g,b]
        if img_file is not None:
            data = gdal.Open(img_file)
            #print("finished gdal.Open")
            width = data.RasterXSize
            height = data.RasterYSize

            if data.RasterCount==4:
                #高分2多光谱，4bands，[b,g,r,Nr]
                band1 = data.GetRasterBand(3)
                img_r = band1.ReadAsArray(0,0,width,height)
                img_r = (img_r-img_r.min())/(img_r.max()-img_r.min())
                img_r = np.round(img_r*255)
                img_r = np.uint8(img_r)

                band2 = data.GetRasterBand(2)
                img_g = band2.ReadAsArray(0,0,width,height)
                img_g = (img_g-img_g.min())/(img_g.max()-img_g.min())
                img_g = np.round(img_g*255)
                img_g = np.uint8(img_g)

                band3 = data.GetRasterBand(1)
                img_b = band3.ReadAsArray(0,0,width,height)
                img_b = (img_b-img_b.min())/(img_b.max()-img_b.min())
                img_b = np.round(img_b*255)
                img_b = np.uint8(img_b)
                img_rgb = cv2.merge([img_r, img_g, img_b])
                img_bgr = cv2.merge([img_b, img_g, img_r])

            elif data.RasterCount==3:
                #高分1三通道图
                band1 = data.GetRasterBand(1)
                img_r = band1.ReadAsArray(0,0,width,height)
                img_r = (img_r-img_r.min())/(img_r.max()-img_r.min())
                img_r = np.round(img_r*255)
                img_r = np.uint8(img_r)

                band2 = data.GetRasterBand(2)
                img_g = band2.ReadAsArray(0,0,width,height)
                img_g = (img_g-img_g.min())/(img_g.max()-img_g.min())
                img_g = np.round(img_g*255)
                img_g = np.uint8(img_g)

                band3 = data.GetRasterBand(3)
                img_b = band3.ReadAsArray(0,0,width,height)
                img_b = (img_b-img_b.min())/(img_b.max()-img_b.min())
                img_b = np.round(img_b*255)
                img_b = np.uint8(img_b)
                img_rgb = cv2.merge([img_r, img_g, img_b])
                img_bgr = cv2.merge([img_b, img_g, img_r])

            elif data.RasterCount == 1:
                band1 = data.GetRasterBand(1)
                img_arr = band1.ReadAsArray(0,0,width,height)
                img_arr = (img_arr-img_arr.min())/(img_arr.max()-img_arr.min())
                img_arr = np.uint8(np.round(img_arr*255))

                img_rgb = cv2.merge([img_arr,img_arr,img_arr])
                img_bgr = cv2.merge([img_arr,img_arr,img_arr])
        else:
            #raise TypeError("Please input correct image format: png, jpg, tif/tiff!")
            img_rgb = None
            img_bgr = None
        return img_rgb, img_bgr


class BatchSizeDataset(object):
    '''
    用于处理检测大图，返回子图img和子图在大图的左上点坐标subimage_coordinate
    '''
    def __init__(self,subimages, transforms):
        self.subimages = subimages
        self.transforms = transforms

        # load all image files, sorting them to
        # ensure that they are aligned
        self.subimage_coordinates = list(subimages.keys())

    def __getitem__(self, idx):
        subimage_coordinate = self.subimage_coordinates[idx]

        img = self.subimages[subimage_coordinate]

        if self.transforms is not None:
            img, subimage_coordinate = self.transforms(img, subimage_coordinate)

        return img, subimage_coordinate

    def __len__(self):
        return len(self.subimages)



def get_bandmax_min(data):
    bandmax = []
    bandmin = []
    width = data.RasterXSize
    height = data.RasterYSize
    deta_width = width//32
    deta_height = height//32
    for n in range(data.RasterCount):
        bandmax.append(0)
        bandmin.append(70000)
        band1 = data.GetRasterBand(n+1)
        for i in range(31):
            img_r = band1.ReadAsArray(deta_width*i,deta_height*i,32,32)
            bandmax[n] = max(bandmax[n],img_r.max())
            bandmin[n] = min(bandmin[n],img_r.min())

        img_r = band1.ReadAsArray(deta_width*31,deta_height*31,width-31*deta_width,height-31*deta_height)
        bandmax[n] = max(bandmax[n],img_r.max())
        bandmin[n] = min(bandmin[n],img_r.min())
    return bandmax,bandmin
def get_subimg(data,bandmax,bandmin,subimage_coordinate,end_coordinate):

    start_x, start_y = subimage_coordinate
    end_x, end_y = end_coordinate
    subsizex=int(end_x-start_x)
    subsizey=int(end_y-start_y)
    img_rgb = []
    img_bgr = []
    if data.RasterCount==4:
        #高分2多光谱，4bands，[b,g,r,Nr]
        band1 = data.GetRasterBand(3)
        img_r = band1.ReadAsArray(int(start_x),int(start_y),int(subsizex),int(subsizey))
        if bandmax[2]>255 or bandmin[2]<0:
            img_r = (img_r-bandmin[2])/(bandmax[2]-bandmin[2])
            img_r = np.round(img_r*255)
            img_r = np.uint8(img_r)

        band2 = data.GetRasterBand(2)
        img_g = band2.ReadAsArray(int(start_x),int(start_y),int(subsizex),int(subsizey))
        if bandmax[1]>255 or bandmin[1]<0:
            img_g = (img_g-bandmin[1])/(bandmax[1]-bandmin[1])
            img_g = np.round(img_g*255)
            img_g = np.uint8(img_g)

        band3 = data.GetRasterBand(1)
        img_b = band3.ReadAsArray(int(start_x),int(start_y),int(subsizex),int(subsizey))
        if bandmax[0]>255 or bandmin[0]<0:
            img_b = (img_b-bandmin[0])/(bandmax[0]-bandmin[0])
            img_b = np.round(img_b*255)
            img_b = np.uint8(img_b)
        img_rgb = cv2.merge([img_r, img_g, img_b])
        img_bgr = cv2.merge([img_b, img_g, img_r])

    elif data.RasterCount==3:
        #高分1三通道图
        band1 = data.GetRasterBand(1)
        img_r = band1.ReadAsArray(int(start_x),int(start_y),int(subsizex),int(subsizey))
        if bandmax[0]>255 or bandmin[0]<0:
            img_r = (img_r-bandmin[0])/(bandmax[0]-bandmin[0])
            img_r = np.round(img_r*255)
            img_r = np.uint8(img_r)

        band2 = data.GetRasterBand(2)
        img_g = band2.ReadAsArray(int(start_x),int(start_y),int(subsizex),int(subsizey))
        if bandmax[1]>255 or bandmin[1]<0:
            img_g = (img_g-bandmin[1])/(bandmax[1]-bandmin[1])
            img_g = np.round(img_g*255)
            img_g = np.uint8(img_g)

        band3 = data.GetRasterBand(3)
        img_b = band3.ReadAsArray(int(start_x),int(start_y),int(subsizex),int(subsizey))
        if bandmax[2]>255 or bandmin[2]<0:
            img_b = (img_b-bandmin[2])/(bandmax[2]-bandmin[2])
            img_b = np.round(img_b*255)
            img_b = np.uint8(img_b)

        img_rgb = cv2.merge([img_r, img_g, img_b])
        img_bgr = cv2.merge([img_b, img_g, img_r])

    elif data.RasterCount==1:
        band1 = data.GetRasterBand(1)
        img_arr = band1.ReadAsArray(int(start_x),int(start_y),int(subsizex),int(subsizey))
        if bandmax[0]>255 or bandmin[0]<0:
            img_arr = (img_arr-bandmin[0])/(bandmax[0]-bandmin[0])
            img_arr = np.uint8(np.round(img_arr*255))
        else:
            img_arr = np.uint8(img_arr)
        img_rgb = cv2.merge([img_arr,img_arr,img_arr])
        img_bgr = cv2.merge([img_arr,img_arr,img_arr])

    return img_rgb, img_bgr