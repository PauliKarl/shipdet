from lxml import etree as ET
import numpy as np
def simple_obb_xml_dump(objects, img_name, save_dir):
    bboxes, rbboxes, pointobbs, labels, scores, areas, number = [], [], [], [], [], [], 0
    for obj in objects:
        bboxes.append(obj['bbox'])
        rbboxes.append(obj['rbbox'])
        pointobbs.append(obj['pointobbs'])
        labels.append(obj['label'])
        scores.append(obj['score'])
        areas.append(obj['rbbox'][2] * obj['rbbox'][3])
        number += 1

    root=ET.Element("annotations")
    ET.SubElement(root, "filename").text = img_name
    ET.SubElement(root,"number").text = str(number)
    
    for idx in range(number):
        object=ET.SubElement(root, "object")

        ET.SubElement(object, "object_crop").text = "ObjectID_{}.jpg".format(idx+1)

        ET.SubElement(object, "det_id").text = str(idx+1)

        #写入检测框的位置信息
        ET.SubElement(object,"name").text = " "
        ET.SubElement(object,"className").text = str(labels[idx])
        ET.SubElement(object,"boxType").text = "robndbox"
        ET.SubElement(object,"area").text = str(areas[idx])
        ET.SubElement(object,"score").text = str(scores[idx])
        robndbox=ET.SubElement(object,"robndbox")
        #写入矩形框，角度为[-np.pi, np.pi)即长边与水平方向夹角、上负下正, 矩形长边为h, 短边为w
        if rbboxes[idx][2]>rbboxes[idx][3]:
            w = rbboxes[idx][3]
            h = rbboxes[idx][2]
            theta = rbboxes[idx][4]
        else:
            w = rbboxes[idx][2]
            h = rbboxes[idx][3]
            theta = np.pi/2.0 + rbboxes[idx][4]
        ET.SubElement(robndbox,"cx").text = str(rbboxes[idx][0])
        ET.SubElement(robndbox,"cy").text = str(rbboxes[idx][1])
        ET.SubElement(robndbox,"w").text = str(w)
        ET.SubElement(robndbox,"h").text = str(h)
        ET.SubElement(robndbox,"angle").text = str(theta)

        pobndbox=ET.SubElement(object,"point4")
        ET.SubElement(pobndbox,"x1").text = str(pointobbs[idx][0])
        ET.SubElement(pobndbox,"y1").text = str(pointobbs[idx][1])
        ET.SubElement(pobndbox,"x2").text = str(pointobbs[idx][2])
        ET.SubElement(pobndbox,"y2").text = str(pointobbs[idx][3])
        ET.SubElement(pobndbox,"x3").text = str(pointobbs[idx][4])
        ET.SubElement(pobndbox,"y3").text = str(pointobbs[idx][5])
        ET.SubElement(pobndbox,"x4").text = str(pointobbs[idx][6])
        ET.SubElement(pobndbox,"y4").text = str(pointobbs[idx][7])
        
    tree = ET.ElementTree(root)
    tree.write("{}/{}_detection.xml".format(save_dir, img_name), pretty_print=True, xml_declaration=True, encoding='utf-8')