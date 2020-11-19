from shipdet.model.faster_rcnn import FastRCNNPredictor
from shipdet.model.mask_rcnn import MaskRCNNPredictor
import shipdet.model.mask_rcnn

model_name={
    'resnet50': 'maskrcnn_resnet50_fpn',
    'resnet101': 'maskrcnn_resnet101_fpn',
    'resnet152': 'maskrcnn_resnet152_fpn',
    'resnext50_32x4d': 'maskrcnn_resnext50_32x4d_fpn',
    'resnext101_32x8d': 'maskrcnn_resnext101_32x8d_fpn',
}

def get_model_object_detection(num_classes, model_dir = None, backbone_name = 'resnet50',**kwargs):
    # load an instance segmentation model pre-trained pre-trained on COCO
    # "model_dir" is the path to save the pretrained model, if there is no any file, it will be downloaded from web
    
    model = shipdet.model.mask_rcnn.__dict__[model_name[backbone_name]](
                                                            pretrained=False, 
                                                            num_classes=num_classes,
                                                            model_dir=model_dir,
                                                            **kwargs)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # now get the number of inputfeatures for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model
