from lxml import etree as ET


def parse_params_xmlfile(params_xml_file):

    parameter = dict()
    tree = ET.parse(params_xml_file)
    root = tree.getroot()
    global_parameter = root.find('global')
    parameter['rpn_nms_thresh'] = float(global_parameter.find('rpn_nms_thresh').text)
    parameter['rpn_fg_iou_thresh'] = float(global_parameter.find('rpn_fg_iou_thresh').text)
    parameter['rpn_bg_iou_thresh'] = float(global_parameter.find('rpn_bg_iou_thresh').text)
    parameter['box_nms_thresh'] = float(global_parameter.find('box_nms_thresh').text)
    parameter['box_fg_iou_thresh'] = float(global_parameter.find('box_fg_iou_thresh').text)
    parameter['box_bg_iou_thresh'] = float(global_parameter.find('box_bg_iou_thresh').text)

    input_parameter = root.find('train')
    parameter['batchsize'] = int(input_parameter.find('batchsize').text)
    parameter['num_workers'] = int(input_parameter.find('num_workers').text)
    parameter['learning_rate'] = float(input_parameter.find('learning_rate').text)
    parameter['backbone'] = input_parameter.find('backbone').text
    parameter['test_dir'] = input_parameter.find('test_dir').text
    parameter['resume_from'] = input_parameter.find('resume_from').text
    parameter['test_ratio'] = float(input_parameter.find('test_ratio').text) if input_parameter.find('test_ratio').text else None
    parameter['save_log_path'] = input_parameter.find('save_log_path').text
    return parameter


def parse_params_xmlfile_test(params_xml_file):

    parameter = dict()
    from lxml import etree as ET

    tree = ET.parse(params_xml_file)
    root = tree.getroot()

    global_parameter = root.find('global')
    parameter['rpn_nms_thresh'] = float(global_parameter.find('rpn_nms_thresh').text)
    parameter['rpn_fg_iou_thresh'] = float(global_parameter.find('rpn_fg_iou_thresh').text)
    parameter['rpn_bg_iou_thresh'] = float(global_parameter.find('rpn_bg_iou_thresh').text)
    parameter['box_nms_thresh'] = float(global_parameter.find('box_nms_thresh').text)
    parameter['box_fg_iou_thresh'] = float(global_parameter.find('box_fg_iou_thresh').text)
    parameter['box_bg_iou_thresh'] = float(global_parameter.find('box_bg_iou_thresh').text)

    input_parameter = root.find('test')
    parameter['batchsize'] = int(input_parameter.find('batchsize').text)
    parameter['num_workers'] = int(input_parameter.find('num_workers').text)
    parameter['gap'] = int(input_parameter.find('gap').text)
    parameter['backbone'] = input_parameter.find('backbone').text
    parameter['mask_threshold'] = float(input_parameter.find('mask_threshold').text)
    parameter['nms_name'] = input_parameter.find('nms_name').text
    parameter['iou_threshold'] = float(input_parameter.find('iou_threshold').text)
    parameter['score_threshold'] = float(input_parameter.find('score_threshold').text)

    return parameter