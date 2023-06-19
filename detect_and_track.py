import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, strip_optimizer, set_logging, increment_path, non_max_suppression_kpt
from utils.plots import plot_one_box, draw_boxes, output_to_keypoint
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.download_weights import download

from utils.keypoints_utils import bbox_iou_vehicle, load_model, run_inference, plot_skeleton_kpts, scale_keypoints_kpts, xywh2xyxy_personalizado, scale_coords_kpts

#For SORT tracking
import skimage
from sort import *

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, keypoints = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.keypoints
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
        #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    
    
    #........Rand Color for every trk.......
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................


    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16
        
    model_kpts = None

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    
    # Criar um dicionário vazio para armazenar os tempos
    tempos = {} 

    t0 = time.time()
    
    #im0s é o array e img é o tensor
    for path, img, im0s, vid_cap in dataset:
        fps = None
        if vid_cap is not None:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Saves the boxes of vehicles
        # Usado para calcular IoU
        vehicles_objs = []
        persons_objs = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                 #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                #6 para apenas deteccao / 7 para deteccao com keypoints
                dets_to_sort = np.empty((0,6)) if keypoints == 0 else np.empty((0,7))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    if detclass == 0:
                        persons_objs.append([x1, y1, x2, y2, conf, detclass])
                    elif detclass == 2 or detclass == 3: #adiciona os veiculos na lista
                        vehicles_objs.append([x1,y1,x2,y2])
                        
                #vetor auxiliar que guarda as pessoas que sofreram interseção
                test_squat = []
                        
                # chamada para calcular interseção
                for aux, person_box in enumerate(persons_objs): 
                    for vehicle_box in vehicles_objs:
                        if bbox_iou_vehicle(vehicle_box, person_box[:4]) > 0:
                            if(keypoints == 1):
                                test_squat.append(person_box)
                            else:
                                dets_to_sort = np.vstack((dets_to_sort, person_box.copy()))
                            # Remover person_box de persons_objs
                            persons_objs.pop(aux)
                            break
                        
                #se ocorreu interseccao
                if(test_squat and not np.any(dets_to_sort)):
                    #carrega o modelo de keypoints
                    if model_kpts == None: 
                        model_kpts = load_model(device)
                    #faz deteccao de keypoints - utilizar o img
                    output, nimg = run_inference(img, model_kpts,device)
                    output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model_kpts.yaml['nc'], # Number of Classes
                                     nkpt=model_kpts.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
                    with torch.no_grad():
                            output = output_to_keypoint(output)
                            
                    '''
                    nimg = nimg[0].permute(1, 2, 0) * 255
                    nimg = nimg.cpu().numpy().astype(np.uint8)
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
                    '''
                    #dicionario auxiliar para acesso de keypoints
                    dic = {}

                    for idx in range(output.shape[0]):
                        class_id = output[idx, 1]
                        if class_id == 0:
                            x = output[idx, 2]
                            y = output[idx, 3]
                            w = output[idx, 4]
                            h = output[idx, 5]
                            conf = output[idx, 6]
                            kpts = scale_keypoints_kpts(nimg.shape[2:], output[idx, 7:].T, im0.shape).round()
                            #guarda valores para tracking
                            x1,y1,x2,y2 = xywh2xyxy_personalizado([x, y, w, h])
                            #ajusta a escala da bbox
                            [x1,y1,x2,y2] = scale_coords_kpts(img.shape[2:], [x1,y1,x2,y2], im0.shape).round()
                            #guarda os keypoints no dicionário
                            dic[idx] = keypoints
                            #guarda as detecções de pessoas para o tracker
                            dets_to_sort = np.vstack((dets_to_sort, 
                                    np.array([x1, y1, x2, y2, idx, conf, class_id])))
                            
                            #faz desenho dos esqueletos - usar o im0
                            plot_skeleton_kpts(im0, [x,y,w,h], conf, kpts, 3)
                        
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort) if keypoints == 0 else sort_tracker.update_kpts(dets_to_sort)
                
                #tracks = sort_tracker.getTrackers()
                
                # draw boxes of tracked person
                if len(tracked_dets)>0:
                    print('tracked dets')
                    print(tracked_dets)
                    bbox_xyxy = tracked_dets[:,:4]
                    categories = tracked_dets[:, 4]
                    kpts_idxs = tracked_dets[:, 5]
                    identities = tracked_dets[:, 6]
                    #draw_boxes(im0, bbox_xyxy, vehicles_objs, tempos, fps, identities, categories, names, txt_path)
                    
                # draw boxes of non tracked person
                for person in persons_objs:
                    if save_img or view_img:  # Add bbox to image
                        label = 'pessoa'
                        plot_one_box(person[:4], im0, label=label, color=colors[int(person[-1])], line_thickness=1)
                
                # draw boxes of behicles
                for vehicle in vehicles_objs:
                    if save_img or view_img:  # Add bbox to image
                        label = 'veiculo'
                        plot_one_box(vehicle, im0, label=label, color=colors[2], line_thickness=1)

                '''
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                '''

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--keypoints', type=int, default=0, help='keypoints off or on, i.e. 0 or 1')
    
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
