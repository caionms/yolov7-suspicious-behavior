from utils.datasets import letterbox
import torch
from torchvision import transforms
import numpy as np
import cv2


def bbox_iou_vehicle(box1, box2):
    """
    Calcula o Índice de sobreposição de Jaccard (IoU) entre duas caixas delimitadoras.

    Parâmetros:
    box1: list[float]
        Lista contendo as coordenadas [x1, y1, x2, y2] da primeira caixa delimitadora.
    box2: list[float]
        Lista contendo as coordenadas [x1, y1, x2, y2] da segunda caixa delimitadora.

    Retorna:
    float
        O valor do Índice de sobreposição de Jaccard (IoU) entre as duas caixas delimitadoras.
    """
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Coordenadas da intersecção
    x1_intersection = max(x1_box1, x1_box2)
    y1_intersection = max(y1_box1, y1_box2)
    x2_intersection = min(x2_box1, x2_box2)
    y2_intersection = min(y2_box1, y2_box2)

    # Área da intersecção
    intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)

    # Áreas das caixas delimitadoras
    box1_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    box2_area = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)

    # União das áreas
    union_area = box1_area + box2_area - intersection_area

    # Cálculo do IoU
    iou = intersection_area / union_area

    return iou

def load_model(device):
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()
    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model

def run_inference(image, model, device):
    if(np.isscalar(image)): # se image não é um tensor
        # Apply transforms
        image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
        if torch.cuda.is_available():
            image = image.half().to(device)
            # Turn image into batch
            image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
      output, _ = model(image)
    return output, image

def plot_skeleton_kpts(im, xywh_person, conf_person, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    #Conexões entre keypoints
    skeleton = [
                #Rosto 
                [1, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7],
                #Braços
                [6, 7], [6, 8], [7, 9],  [8, 10], [9, 11], 
                #Tronco
                [7, 13], [6, 12], [12, 13],
                #Cintura e pernas
                [14, 12], [15, 13], [16, 14], [17, 15]
      ]

    #Cores para as linhas (seguindo a ordem de skeleton)
    pose_limb_color = palette[[16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9]]
    #Cores para keypoints (seguindo a ordem definida)
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps
    is_suspect = False #Condição para pintar de suspeito
    r, g, b = 0, 0, 255 #RED - Ordem inversa

    ##Calculate if squat
    #if(is_squat_v4(kpts, steps)):
    #  is_suspect = True
    #  plot_text_box(im, int(80), int(80), "Agachado")

    #Plot keypoints
    for kid in range(num_kpts):
        if(not is_suspect):
          r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        #print('Keypoint ' + str(kid) + ': x - ' + str(x_coord) + ' / y - ' +  str(y_coord))
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2] # acima de 0.5 para considerar o ponto correto
                if conf < 0.5:
                    continue
            if(kid == 61 or kid == 122 or kid == 113): # condição para definir quais pontos vão ser desenhados
              r = g = b = 255
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
            #plot_number(im, int(x_coord), int(y_coord), (int(r), int(g), int(b)), str(kid))

    #Plot lines
    for sk_id, sk in enumerate(skeleton):
        if(not is_suspect):
          r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
        
def scale_keypoints_kpts(img1_shape, keypoints, img0_shape, ratio_pad=None):
    # Rescale coords of keypoints [x,y] from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    # [x1,y1,conf1,x2,y2,conf2,...,x17,y7,conf17]
    for i in range(17):
      keypoints[(3*i)] -= pad[0]  # x padding
      keypoints[(3*i)+1] -= pad[1]  # y padding
    
    for i in range(17):
      keypoints[(3*i)] /= gain  # x padding
      keypoints[(3*i)+1] /= gain  # y padding

    #chama o método de clip
    tensor = clip_keypoints_kpts(keypoints, img0_shape)
    return tensor.detach().numpy()