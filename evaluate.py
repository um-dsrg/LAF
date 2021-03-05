import numpy as np
import torch
import torch.optim
import torch.utils.data
import os
from skimage import io as io #numpy version has to be ==1.15.0
import LAF.models as models

def compute_LAF_confidence(LAF_model_dir,img_path,left_flag):
    '''

    Args:
        LAF_model_dir: the path including LAFNet_mb_left.pth,LAFNet_mb_right.pth
        img_path: the path including EO_img, disparity, cost
        left_flag: # switch for left or right confidence

    Returns: confidence_img

    '''

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(device)

    model = models.LAFNet_CVPR2019().to(device)
    left_LAF_model_file = os.path.join(LAF_model_dir,'LAFNet_left.pth')
    right_LAF_model_file = os.path.join(LAF_model_dir,'LAFNet_right.pth')
    if device=='gpu':
        if left_flag:
            model.load_state_dict(torch.load(left_LAF_model_file))
        else:
            model.load_state_dict(torch.load(right_LAF_model_file))
    else:
        if left_flag:
            model.load_state_dict(torch.load(left_LAF_model_file,map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(right_LAF_model_file,map_location=torch.device('cpu')))
    model.eval()

    if left_flag:
        image_file = os.path.join(img_path,'rectified_ref.tif')
        disp_file = os.path.join(img_path,'disp0MCCNN.npy')
        cost_file = os.path.join(img_path,'cost0MCCNN.npy')
    else:
        image_file = os.path.join(img_path,'rectified_sec.tif')
        disp_file = os.path.join(img_path,'disp1MCCNN.npy')
        cost_file = os.path.join(img_path,'cost1MCCNN.npy')

    with torch.no_grad():

        imagEO = io.imread(image_file).astype(np.float32)
        imagEO[imagEO==np.inf] = np.nan
        imagEO = (imagEO - np.nanmean(imagEO)) / np.nanstd(imagEO)
        imag = np.zeros((imagEO.shape[0],imagEO.shape[1],3),dtype = np.float32)
        imag[:,:,0] = imagEO
        imag[:,:,1] = imagEO
        imag[:,:,2] = imagEO
        imag = torch.Tensor(imag.astype(np.float32))
        imag = imag.transpose(0,1).transpose(0,2)

        ch, hei, wei = imag.size()

        cost = np.load(cost_file)
        cost_d, cost_h, cost_w = cost.shape
        cost = torch.Tensor(cost)
        cost[cost==0] = cost.max()

        disp = np.load(disp_file)
        disp_h,disp_w = disp.shape
        disp = (disp - np.nanmean(disp)) / np.nanstd(disp)
        disp = torch.Tensor(disp.astype(np.float32)).unsqueeze(0)

        input_cost = cost.to(device).unsqueeze(0)
        input_imag = imag.to(device).unsqueeze(0)
        input_disp = disp.to(device).unsqueeze(0)

        # Confidence estimation
        conf = model(input_cost, input_disp, input_imag)

        conf = conf.squeeze().cpu().detach().numpy()

        return conf

