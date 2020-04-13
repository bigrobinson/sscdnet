from argparse import ArgumentParser
import cv2
import numpy as np
import torch
from torch.autograd import Variable
#import torch.nn.functional as F
import sys
sys.path.append("./correlation_package/build/lib.linux-x86_64-3.6")
import os
import cscdnet

def colormap():
    cmap=np.zeros([2, 3]).astype(np.uint8)

    cmap[0,:] = np.array([0, 0, 0])
    cmap[1,:] = np.array([255, 255, 255])

    return cmap

class Colorization:

    def __init__(self, n=2):
        self.cmap = colormap()
        self.cmap = torch.from_numpy(np.array(self.cmap[:n]))

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

class ChangeDetect:

    def __init__(self, img0_path, img1_path, out_dir, model_path, use_corr=True):
        
        self.img0_path = img0_path
        self.img1_path = img1_path
        self.use_corr = use_corr
        self.out_dir = out_dir
        
    
    def preprocess_image(self):
           
        if os.path.isfile(self.img0_path) == False:
            print ('Error: File Not Found: ' + self.img0_path)
            exit(-1)
        if os.path.isfile(self.img1_path) == False:
            print ('Error: File Not Found: ' + self.img1_path)
            exit(-1)

        img0 = cv2.imread(self.img0_path, cv2.IMREAD_COLOR)
        img1 = cv2.imread(self.img1_path, cv2.IMREAD_COLOR)
        
        # Images must be 256 x 256       
        img0_ = np.asarray(img0).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
        img1_ = np.asarray(img1).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
        
        #### TEMP #########
        img0_ = img0_[:,1100:1100+256, 2090:2090+256]
        img1_ = img1_[:,1100:1100+256, 2090:2090+256]
        ###################
        
        input_ = torch.from_numpy(np.concatenate((img0_, img1_), axis=0))
        
        print(input_.shape) # returns torch.size([6, 256, 256])

        input_ = input_[np.newaxis, :, :]
        input_ = input_.cuda()
        input_ = Variable(input_)
        
        return input_
    
    
    def run_inference(self, input_, model_path):
        
        self.color_transform = Colorization(2)
        
        # pretrained keyword refers to resnet feature detector being pretrained
        if self.use_corr:
            print('Correlated Siamese Change Detection Network (CSCDNet)')
            self.model = cscdnet.Model(inc=6, outc=2, corr=True, pretrained=True)

        else:
            print('Siamese Change Detection Network (Siamese CDResNet)')
            self.model = cscdnet.Model(inc=6, outc=2, corr=False, pretrained=True)

        if os.path.isfile(model_path) is False:
            print("Error: Cannot read file ... " + model_path)
            exit(-1)
        else:
            print("Reading model ... " + model_path)
        
        # Load trained model (from dataparallel module if necessary)
        state_dict = torch.load(model_path)
        first_pair = next(iter(state_dict.items()))
        if first_pair[0][:7] == "module.":
            # create new OrderedDict with generic keys
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove "module."
                new_state_dict[name] = v
            # load params
            self.model.load_state_dict(new_state_dict)
        else:       
            self.model.load_state_dict(state_dict)
        
        self.model = self.model.cuda()
        
        output_ = self.model(input_)
  
        inputs = input_[0].cpu().data
        img0 = inputs[0:3, :, :]
        img1 = inputs[3:6, :, :]
        img0 = (img0 + 1.0) * 128
        img1 = (img1 + 1.0) * 128
      
        output = output_[0][np.newaxis, :, :, :]
        output = output[:, 0:2, :, :]
        mask_pred = np.transpose(self.color_transform(output[0].cpu().max(0)[1][np.newaxis, :, :].data).numpy(),
                                 (1, 2, 0)).astype(np.uint8)
                
        img_out = self.display_result(img0, img1, mask_pred)
        
        return mask_pred, img_out
    
    def display_result(self, img0, img1, mask_pred):
        
        rows = cols = 256
        img_out = np.zeros((rows, cols * 3, 3), dtype=np.uint8)
        img_out[0:rows, 0:cols, :] = np.transpose(img0.numpy(), (1, 2, 0)).astype(np.uint8)
        img_out[0:rows, cols:cols * 2, :] = np.transpose(img1.numpy(), (1, 2, 0)).astype(np.uint8)
        img_out[0:rows, cols*2:cols*3, :] = mask_pred

        img_filename, _ = os.path.splitext(os.path.basename(self.img0_path))        
        img_save_path = os.path.join(self.out_dir, '{}.png'.format(img_filename))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        print('Writing ... ' + img_save_path)
        cv2.imwrite(img_save_path, img_out)
        
        return img_out
        
if __name__ == "__main__":
    
    parser = ArgumentParser(description = 'Class to preprocess and perform change detection')
    parser.add_argument('--img0_path', type=str, help='path to first image')
    parser.add_argument('--img1_path', type=str, help='path to second image')
    parser.add_argument('--out_dir' , type=str, help='path to output path')
    parser.add_argument('--model_path', type=str, help='path to trained .pth model')
    parser.add_argument('--use_corr', type=bool, help='use correlation?')
    opt = parser.parse_args()
    
    change_det = ChangeDetect(opt.img0_path, opt.img1_path, opt.out_dir, opt.model_path, opt.use_corr)
    input_ = change_det.preprocess_image()
    mask_pred, img_out = change_det.run_inference(input_, opt.model_path)
    
    