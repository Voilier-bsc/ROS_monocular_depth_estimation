#!/usr/bin/env python
# license removed for brevity

import os
import time
import numpy as np
import sys
import types
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

args = types.SimpleNamespace()
args.model_name         = 'bts_eigen_v2_pytorch_densenet121'
args.encoder            = 'densenet121_bts'
args.checkpoint_path    = '../bts_eigen_v2_pytorch_densenet121/model'
args.max_depth          = 80
args.input_height       = 480
args.input_width        = 640
args.do_kb_crop         = False
args.bts_size           = 512
args.dataset            = 'kitti'

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val

model = BtsModel(params=args)
model = torch.nn.DataParallel(model)

checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint['model'])
model.eval()
model.cuda()


def preprocessing_transforms():
    return transforms.Compose([ToTensor()])


class BtsDataLoader(object):
    def __init__(self, img_array):
        self.testing_samples = DataLoadPreprocess(img_array, transform=preprocessing_transforms())
        self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

            
class DataLoadPreprocess(Dataset):
    def __init__(self, img_array, transform=None):
        self.transform = transform
        self.to_tensor = ToTensor
        self.img_raw = img_array
    
    def __getitem__(self, idx):

        focal = 721.5377
        image = np.asarray(self.img_raw, dtype=np.float32) / 255.0 ##normalize

        sample = {'image': image, 'focal': focal}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return 1
        # return len(self.filenames)


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        return {'image': image, 'focal': focal}

    
    def to_tensor(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img


def inference(frame):
    # start_time = time.time()
    dataloader = BtsDataLoader(frame)
    with torch.no_grad():
        for _, sample in enumerate(dataloader.data):
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())

            # Predict
            _, _, _, _, depth_est = model(image, focal)
            depth_est = depth_est.cpu().numpy().squeeze()
    pred_depth_scaled = depth_est * 1000
    pred_depth_scaled = pred_depth_scaled.astype(np.uint16)

    rospy.loginfo("Running test")
    # elapsed_time = time.time() - start_time
    # print('Elapesed time: %s' % str(elapsed_time))

    return pred_depth_scaled


def Image_to_opencv(msg):

    torch.cuda.empty_cache()
    cvb=CvBridge()
    cv_image = cvb.imgmsg_to_cv2(msg,"bgr8")

    inference_img = inference(cv_image)
    img_pub = rospy.Publisher("bts_inference_img", Image, queue_size=1)
    img_pub.publish(cvb.cv2_to_imgmsg(inference_img, "mono16"))
    rospy.loginfo("complete pub")


if __name__ == '__main__':
    
    rospy.init_node("ROS_monocular_depth_estimation", anonymous=True)
    rospy.loginfo("Running bts_cam_inference")
    rospy.Subscriber("/usb_cam/image_raw", Image, Image_to_opencv, queue_size=1)

    rate=rospy.Rate(0.5)
    rate.sleep()
    
    rospy.spin()

