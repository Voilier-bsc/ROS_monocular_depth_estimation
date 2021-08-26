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
from cv_bridge import CvBridge
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
import random

sys.path.insert(0, '/home/cordin/catkin_ws/src/ROS_monocular_depth_estimation')
sys.path.insert(0, '/home/cordin/catkin_ws/src/ROS_monocular_depth_estimation/bts_eigen_v2_pytorch_densenet121')

from yolo.util import *
from yolo.darknet import Darknet
import pickle as pkl



args = types.SimpleNamespace()
args.model_name         = 'bts_eigen_v2_pytorch_densenet121'
args.encoder            = 'densenet121_bts'
args.checkpoint_path    = '/home/cordin/catkin_ws/src/ROS_monocular_depth_estimation/bts_eigen_v2_pytorch_densenet121/model'
args.max_depth          = 80
args.input_height       = 480
args.input_width        = 640
args.do_kb_crop         = True
args.bts_size           = 512
args.dataset            = 'kitti'

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val

model = BtsModel(params=args)
model = torch.nn.DataParallel(model,device_ids=[0])

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
    
    dataloader = BtsDataLoader(frame)
    with torch.no_grad():
        for _, sample in enumerate(dataloader.data):
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())

            # Predict
            _, _, _, _, depth_est = model(image, focal)
            depth_est = depth_est.cpu().numpy().squeeze()
    depth_img = depth_est * 1000
    depth_img = depth_img.astype(np.uint16)
    pred_depth_scaled = depth_est
    pred_depth_scaled = pred_depth_scaled.astype(np.uint16)

    return pred_depth_scaled , depth_img

class yolo():
    def __init__(self):
        self.config()
        self.load_network()
    
    def load_network(self, ):
        model = Darknet(self.cfgfile)
        model.load_weights(self.weightsfile)
        model.net_info["height"] = self.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32
        model = model.to(self.device)
        model.eval()

        self.model = model
        self.inp_dim = inp_dim

    def config(self):
        self.confidence = float(0.5)
        self.nms_thresh = float(0.4)
        self.cfgfile = "/home/cordin/catkin_ws/src/ROS_monocular_depth_estimation/yolo/data/yolov3.cfg"
        self.weightsfile = "/home/cordin/catkin_ws/src/ROS_monocular_depth_estimation/yolo/data/yolov3.weights"
        self.reso = "416"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 80
        self.classes = self.load_classes("/home/cordin/catkin_ws/src/ROS_monocular_depth_estimation/yolo/data/coco.names")
        self.colors = pkl.load(open("/home/cordin/catkin_ws/src/ROS_monocular_depth_estimation/yolo/data/pallete", "rb"))

    def load_classes(self, namesfile):
        fp = open(namesfile, "r")
        names = fp.read().split("\n")[:-1]
        return names

    def prep_image(self, img, inp_dim):
        """
        prepare image for inputting to the neural network
        Returns a variable
        """
        img = self.letterbox_image(img, inp_dim)
        # img = cv2.resize(img, (inp_dim, inp_dim)) # 또 리사이즈를..?
        img = (
            img[:, :, ::-1].transpose((2, 0, 1)).copy()
        )  # channel을 역순으로 channel먼저로 바꿔줌 BGR -> RGB
        img = (
            torch.from_numpy(img).float().div(255.0).unsqueeze(0)
        )  # normalization, 255.0(pixel intensity)
        return img      

    def letterbox_image(self, img, inp_dim):
        """ resize image with unchaged aspect ratio using padding """
        img_w, img_h = img.shape[1], img.shape[0]
        # w, h = inp_dim
        w = self.inp_dim
        h = self.inp_dim
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # canvas = np.full((inp_dim[1], inp_dim[0], 3), 128) #return a new img h x w with color 128
        canvas = np.full((w, h, 3), 128)
        canvas[
            (h - new_h) // 2 : (h - new_h) // 2 + new_h,
            (w - new_w) // 2 : (w - new_w) // 2 + new_w,
            :,
        ] = resized_image
        return canvas

    def infer(self, data, depth_map):
        with torch.no_grad():
            input = self.prep_image(data, self.inp_dim)
            input = input.to(self.device)
            im_dim = (data.shape[1], data.shape[0])
            im_dim = torch.FloatTensor(im_dim).repeat(1,2) # tensor([[480., 640., 480., 640.]])
            im_dim=im_dim.to(self.device)

            prediction = self.model(input.to(self.device), True)
            output = write_results(prediction, self.confidence, self.num_classes, nms_conf=self.nms_thresh)

            try: 
                output
            except NameError:
                print("No detections were made")
                exit()
            
            output = self.edit_bbox(output, im_dim) # [obj_num, x, y, w, h, o, conf_score, class] in list
            
            for i in output:
                center = ((i[2]+i[4])/2).cpu().int(), ((i[1]+i[3])/2).cpu().int()
                dist = depth_map[center]
                self.write(i, data, random.choice(self.colors), dist, center)
                
        return data 

    def edit_bbox(self, output, im_dim):
                # transform the corner attributes of each bounding box, to the original dim.
        im_dim = torch.index_select(
            im_dim, 0, output[:, 0].long()
        )  # row 에 대해서 output[:,0].long() 인덱스만 셀렉트

        scaling_factor = torch.min(self.inp_dim / im_dim, 1)[0].view(-1, 1)  # dim = 1 에 대해서 min
        output[:, [1, 3]] -= (
            self.inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)
        ) / 2  # x, w 땡겨주기

        output[:, [2, 4]] -= (
            self.inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)
        ) / 2  # y, h 땡겨주기
        output[:, 1:5] /= scaling_factor  # x, y, w, h scaling

        # clip any bounding box outside of the image to the edges.
        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])  # (input, min, max)
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        return output

    def write(self, x, img, color, dist, center):
        c1 = tuple(x[1:3])
        c2 = tuple(x[3:5])
        cls = int(x[-1])
        label = "{0}".format(self.classes[cls])
        cv2.rectangle(img, (int(x[1]),int(x[2])),(int(x[3]),int(x[4])), color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        c_1 = x[1] + t_size[0] + 3
        c_2 = x[2] + t_size[1] + 4
        cv2.rectangle(img,  (int(x[1]),int(x[2])),(int(c_1),int(c_2)), color, -1)
        cv2.putText(
            img,
            label,
            (int(x[1]), int(x[2]) + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            [225, 255, 255],
            1,
        )
        cv2.circle(img, (int(center[1]), int(center[0])), 2, color, -1)
        cv2.putText(img, "depth: {:.2f} m".format(dist), (int(center[1]), int(center[0])+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

def Image_to_opencv(msg):
    
    torch.cuda.empty_cache()
    cvb=CvBridge()
    cv_image = cvb.imgmsg_to_cv2(msg,"bgr8")

    input_img = cv_image.copy()

    start_time = time.time()

    real_depth, depth_img = inference(input_img)

    elapsed_time = time.time() - start_time
    yolo_start_time = time.time()
    print('bts inference time: %s' % str(elapsed_time))

    img_yolo = yolo.infer(input_img, real_depth)
    elapsed_time2 = time.time() - yolo_start_time
    print('yolo inference time: %s' % str(elapsed_time))

    img_pub1 = rospy.Publisher("bts_depth_img", Image, queue_size=1)
    img_pub1.publish(cvb.cv2_to_imgmsg(depth_img, "mono16"))

    img_pub2 = rospy.Publisher("bts_yolo_img", Image, queue_size=1)
    img_pub2.publish(cvb.cv2_to_imgmsg(img_yolo, "bgr8"))

    img_pub3 = rospy.Publisher("bts_original_img", Image, queue_size=1)
    img_pub3.publish(cvb.cv2_to_imgmsg(cv_image, "bgr8"))



if __name__ == '__main__':
    try:
        yolo = yolo()
        rospy.init_node("ROS_monocular_depth_estimation", anonymous=True)
        rospy.loginfo("Running bts_cam_inference")
        rospy.Subscriber("/usb_cam/image_raw", Image, Image_to_opencv, queue_size=1)

        rate=rospy.Rate(0.5)
        rate.sleep()
        
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.logerr('NO!')
