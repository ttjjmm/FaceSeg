import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from torch2trt import TRTModule
import onnxruntime
import torch

from model import FCN


curPath = os.path.abspath(os.path.dirname(__file__))


class FaceSegBase(metaclass=ABCMeta):
    def __init__(self):
        self.seg = None

    @staticmethod
    def input_transform(image):
        image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
        image = (image / 255.)[np.newaxis, :, :, :]
        image_input = np.transpose(image, (0, 3, 1, 2)).astype(np.float32)
        # image_input = torch.from_numpy(image)
        return image_input

    @staticmethod
    def output_transform(output, shape):
        output = cv2.resize(output, (shape[1], shape[0]))
        image_output = (output * 255).astype(np.uint8)
        return image_output

    @abstractmethod
    def get_mask(self, image) -> np.array:
        pass



class FaceSeg_Torch(FaceSegBase):
    def __init__(self, model_path='/home/ubuntu/Documents/pycharm/FaceSeg/weights/seg_model_384.pt'):
        super(FaceSeg_Torch, self).__init__()
        self.seg = FCN(num_classes=2, backbone='HRNet_W18').to('cuda:0')
        para_state_dict = torch.load(model_path)
        self.seg.load_state_dict(para_state_dict)
        self.seg.eval()

    def get_mask(self, image):
        image_input = torch.from_numpy(self.input_transform(image))
        image_input = image_input.to('cuda:0')
        t0 = time.time()
        with torch.no_grad():
            logits = self.seg(image_input)
        print("Infer time: {:.4f}s".format(time.time() - t0))
        pred = torch.argmax(logits[0], dim=1)
        pred = pred.cpu().numpy()
        mask = np.squeeze(pred).astype('uint8')
        mask = self.output_transform(mask, shape=image.shape[:2])
        return mask



class FaceSeg_ONNX(FaceSegBase):
    """
    GPU enable
    """
    def __init__(self, model_path='/home/ubuntu/Documents/pycharm/FaceSeg/onnx/seg_model_384_sim.onnx'):
        super(FaceSeg_ONNX, self).__init__()
        self.seg = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    def get_mask(self, image):
        image_input = self.input_transform(image)
        t0 = time.time()
        pred = self.seg.run(['mask'], input_feed={'input0': image_input})
        print("Infer time: {:.4f}s".format(time.time() - t0))
        pred = np.argmax(pred[0], axis=1)
        mask = np.squeeze(pred).astype('uint8')
        mask = self.output_transform(mask, shape=image.shape[:2])
        return mask



class FaceSeg_TRT(FaceSegBase):
    def __init__(self, model_path='/home/ubuntu/Documents/pycharm/FaceSeg/weights/seg_model_384.pt'):
        super(FaceSeg_TRT, self).__init__()
        self.seg = TRTModule()
        para_state_dict = torch.load(model_path)
        self.seg.load_state_dict(para_state_dict)

    def get_mask(self, image):
        pass



if __name__ == '__main__':
    seg = FaceSeg_Torch()
    img = cv2.imread('/home/ubuntu/Documents/pycharm/FaceSeg/samples/photo_test.jpg')
    mask = seg.get_mask(img)
    # print(mask.max(), mask.min())
    plt.imshow(mask)
    plt.show()


