from torch2trt import torch2trt
import torch
import argparse
from model import FCN


parser = argparse.ArgumentParser(description='export')
parser.add_argument('--m_type', type=str, default='HRNet_W18',
                    help='config file path')
parser.add_argument('--ckpt', type=str, default='weights/seg_model_384.pt',
                    help='checkpoint file path')
parser.add_argument('--output', type=str, default='weights/seg_model_384_trt.onnx',
                    help='output onnx file path')
parser.add_argument('--out_size', type=tuple, default=(384, 384),
                    help='output image size')
args = parser.parse_args()
print(args)


def export_trt(args):
    model = FCN(num_classes=2, backbone=args.m_type)
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.eval().to('cuda:0')
    x = torch.randn((1, 3, args.out_size[0], args.out_size[1])).to('cuda:0')
    model_trt = torch2trt(model, [x], fp16_mode=True)






if __name__ == '__main__':
    export_trt(args)

