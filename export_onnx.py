import os
import onnx
import torch
import argparse
from model import FCN


parser = argparse.ArgumentParser(description='export')
parser.add_argument('--m_type', type=str, default='HRNet_W18',
                    help='config file path')
parser.add_argument('--ckpt', type=str, default='weights/seg_model_384.pt',
                    help='checkpoint file path')
parser.add_argument('--output', type=str, default='onnx/seg_model_384.onnx',
                    help='output onnx file path')
parser.add_argument('--out_size', type=tuple, default=(384, 384),
                    help='output image size')
parser.add_argument('--simplify', action='store_true', default=True,
                    help='simplify onnx model')
args = parser.parse_args()
print(args)



def export_onnx(args):
    model = FCN(num_classes=2, backbone=args.m_type)
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.eval()
    dummy_input = torch.autograd.Variable(torch.randn(1, 3, args.out_size[0], args.out_size[1]))
    input_names = ['input0']
    output_names = ['mask']
    torch.onnx.export(model,
                      dummy_input,
                      args.output,
                      verbose=True,
                      keep_initializers_as_inputs=True,
                      opset_version=11,
                      input_names=input_names,
                      output_names=output_names)
    print(f'Successfully exported ONNX model: {args.output}')
    if args.simplify:
        try:
            from onnxsim import simplify
        except ImportError:
            raise ImportError('Please run "pip install onnx-simplifier" to install onnxsim pakage!')
        onnx_model = onnx.load(args.output)
        model, check = simplify(onnx_model)
        assert check, 'Simplified ONNX model could not be validated'
        par_dir, file_name = os.path.split(args.output)
        sim_path = os.path.join(par_dir, '{}_sim.onnx'.format(file_name.split('.')[0]))
        onnx.save(model, sim_path)
        print(f'Successfully simplified ONNX model: {sim_path}')


if __name__ == '__main__':
    export_onnx(args)