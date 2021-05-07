import argparse

import torch

from mobilenet_v2_tsm import MobileNetV2


def export_onnx(model, inputs, output_path):
    model.eval()
    input_names = []
    input_shapes = {}
    with torch.no_grad():
        for index, torch_input in enumerate(inputs):
            name = "i" + str(index)
            input_names.append(name)
            input_shapes[name] = torch_input.shape
        torch.onnx.export(model, inputs, output_path, input_names=input_names,
                          output_names=["o" + str(i) for i in range(len(inputs))], opset_version=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--simplify', action='store_true')
    args = parser.parse_args()
    model = MobileNetV2(n_class=27)
    model.load_state_dict(torch.load("mobilenetv2_jester_online.pth.tar"))
    inputs = (torch.ones(1, 3, 224, 224),
              torch.zeros([1, 3, 56, 56]),
              torch.zeros([1, 4, 28, 28]),
              torch.zeros([1, 4, 28, 28]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 8, 14, 14]),
              torch.zeros([1, 12, 14, 14]),
              torch.zeros([1, 12, 14, 14]),
              torch.zeros([1, 20, 7, 7]),
              torch.zeros([1, 20, 7, 7]))
    export_onnx(model, inputs, args.output_path)