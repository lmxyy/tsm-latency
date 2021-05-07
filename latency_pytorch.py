import argparse
import time

import torch
from tqdm import tqdm

from mobilenet_v2_tsm import MobileNetV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup_times', type=int, default=10)
    parser.add_argument('--test_times', type=int, default=20)
    args = parser.parse_args()
    net = MobileNetV2(n_class=27)
    net.load_state_dict(torch.load("models/pytorch/mobilenetv2_jester_online.pth.tar"))
    x = torch.ones([1, 3, 224, 224])
    shift_buffer = [torch.zeros([1, 3, 56, 56]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 20, 7, 7]),
                    torch.zeros([1, 20, 7, 7])]
    net.eval()
    with torch.no_grad():
        for _ in tqdm(range(args.warmup_times)):
            output = net(x, *shift_buffer)
            # print(output[0].abs().sum())
    start_time = time.time()
    with torch.no_grad():
        for _ in tqdm(range(args.test_times)):
            output = net(x, *shift_buffer)
    end_time = time.time()
    print((end_time - start_time) / args.test_times)
