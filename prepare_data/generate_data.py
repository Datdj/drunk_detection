import sys
sys.path.append('simple_HRNet_master')
import os
from video import PoseSeriesGenerator, Video, Person
import argparse
import torch
from torchvision.transforms import transforms
from models2.hrnet import HRNet
import numpy as np
from tqdm import tqdm
import pickle

def main():

    parser = argparse.ArgumentParser(description='Generate series of poses from RWF-2000 dataset.')
    parser.add_argument(
        '--data-path',
        type=str,
        help='The path to the folder containing videos',
        required=True
    )
    parser.add_argument(
        '--out',
        type=str,
        help='output file path',
        required=True
    )
    parser.add_argument(
        '--series-length',
        type=int,
        help='the length of a pose series',
        default=10
    )
    parser.add_argument(
        '--min_poses',
        type=int,
        help='minimum number of poses detected for a series to be considered valid',
        default=7
    )
    parser.add_argument(
        '--hrnet-weight',
        type=str,
        help='hrnet weight file',
        default='weight/pose_hrnet_w48_384x288.pth'
    )
    parser.add_argument(
        '--input_format',
        type=str,
        help='video or images',
        default='video'
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load yolov5x
    yolov5x = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    yolov5x.to(device)

    # transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 288)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load hrnet
    hrnet = HRNet(48)
    hrnet.to(device)
    hrnet.eval()
    checkpoint = torch.load(args.hrnet_weight, map_location=device)
    hrnet.load_state_dict(checkpoint)

    # generate fight data
    # generate_data(args.data_path, yolov5x, transform, device, hrnet, args.out)
    gen_train_data(args.data_path, yolov5x, transform, device, hrnet, args.out, args.input_format)

def generate_data(data_dir, detector, transform, device, pose_model, out):
    '''
    Generate series of poses from a folder of videos and then normalize them.
    '''
    data = []
    mask = []
    for filename in tqdm(os.listdir(data_dir)):
        video = Video(os.path.join(data_dir, filename), detector, transform, device, pose_model)
        video.extract_poses()
        generator = PoseSeriesGenerator(video, 10, 7)
        series, mask_ = generator.generate()
        data.extend(series)
        mask.extend(mask_)
    data = np.asarray(data)
    mask = np.asarray(mask)

    # get the head by taking the average of five key points on the head (nose, left_eye, right_eye, left_ear, right_ear)
    data[:, :, 4][mask] = np.mean(data[:, :, :5][mask], axis=1)
    data = data[:, :, 4:]

    # min-max normalization
    min = np.min(data[:, :, :, :2][mask], axis=1, keepdims=True)
    max = np.max(data[:, :, :, :2][mask], axis=1, keepdims=True)
    data[:, :, :, :2][mask] = (data[:, :, :, :2][mask] - min) / (max - min)

    # get the origin by taking the average of four key points on the body (left_shoulder, right_shoulder, left_hip, right_hip)
    origin = (np.sum(data[:, :, 1:3, :2][mask], axis=1, keepdims=True) + np.sum(data[:, :, 7:9, :2][mask], axis=1, keepdims=True)) / 4

    # shift the origin
    data[:, :, :, :2][mask] = data[:, :, :, :2][mask] - origin

    # save into file
    np.save(out, data)

def gen_train_data(data_dir, detector, transform, device, pose_model, out, input_format):

    data = []
    for filename in tqdm(os.listdir(data_dir)):
        video = Video(os.path.join(data_dir, filename), detector, transform, device, pose_model, input_format)
        video.extract_poses_v2()
        video.normalize_poses()
        data.extend(video.people)
    with open(out, 'wb') as writer:
        pickle.dump(data, writer)

if __name__ == '__main__':
    main()