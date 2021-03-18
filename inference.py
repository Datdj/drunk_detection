import sys
sys.path.append('simple_HRNet_master')
sys.path.append('models')
import numpy as np
import cv2
import argparse
from prepare_data.video import Video
import torch
from torchvision.transforms import transforms
from models2.hrnet import HRNet
from lstm import DatLSTM
import torch.nn.functional as F
import os
from tqdm import tqdm

class ViolenceDetector():
    def __init__(self, person_detector, pose_estimater, fight_model, series_length, min_num_poses, device):
        self.person_detector = person_detector
        self.pose_estimater = pose_estimater
        self.fight_model = fight_model
        self.series_length = series_length
        self.min_num_poses = min_num_poses
        self.device = device

    def predict_and_save(self, video, out_file):
        # extract poses from this video
        video.extract_poses_v2()

        # normalize all the poses
        video.normalize_poses()

        # get all the valid frames
        video.get_valid_frames(self.series_length, self.min_num_poses)

        # get all the series
        out_frames = []
        series = []
        for person in video.people:
            for start in person.valid_frames:
                out_frames.append(person.frames[person.mapping[start]])
                series.append([person.frames[person.mapping[i]].pose if i in person.mapping.keys() else np.zeros((13, 3)) for i in range(start, start + self.series_length)])
        series = torch.Tensor(series).type(torch.float32).to(self.device)
        series = torch.flatten(series, start_dim=2)
        
        # forward all series through lstm
        with torch.no_grad():
            out = self.fight_model(series)
        out = F.softmax(out, dim=1)
        result = out[:, 1] > 0.5

        # draw results on the video
        cap = cv2.VideoCapture(video.file_path)
        fourcc = int(cap.get(6))
        fps = cap.get(5)
        w = int(cap.get(3))
        h = int(cap.get(4))
        out_vid = cv2.VideoWriter(out_file, fourcc, fps, (w, h))

        out_results = {}
        for i in range(len(out_frames)):
            if out_frames[i].index not in out_results.keys():
                out_results[out_frames[i].index] = [(out_frames[i], result[i])]
            else:
                out_results[out_frames[i].index].append((out_frames[i], result[i]))

        frame_idx = 0
        while True:
            ret, frame = cap.read()

            # Check for end of video
            if ret == False:
                break

            if frame_idx in out_results.keys():
                for person_i in out_results[frame_idx]:
                    box = person_i[0].bbox
                    if person_i[1]:
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255))
                    else:
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))
            out_vid.write(frame)
            frame_idx += 1

        cap.release()
        out_vid.release()
           
def main():

    parser = argparse.ArgumentParser(description='A demo of the ViolenceDetector')
    parser.add_argument(
        '--video',
        type=str,
        help='The path to the video or the folder containing videos',
        required=True
    )
    parser.add_argument(
        '--lstm-weight',
        type=str,
        help='lstm weight file',
        required=True
    )
    parser.add_argument(
        '--hrnet-weight',
        type=str,
        help='hrnet weight file',
        default='weight/pose_hrnet_w48_384x288.pth'
    )
    parser.add_argument(
        '--series-length',
        type=int,
        help='the length of a pose series',
        default=10
    )
    parser.add_argument(
        '--min-poses',
        type=int,
        help='minimum number of poses detected for a series to be considered valid',
        default=7
    )
    parser.add_argument(
        '--out',
        type=str,
        help='output file path',
        required=True
    )
    parser.add_argument(
        '--mode',
        type=str,
        help="'video' or 'folder'",
        default='video'
    )
    args = parser.parse_args()
    if args.mode not in ['video', 'folder']:
        raise ValueError('mode must be video or folder')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load yolov5
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

    # Load pretrained lstm model
    checkpoint = torch.load(args.lstm_weight)
    lstm_model = DatLSTM(39, 64, 2, args.series_length)
    lstm_model.to(device)
    lstm_model.eval()
    lstm_model.load_state_dict(checkpoint)

    # Load the detector
    violence_detector = ViolenceDetector(yolov5x, hrnet, lstm_model, args.series_length, args.min_poses, device)

    if args.mode == 'video':

        # Load a video
        vid = Video(args.video, yolov5x, transform, device, hrnet, 'video')

        # inference
        violence_detector.predict_and_save(vid, args.out)
    
    else:
        for video in tqdm(os.listdir(args.video)):
            vid = Video(os.path.join(args.video, video), yolov5x, transform, device, hrnet, 'video')
            violence_detector.predict_and_save(vid, os.path.join(args.out, video))

if __name__ == '__main__':
    main()