import sys
sys.path.append('sort_master')

from sort import *

import random
import numpy as np
import cv2
import torch
import os

class PoseSeriesGenerator():
    '''
    This will take in a video and spit out a list of series of poses extracted
    from the video. I probably should have wrote this as a function instead of
    a class though :v
    '''

    def __init__(self, video, series_length, min_num_poses):
        self.video = video
        self.series_length = series_length
        self.min_num_poses = min_num_poses
        
    def generate(self):
        '''
        The function that spits out a a list of series of poses AND a boolean
        mask that shows which frame in a series is valid (we could lost some
        frames due to misdetections by the detector)
        '''

        series = []
        mask = []
        # for each person appearing in the video
        for person in self.video.people:

            # If a person appears in n frames and we want a series of poses of
            # length m then we will randomly generate n // m series of this 
            # person. For example, if a person appears in 70 frame and the
            # series' length is 10 then we generate 7 series from this person
            num_series_to_get = person.num_frames // self.series_length
            starts = random.sample(person.mapping.keys(), num_series_to_get)
            for start in starts:
                frames = [i for i in range(start, start + 10) if i in person.mapping.keys()]
                if len(frames) >= self.min_num_poses:
                    mask.append([True if i in person.mapping.keys() else False for i in range(start, start + 10)])
                    series.append([person.frames[person.mapping[i]].pose if i in person.mapping.keys() else np.zeros((17, 3)) for i in range(start, start + 10)])
        return series, mask

class Video():
    def __init__(self, file_path, detector, transform, device, pose_model, format_):
        self.num_people = 0
        self.people = []
        self.file_path = file_path
        self.mapping = {}
        self.poses_extracted = False
        self.poses_normalized = False
        self.detector = detector
        self.transform = transform
        self.device = device
        self.pose_model = pose_model
        if format_ in ['video', 'images']:
            self.format_ = format_ # format is either 'video' or 'images'
        else:
            raise ValueError("format_ must be either 'video' or 'images'")

    def add_new_person(self, id, first_bbox, first_pose, first_frame):
        self.people.append(Person(id, first_bbox, first_pose, first_frame))
        self.mapping[id] = self.num_people
        self.num_people += 1

    def update_old_person(self, id, bbox, pose, frame):
        self.people[self.mapping[id]].update(bbox, pose, frame)

    def is_person_exist(self, id):
        if id in self.mapping:
            return True
        else:
            return False

    def extract_poses(self):
        if self.poses_extracted:
            print('Poses have already been extracted.')
            return

        # Initialize a VideoCapture object
        cap = cv2.VideoCapture(self.file_path)
        w = cap.get(3)
        h = cap.get(4)

        # Create an instance of SORT
        KalmanBoxTracker.count = 0
        mot_tracker = Sort()

        frame_idx = 0

        while True:
            # Read a single frame
            ret, frame = cap.read()

            # Check for end of video
            if ret == False:
                break

            # TODO: read a batch of frames before passing them through yolo and hrnet,
            # hopefully that will speed up data generation process

            # Run yolov5 on the frame
            result = self.detector(frame[:, :, ::-1])

            # Get bounding boxes
            result = result.pred[0].cpu().numpy()
            bboxes = result[result[:, -1] == 0][:, :-1]
            bboxes = bboxes[bboxes[:, -1] > 0.6]

            # Get number of people
            num_people = bboxes.shape[0]

            # Update SORT
            if num_people == 0:
                bboxes = mot_tracker.update()
            else:
                bboxes = mot_tracker.update(bboxes)
                num_people = bboxes.shape[0]

            if num_people > 0:

                # Making sure the bounding box is inside the image
                bboxes[:, 0] = np.maximum(bboxes[:, 0], 0)
                bboxes[:, 1] = np.maximum(bboxes[:, 1], 0)
                bboxes[:, 2] = np.minimum(bboxes[:, 2], w)
                bboxes[:, 3] = np.minimum(bboxes[:, 3], h)
                
                people = torch.empty((num_people, 3, 384, 288))
                points = np.empty((num_people, 17, 3))
                for i in range(num_people): # For each person
                    # Get rounded coordinates for cropping
                    x1 = int(np.round(bboxes[i, 0]))
                    y1 = int(np.round(bboxes[i, 1]))
                    x2 = int(np.round(bboxes[i, 2]))
                    y2 = int(np.round(bboxes[i, 3]))
                    # Crop out the person
                    person = frame[y1:y2, x1:x2]
                    person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                    people[i] = self.transform(person)

                # Pass through HRNet
                people = people.to(self.device)
                poses = self.pose_model(people).detach().cpu().numpy()

                # Get back the original coordinate
                poses = poses.reshape((num_people, 17, -1))
                max_point = np.unravel_index(np.argmax(poses, axis=2), shape=(96, 72))
                points[:, :, 1] = max_point[0] / 96 * (bboxes[:, 3:4] - bboxes[:, 1:2]) + bboxes[:, 1:2]
                points[:, :, 0] = max_point[1] / 72 * (bboxes[:, 2:3] - bboxes[:, 0:1]) + bboxes[:, 0:1]
                points[:, :, 2] = np.max(poses, axis=2)

                # Update data
                for i in range(num_people):
                    if not self.is_person_exist(bboxes[i, -1]):
                        self.add_new_person(bboxes[i, -1], bboxes[i, :-1], points[i], frame_idx)
                    else:
                        self.update_old_person(bboxes[i, -1], bboxes[i, :-1], points[i], frame_idx)
                frame_idx += 1
        
        # Release the VideoCapture
        cap.release()

        # Update the state
        self.poses_extracted = True
        return

    def extract_poses_v2(self, yolo_batch_size=16, hrnet_batch_size=16):
        if self.poses_extracted:
            print('Poses have already been extracted.')
            return

        if self.format_ == 'video':
            # Initialize a VideoCapture object
            cap = cv2.VideoCapture(self.file_path)
            w = cap.get(3)
            h = cap.get(4)
        else:
            images = os.listdir(self.file_path)
            num_frames = len(images)
            start = 0
            _ = cv2.imread(os.path.join(self.file_path, images[0]))
            w = _.shape[1]
            h = _.shape[0]

        # Create an instance of SORT
        KalmanBoxTracker.count = 0
        mot_tracker = Sort()

        frame_idx = 0

        EOF = False
        while not EOF:

            frames = []
            if self.format_ == 'video':
                for i in range(yolo_batch_size):
                    # Read a single frame
                    ret, frame = cap.read()
                    # Check for end of video
                    if ret == False:
                        EOF = True
                        break
                    frames.append(frame[:, :, ::-1])
            else:
                if start + yolo_batch_size < num_frames:
                    end = start + yolo_batch_size
                else:
                    end = num_frames
                    EOF = True
                for i in range(start, end):
                    # Read a single frame
                    frame = cv2.imread(os.path.join(self.file_path, images[i]))
                    frames.append(frame[:, :, ::-1])
                start = end
            
            if len(frames) == 0:
                continue

            # Run yolov5 on the frames
            detections = self.detector(frames)

            bboxes = []
            indices = []
            current_index = 0
            for i in range(len(frames)):
                # Get bounding boxes
                bboxes_i = detections.pred[i].cpu().numpy()
                bboxes_i = bboxes_i[bboxes_i[:, -1] == 0][:, :-1]
                bboxes_i = bboxes_i[bboxes_i[:, -1] > 0.6]
                num_people_i = bboxes_i.shape[0]

                # Update SORT
                if num_people_i == 0:
                    bboxes_i = mot_tracker.update()
                    indices.append((-1, -1))
                else:
                    bboxes_i = mot_tracker.update(bboxes_i)
                    num_people_i = bboxes_i.shape[0]
                    if num_people_i > 0:
                        bboxes.append(bboxes_i)
                        indices.append((current_index, current_index + num_people_i))
                        current_index += num_people_i
                    else:
                        indices.append((-1, -1))
            if len(bboxes) > 0:
                debug = bboxes
                bboxes = np.concatenate(bboxes)
                # Making sure the bounding box is inside the image
                bboxes[:, 0] = np.maximum(bboxes[:, 0], 0)
                bboxes[:, 1] = np.maximum(bboxes[:, 1], 0)
                bboxes[:, 2] = np.minimum(bboxes[:, 2], w)
                bboxes[:, 3] = np.minimum(bboxes[:, 3], h)
                num_people = bboxes.shape[0]
                people = torch.empty((num_people, 3, 384, 288))
                points = np.empty((num_people, 17, 3))
                for i in range(len(frames)):
                    if indices[i] != (-1, -1):
                        for j in range(indices[i][0], indices[i][1]):
                            # Get rounded coordinates for cropping
                            x1 = int(np.round(bboxes[j, 0]))
                            y1 = int(np.round(bboxes[j, 1]))
                            x2 = int(np.round(bboxes[j, 2]))
                            y2 = int(np.round(bboxes[j, 3]))
                            # Crop out the person
                            person = frames[i][y1:y2, x1:x2]
                            person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                            people[j] = self.transform(person)
                            # TODO: resize individual person but normalize a whole batch

                # Pass through HRNet
                poses = np.empty((num_people, 17, 96, 72))
                people = people.to(self.device)
                for i in range(0, num_people, hrnet_batch_size):
                    ii = min(i + hrnet_batch_size, num_people)
                    poses[i:ii] = self.pose_model(people[i:ii]).detach().cpu().numpy()
                
                # Get back the original coordinate
                try:
                    poses = poses.reshape((num_people, 17, -1))
                except:
                    print(debug)
                    exit()
                max_point = np.unravel_index(np.argmax(poses, axis=2), shape=(96, 72))
                points[:, :, 1] = max_point[0] / 96 * (bboxes[:, 3:4] - bboxes[:, 1:2]) + bboxes[:, 1:2]
                points[:, :, 0] = max_point[1] / 72 * (bboxes[:, 2:3] - bboxes[:, 0:1]) + bboxes[:, 0:1]
                points[:, :, 2] = np.max(poses, axis=2)
            
            # Update data
            for i in range(len(frames)):
                if indices[i] != (-1, -1):
                    for j in range(indices[i][0], indices[i][1]):
                        if not self.is_person_exist(bboxes[j, -1]):
                            self.add_new_person(bboxes[j, -1], bboxes[j, :-1], points[j], frame_idx)
                        else:
                            self.update_old_person(bboxes[j, -1], bboxes[j, :-1], points[j], frame_idx)
                frame_idx += 1
        
        if self.format_ == 'video':
            # Release the VideoCapture
            cap.release()

        # Update the state
        self.poses_extracted = True
        return
    
    # TODO: Maybe in version 3, we can extract all people before extracting poses.

    def normalize_poses(self):
        if not self.poses_extracted:
            print("Poses haven't been extracted.")
            return
        if self.poses_normalized:
            print('Poses have already been normalized.')
            return
        
        for person in self.people:
            person.normalize()
        
        self.poses_normalized = True

    def get_valid_frames(self, series_length, min_num_poses):
        for person in self.people:
            person.get_valid_frames(series_length, min_num_poses)

class Person():
    '''
    A person with a unique id in a video
    '''
    def __init__(self, id, first_bbox, first_pose, first_frame):
        self.id = id
        self.frames = [Frame(first_frame, first_bbox, first_pose)]
        self.num_frames = 1
        self.mapping = {first_frame: 0}
        self.is_normalized = False
        self.valid_frames = None

    def update(self, bbox, pose, frame):
        self.frames.append(Frame(frame, bbox, pose))
        self.mapping[frame] = self.num_frames
        self.num_frames += 1

    def normalize(self):
        if self.is_normalized:
            return

        for frame in self.frames:
            frame.normalize()

        self.is_normalized = True

    def get_valid_frames(self, series_length, min_num_poses):
        '''
        A valid frame is a frame that can be the start of a series
        '''
        self.valid_frames = []
        if self.num_frames < min_num_poses:
            return
        for i in self.mapping.keys():
            test = [j for j in range(i, i + series_length) if j in self.mapping.keys()]
            if len(test) >= min_num_poses:
                self.valid_frames.append(i)
        return

class Frame():
    '''
    Containing bounding box and pose of a person in one frame
    '''
    def __init__(self, index, bbox, pose):
        self.index = index
        self.bbox = bbox
        self.pose = pose
        self.is_normalized = False

    def normalize(self):
        if self.is_normalized:
            return

        # get the head by taking the average of five key points on
        # the head (nose, left_eye, right_eye, left_ear, right_ear)
        self.pose[4] = np.mean(self.pose[:5], axis=0)
        self.pose = self.pose[4:]

        # min-max normalization
        min_ = np.min(self.pose[:, :2], axis=0, keepdims=True)
        max_ = np.max(self.pose[:, :2], axis=0, keepdims=True)
        self.pose[:, :2] = (self.pose[:, :2] - min_) / (max_ - min_)

        # get the origin by taking the average of four key points on
        # the body (left_shoulder, right_shoulder, left_hip, right_hip)
        origin = (np.sum(self.pose[1:3, :2], axis=0, keepdims=True) + np.sum(self.pose[7:9, :2], axis=0, keepdims=True)) / 4

        # shift the origin
        self.pose[:, :2] = self.pose[:, :2] - origin

        self.is_normalized = True

        # TO DO: Do this normalization right after HRNet to save time and simplify things