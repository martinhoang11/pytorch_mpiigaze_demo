import os
from ptgaze.common.face import Face
from typing import List
import numpy as np
import math
import cv2
import onnxruntime
from PIL import Image
import matplotlib.pyplot as plt
import re
import sys
import os
import cv2
import numpy as np
import time
import traceback
import gc
from omegaconf import DictConfig
from utils import *

class Tracker:
    def __init__(self, width, height, model_type=3, config=DictConfig, threshold=None) -> None:
        self._config = config
        self.threshold = 0.15
        self.max_threads = 4
        self.max_faces = 1
        self.detection_threshold = 0.2
        self.res = 224.
        self.out_res = 27.
        self.out_res_i = int(self.out_res) + 1
        self.logit_factor = 16.
        self.res_i = int(self.res)

        self.mean = np.float32(np.array([0.485, 0.456, 0.406]))
        self.std = np.float32(np.array([0.229, 0.224, 0.225]))
        self.mean_1 = self.mean / self.std
        self.std_1 = self.std * 255.0

        self.mean_1 = - self.mean_1
        self.std_1 = 1.0 / self.std_1

        self.mean_224 = np.tile(self.mean_1, [224, 224, 1])
        self.std_224 = np.tile(self.std_1, [224, 224, 1])

        self.mean_res = self.mean_224
        self.std_res = self.std_224

        self.width = width
        self.height = height

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = max(self.max_threads,4)
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.model_type = model_type
        self.models = [
            "lm_model0_opt.onnx",
            "lm_model1_opt.onnx",
            "lm_model2_opt.onnx",
            "lm_model3_opt.onnx",
            "lm_model4_opt.onnx"
        ]
        model = "lm_modelT_opt.onnx"
        if model_type >= 0:
            model = self.models[self.model_type]
        if model_type == -2:
            model = "lm_modelV_opt.onnx"
        if model_type == -3:
            model = "lm_modelU_opt.onnx"
        model_base_path = get_model_base_path(self._config.lms_detector.model_path)
        

        if threshold is None:
            threshold = 0.6
            if model_type < 0:
                threshold = 0.87

        self.session = onnxruntime.InferenceSession(os.path.join(model_base_path, model), sess_options=options)
        self.input_name = self.session.get_inputs()[0].name
        options.intra_op_num_threads = 1
        self.detection = onnxruntime.InferenceSession(os.path.join(model_base_path, "mnv3_detection_opt.onnx"), sess_options=options)

    def detect_faces(self, frame):
        im = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)[:,:,::-1] * self.std_224 + self.mean_224
        im = np.expand_dims(im, 0)
        im = np.transpose(im, (0,3,1,2))
        outputs, maxpool = self.detection.run([], {'input': im})
        outputs = np.array(outputs)
        maxpool = np.array(maxpool)
        outputs[0, 0, outputs[0, 0] != maxpool[0, 0]] = 0
        detections = np.flip(np.argsort(outputs[0,0].flatten()))
        results = []
        for det in detections[0:self.max_faces]:
            y, x = det // 56, det % 56
            c = outputs[0, 0, y, x]
            r = outputs[0, 1, y, x] * 112.
            x *= 4
            y *= 4
            r *= 1.0
            if c < self.detection_threshold:
                break
            results.append((x - r, y - r, 2 * r, 2 * r * 1.0))
        results = np.array(results).astype(np.float32)
        if results.shape[0] > 0:
            results[:, [0,2]] *= frame.shape[1] / 224.
            results[:, [1,3]] *= frame.shape[0] / 224.
        # print(results)
        return results
    
    def landmarks(self, tensor, crop_info):
        crop_x1, crop_y1, scale_x, scale_y, _ = crop_info
        avg_conf = 0
        res = self.res - 1
        c0, c1, c2 = 66, 132, 198
        if self.model_type == -1:
            c0, c1, c2 = 30, 60, 90
        t_main = tensor[0:c0].reshape((c0, self.out_res_i * self.out_res_i))
        t_m = t_main.argmax(1)
        indices = np.expand_dims(t_m, 1)
        t_conf = np.take_along_axis(t_main, indices, 1).reshape((c0,))
        t_off_x = np.take_along_axis(tensor[c0:c1].reshape((c0,self.out_res_i * self.out_res_i)), indices, 1).reshape((c0,))
        t_off_y = np.take_along_axis(tensor[c1:c2].reshape((c0,self.out_res_i * self.out_res_i)), indices, 1).reshape((c0,))
        t_off_x = res * logit_arr(t_off_x, self.logit_factor)
        t_off_y = res * logit_arr(t_off_y, self.logit_factor)
        t_x = crop_y1 + scale_y * (res * np.floor(t_m / self.out_res_i) / self.out_res + t_off_x)
        t_y = crop_x1 + scale_x * (res * np.floor(np.mod(t_m, self.out_res_i)) / self.out_res + t_off_y)
        avg_conf = np.average(t_conf)
        lms = np.stack([t_x, t_y, t_conf], 1)
        lms[np.isnan(lms).any(axis=1)] = np.array([0.,0.,0.], dtype=np.float32)
        if self.model_type == -1:
            lms = lms[[0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,6,7,7,8,8,9,10,10,11,11,12,21,21,21,22,23,23,23,23,23,13,14,14,15,16,16,17,18,18,19,20,20,24,25,25,25,26,26,27,27,27,24,24,28,28,28,26,29,29,29]]
            part_avg = np.mean(np.partition(lms[:,2],3)[0:3])
            if part_avg < 0.65:
                avg_conf = part_avg
        return (avg_conf, np.array(lms))
    
    def predict(self, frame) -> List[Face]:
        im = frame

        """
        Face detection -> crop only face area
        """
        new_faces = []
        detected = []
        new_faces.extend(self.detect_faces(frame))
        crops = []
        crop_info = []
        num_crops = 0
        for j, (x,y,w,h) in enumerate(new_faces):
            crop_x1 = x - int(w * 0.1)
            crop_y1 = y - int(h * 0.125)
            crop_x2 = x + w + int(w * 0.1)
            crop_y2 = y + h + int(h * 0.125)

            crop_x1, crop_y1 = clamp_to_im((crop_x1, crop_y1), self.width, self.height)
            crop_x2, crop_y2 = clamp_to_im((crop_x2, crop_y2), self.width, self.height)
            scale_x = float(crop_x2 - crop_x1) / self.res
            scale_y = float(crop_y2 - crop_y1) / self.res


            if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
                continue
            crop = preprocess(im, (crop_x1, crop_y1, crop_x2, crop_y2), self.res_i, self.std_res, self.mean_res)
            crops.append(crop)
            crop_info.append((crop_x1, crop_y1, scale_x, scale_y, 0.0))

        """
        Landmark detection from cropped area
        """
        outputs = {}
        output = self.session.run([], {self.input_name: crops[0]})[0]
        conf, lms = self.landmarks(output[0], crop_info[0])
        outputs[crop_info[0]] = (conf, (lms, 0), 0)
        actual_faces = []
        good_crops = []
        for crop in crop_info:
            if crop not in outputs:
                continue
            conf, lms, i = outputs[crop]
            x1, y1, _ = lms[0].min(0)
            x2, y2, _ = lms[0].max(0)
            bb = (x1, y1, x2 - x1, y2 - y1)
            outputs[crop] = (conf, lms, i, bb)
            actual_faces.append(bb)
            good_crops.append(crop)
        groups = group_rects(actual_faces)
        best_results = {}
        for crop in good_crops:
            conf, lms, i, bb = outputs[crop]
            if conf < self.threshold:
                continue;
            group_id = groups[str(bb)][0]
            if not group_id in best_results:
                best_results[group_id] = [-1, [], 0]
            if conf > self.threshold and best_results[group_id][0] < conf + crop[4]:
                best_results[group_id][0] = conf + crop[4]
                best_results[group_id][1] = lms
                best_results[group_id][2] = crop[4]

        sorted_results = sorted(best_results.values(), key=lambda x: x[0], reverse=True)[:1]
        lms = lms[0][:, 0:2]
        lms[:,[0, 1]] = lms[:,[1, 0]]
        x1, y1 = tuple(lms[0:66].min(0))
        x2, y2 = tuple(lms[0:66].max(0))
        bbox = (x1, y1, x2, y2)
        bbox = np.array(bbox).reshape(2,2)
        detected.append(Face(bbox, lms))
        return detected






        
