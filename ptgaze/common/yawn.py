from ptgaze.common.face import Face
from omegaconf import DictConfig
import onnxruntime as rt
import cv2
import numpy as np


class Yawn:
    def __init__(self, config: DictConfig):
        self._config = config
        self.MAX_IMAGE_WIDTH = 100
        self.MAX_IMAGE_HEIGHT = 100
        self.IMAGE_PAIR_SIZE = (self.MAX_IMAGE_WIDTH, self.MAX_IMAGE_HEIGHT)
        self.COUNT = 0
        self.YAWN_COUNT = 0
        self.onnx_sess = rt.InferenceSession(self._config.yawn_detector.model_path)
        self.input_name = self.onnx_sess.get_inputs()[0].name
        self.label_name = self.onnx_sess.get_outputs()[0].name

    def _detect_yawn(self, frame, face_bbox, CONFIDENCE_THRESHOLD=0.5) -> int:
        face_bbox = np.hstack(np.round(face_bbox).astype(np.int).tolist())
        x1,y1,x2,y2 = face_bbox
        yawn_frame = frame[y1:y2, x1:x2, :]
        height, width, channels = frame.shape
        rect = (0, 0, width, height)
        nframe, pred = self._detect_image(yawn_frame, rect)   
        is_mouth_opened = True if pred >= CONFIDENCE_THRESHOLD else False
        if is_mouth_opened:
                self.COUNT += 1
        else:
            if self.COUNT >= 5: #should check
                self.YAWN_COUNT += 1
            self.COUNT = 0
        return self.YAWN_COUNT
    
    def _load_model(self):
        onnx_sess = rt.InferenceSession(self._config.yawn_detector.model_path)
        input_name = onnx_sess.get_inputs()[0].name
        label_name = onnx_sess.get_outputs()[0].name
        return onnx_sess, input_name, label_name

    def prepare_input_blob(self, im: np.ndarray) -> np.ndarray:
        if im.shape[0] != self.MAX_IMAGE_WIDTH or im.shape[1] != self.MAX_IMAGE_HEIGHT:
            im = cv2.resize(im, self.IMAGE_PAIR_SIZE)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return im

    def _detect_image(self, frame, face: Face):
        gray_img = self.prepare_input_blob(frame)
        image_frame = gray_img[:, :, np.newaxis]
        image_frame = image_frame / 255.0
        image_frame = np.expand_dims(image_frame, 0).astype(np.float32)
        pred = self.onnx_sess.run([self.label_name], {self.input_name: image_frame})[0]
        pred = np.squeeze(pred)
        pred = round(pred[()], 2)
        return frame, pred
