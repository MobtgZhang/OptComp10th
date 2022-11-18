import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from .utils import tflite_inference
class InceptionResNetV2:
    def __init__(self,facenet_weights_path,min_face_size: int = 40,scale_factor: float = 0.7,):
        self.face_encoder = tflite.Interpreter(model_path=facenet_weights_path)
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
    def encode(self, image):
        image = np.expand_dims(image,0)
        return tflite_inference(self.face_encoder,image)[0]
