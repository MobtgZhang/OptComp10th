import cv2
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
l2_normalizer = Normalizer('l2')
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def tflite_inference(model, img):
    """Inferences an image through the model with tflite interpreter on CPU
    :param model: a tflite.Interpreter loaded with a model
    :param img: image
    :return: list of outputs of the model
    """
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.resize_tensor_input(input_details[0]["index"], img.shape)
    model.allocate_tensors()
    model.set_tensor(input_details[0]["index"], img.astype(np.float32))
    model.invoke()
    return [model.get_tensor(elem["index"]) for elem in output_details]
def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.encode(face)
    return encode
def get_face(img, box):
    x1, y1, x2, y2 = box
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)
def draw_faces(img_frame ,detector,encoder,encoding_dict,
           recognition_t=0.5,confidence_t=0.99,required_size = (160,160)):
    img_rgb = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
    bboxes, landmarks, scores = detector.detect_faces(img_rgb)
    for box, landmark, score in zip(bboxes, landmarks, scores):
        if score < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, box)
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img_frame, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img_frame, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img_frame, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img_frame, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img_frame
