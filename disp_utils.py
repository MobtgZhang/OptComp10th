import base64
import os
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from facenet.utils import get_face,get_encode,l2_normalizer
def get_usb_camera_index():
    # 获取USB摄像头的索引
    camera_path = Path('/sys/class/video4linux/')
    camera_list = list(camera_path.glob('video*'))
    camera_list.sort()
    real_camera_list = []
    for index in range(len(camera_list)):
        camera = camera_list[index]
        name_file = camera.joinpath('name')
        with open(name_file, 'r') as f:
            info = f.readline()
            if 'Capture' in info:
                data_dict = {
                    "index":index,
                    "name":info.strip()
                }
                real_camera_list.append(data_dict)
    return real_camera_list
def draw_from_database(img_frame,detector,encoder,databaseconn,
           recognition_t=0.5,confidence_t=0.99,required_size = (160,160)):
    img_rgb = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
    bboxes = detector.detect_face(img_rgb)
    ret_infos = []
    for data_dict in bboxes:
        box = data_dict['box']
        score = data_dict['confidence']
        if score < confidence_t:
            continue
        ret_real_face, pt_1, pt_2 = get_face(img_rgb, box)
        if ret_real_face.shape[0] == 0 or ret_real_face.shape[1] == 0:
            continue
        ret_real_face = cv2.cvtColor(ret_real_face, cv2.COLOR_BGR2RGB)
        encode = get_encode(encoder, ret_real_face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        ret_name = 'unknown'
        distance = float("inf")
        sql_sent = "SELECT IDName,name,classnumber,image_face_shape,image_face_base64_bytes,image_emb_base64_bytes FROM facedataset"
        cursor = databaseconn.cursor()

        result = cursor.execute(sql_sent)
        databaseconn.commit()
        result_lists = result.fetchall()

        ret_classnumber = None
        for IDName,db_name,classnumber,image_face_shape,image_face_base64_bytes,image_emb_base64_bytes in result_lists:
            db_encode_bytes = base64.b64decode(image_emb_base64_bytes)
            db_encode = np.frombuffer(db_encode_bytes,np.float32)
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                ret_name = db_name
                ret_classnumber = classnumber
                distance = dist
                real_face_bytes = base64.b64decode(image_face_base64_bytes)
                ret_real_face = np.frombuffer(real_face_bytes,np.uint8)
                image_face_shape = eval(image_face_shape)
                ret_real_face = ret_real_face.reshape(image_face_shape)
                break
        if ret_name == 'unknown':
            cv2.rectangle(img_frame, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img_frame, ret_name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img_frame, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img_frame, ret_name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
        ret_infos.append((ret_name,ret_classnumber,ret_real_face))
    return img_frame,ret_infos
def get_filename(file_dir):
    file_count = 0
    root_filename = "screen_shot"
    for save_file_name in os.listdir(file_dir):
        filename,_ = save_file_name.split('.')
        _,filenum = filename.split('-')
        if int(filenum)>file_count:
            file_count = int(filenum)
    save_file_name = os.path.join(file_dir,root_filename + "-"+str(file_count+1)+".png")
    return save_file_name
# 计算人脸特征向量间的欧氏距离，设置阈值，判断是否为同一个人脸
def match_faces(faces_emb, known_faces_emb, match_threshold = 0.8):
    isExistDst = False
    distance = np.linalg.norm(known_faces_emb[0] - faces_emb[0])
    if (distance < match_threshold):
        isExistDst = True
    return isExistDst,distance
