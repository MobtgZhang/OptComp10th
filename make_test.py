import os
import sqlite3
import cv2
import base64
import radar
import random
import string
import uuid
import numpy as np
from mtcnn import MTCNN
from facenet import InceptionResNetV2
from facenet.utils import get_encode

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def test():
    datadb_file = 'people.db'
    conn = sqlite3.connect(datadb_file)
    cursor = conn.cursor()
    sql_sent = "SELECT IDName,name FROM facedataset"
    results = cursor.execute(sql_sent)
    results = results.fetchall()
    print(len(results))
    conn.commit()
    conn.close()
def main():
    datadb_file = 'people.db'
    conn = sqlite3.connect(datadb_file)
    cursor = conn.cursor()
    # 查询表是否存在数据库当中
    sql_sent = "SELECT COUNT(*) FROM sqlite_master where type ='table' and name ='facedataset'"
    result = cursor.execute(sql_sent)
    conn.commit()
    number = result.fetchone()[0]
    if number == 0:
        # 创建数据库
        print("创建数据库")
        sql_sent = "CREATE TABLE facedataset (IDName TEXT  PRIMARY KEY,\
                                                          name TEXT,\
                                                          classnumber TEXT,\
                                                          time TEXT, \
                                                          temperature FLOAT,\
                                                          covid19result TEXT,\
                                                          image_raw_base64_bytes BLOB,\
                                                          image_raw_shape TEXT,\
                                                          image_face_base64_bytes BLOB,\
                                                          image_face_shape TEXT,\
                                                          image_emb_base64_bytes  BLOB)"
        cursor.execute(sql_sent)
        conn.commit()
    # 增加一些常用的值
    pictures_dir = "./pictures"
    weights_dir = "./lite-weights"
    required_shape = (160,160)
    weights_mtcnn_dir = os.path.join(weights_dir, "mtcnn")
    facenet_weights_path = os.path.join(weights_dir,"facenet","facenet_keras_weights.lite")

    for file_name in os.listdir(pictures_dir):
        image_path = os.path.join(pictures_dir, file_name)
        name_of_pic = file_name.split(".")[0]
        sql_query = "SELECT name,image_raw_base64_bytes,image_raw_shape,\
                    image_face_base64_bytes,image_face_shape,image_emb_base64_bytes FROM facedataset WHERE name='%s'" % name_of_pic
        output = cursor.execute(sql_query)
        conn.commit()
        out_v = output.fetchall()
        if len(out_v) > 0:
            name = out_v[0][0]
            image_raw_flatten = np.frombuffer(base64.b64decode(out_v[0][1]),np.uint8)
            image_raw_shape = eval(out_v[0][2])
            image_raw = image_raw_flatten.reshape(image_raw_shape)
            image_raw_flatten = np.frombuffer(base64.b64decode(out_v[0][5]), np.float32)
            print(name,image_raw.shape,image_raw_flatten.shape)
        else:
            print("加入数据库:",name_of_pic)
            img_BGR = cv2.imread(image_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
            face_detector = MTCNN(weights_mtcnn_dir)
            face_encoder = InceptionResNetV2(facenet_weights_path)
            bboxes = face_detector.detect_face(img_RGB)
            x1, y1, x2, y2 = bboxes[0]['box']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            img_face = img_RGB[y1:y2, x1:x2].copy()
            encode = get_encode(face_encoder,img_face,required_shape)
            sql_insert = "INSERT INTO facedataset (IDName,\
                                                        name,\
                                                        classnumber,\
                                                        time,\
                                                        temperature,\
                                                        covid19result,\
                                                        image_raw_base64_bytes,\
                                                        image_raw_shape,\
                                                        image_face_base64_bytes,\
                                                        image_face_shape,\
                                                        image_emb_base64_bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?);"
            image_emb_base64_bytes = base64.b64encode(encode)
            image_raw_base64_bytes = base64.b64encode(img_RGB)
            img_raw_shape = str(img_RGB.shape)
            image_face_base64_bytes = base64.b64encode(img_face)
            img_face_shape = str(img_face.shape)
            time_now = str(radar.random_datetime())
            temperature = (38 - 35) * random.random()
            covid19result = "No" if random.randint(0, 1) == 0 else "Yes"
            num = string.digits
            classnumber = "".join(random.sample(num, 10))
            name_ids = str(uuid.uuid4()).upper()
            cursor.execute(sql_insert,
                           (name_ids, name_of_pic, classnumber, time_now, temperature, covid19result,
                            image_raw_base64_bytes,img_raw_shape,image_face_base64_bytes,img_face_shape,image_emb_base64_bytes))
            conn.commit()
    conn.close()
if __name__ == "__main__":
    main()

