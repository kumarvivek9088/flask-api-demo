from flask import Flask,jsonify,request
from flask_cors import CORS
import cv2
# from keras.models import load_model
import tensorflow as tf
import numpy as np
import base64
# json_file = open("facialemotionmodel.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)

# model.load_weights("facialemotionmodel.h5")
model = tf.keras.models.load_model("facialemotionmodel.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam=cv2.VideoCapture(0)
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
app = Flask(__name__)
CORS(app,origins='*')
@app.route('/')
def hello():
    return 'Hello World !'

@app.route('/predictemotion',methods = ['POST'])
def predict_emotion():
    data = request.get_json()
    if data is None:
        return jsonify({"error":"invalid json data"}),400
    # print(data)
    image_string = data['image']
    image_string = image_string.split(',')
    if len(image_string)==2:
        image_string = image_string[1]
    else:
        image_string = image_string[0]
        
    image_data = base64.b64decode(image_string)
    nparr = np.frombuffer(image_data,np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(img,1.3,5)
    print(faces)
    for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(img,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            im = extract_features(image)
            pred = model.predict(im)
            prediction_label = labels[pred.argmax()]
            print(prediction_label)
            return jsonify({"emotion":prediction_label,"x1": str(p),"y1":str(q),"x2":str(r),"y2":str(s)}),200
    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return jsonify({"emotion":"face is not detected","x1":str(0),"y1":str(0),"x2":str(0),"y2":str(0)}),200
    

if __name__ == '__main__':
    app.run()
    # app.run(debug=True)
    
