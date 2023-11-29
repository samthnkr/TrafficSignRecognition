from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import numpy as np

# Classes of trafic signs
classes = { 
            0:'Speed limit (5km/h)',
            1:'Speed limit (15km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (40km/h)',
            4:'Speed limit (50km/h)',
            5:'Speed limit (60km/h)',
            6:'Speed limit (70km/h)',
            7:'speed limit (80km/h)',
            8:'Dont Go straight or left',
            9:'Dont Go straight or Right',
            10:'Dont Go straight',
            11:'Dont Go Left',
            12:'Dont Go Left or Right',
            13:'Dont Go Right',
            14:'Dont overtake from Left',
            15:'No Uturn',
            16:'No Car',
            17:'No horn',
            18:'Speed limit (40km/h)',
            19:'Speed limit (50km/h)',
            20:'Go straight or right',
            21:'Go straight',
            22:'Go Left',
            23:'Go Left or right',
            24:'Go Right',
            25:'keep Left',
            26:'keep Right',
            27:'Roundabout mandatory',
            28:'watch out for cars',
            29:'Horn',
            30:'Bicycles crossing',
            31:'Uturn',
            32:'Road Divider',
            33:'Traffic signals',
            34:'Danger Ahead',
            35:'Zebra Crossing',
            36:'Bicycles crossing',
            37:'Children crossing',
            38:'Dangerous curve to the left',
            39:'Dangerous curve to the right',
            40:'Go right or straight',
            41:'Go left or straight',
            42:'ZigZag Curve',
            43:'Train Crossing',
            44:'Under Construction',
            45:'Fences',
            46:'Heavy Vehicle Accidents',
            47:'Give Way',
            48:'No stopping',
            49:'No entry', }

from keras.models import load_model
model = load_model('TSR.h5')

def test_on_img(image):
    data=[]
    # image = Image.open(img)
    image = image.convert('L')
    image = image.resize((30,30))
    image = np.array(image)
    data.append(image)
    X_test=np.array(data)
    # print(X_test.shape)
    Y_pred = np.argmax(model.predict(X_test), axis=1)
    return image,Y_pred

app = FastAPI()

# Configure CORS settings
origins = [
    "http://127.0.0.1:8080",  # Change this to the origin of your HTML page
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    plot,prediction = test_on_img(image)
    s = [str(i) for i in prediction] 
    a = int("".join(s)) 
    # print("Predicted traffic sign is: ", classes[a])
    return {"dimension": classes[a]}
