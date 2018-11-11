import numpy as np
import cv2
import spotipy
import webbrowser
import time
from spotipy.oauth2 import SpotifyClientCredentials
from keras.preprocessing import image

# -----------------------------
# opencv initialization

face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
# -----------------------------
# face expression recognizer initialization
from keras.models import model_from_json

model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5')  # load weights

# -----------------------------

mins = 0
client_credentials_manager = SpotifyClientCredentials(client_id='c05f99ee0a2d4c059c835f8f4cbe2c73', client_secret='596ed849c5a44891af5fdd1f74ba8365')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

playlists = sp.user_playlists('li006zvua0ttqvdzzns2gwtu5') #My User ID/Name

def show_tracks(tracks):
    for i, item in enumerate(tracks['items']):
        track = item['track']
        print("   %d %32.32s %s" % (i, track['artists'][0]['name'], track['name']))



emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while (True):
    ret, img = cap.read()
    # img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # print(faces) #locations of detected faces

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle to main image

        detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

        predictions = model.predict(img_pixels)  # store probabilities of 7 expressions

        # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])

        emotion = emotions[max_index]

        print(emotion)

        # write emotion text above rectangle
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        while mins != 2:
            for playlist in playlists['items']:
                if playlist['owner']['id'] == 'li006zvua0ttqvdzzns2gwtu5':
                    if playlist['name'] == emotion:
                        print(playlist['external_urls']['spotify'])
                        webbrowser.open(playlist['external_urls']['spotify'])
                        print('  total tracks', playlist['tracks']['total'])
                        results = sp.user_playlist('li006zvua0ttqvdzzns2gwtu5', playlist['id'], fields="tracks,next")
                        tracks = results['tracks']
                        show_tracks(tracks)
                    break
                break
            time.sleep(60)
            mins += 1

    # process on detected face end
    # -------------------------

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

# kill open cv things
cap.release()
cv2.destroyAllWindows()