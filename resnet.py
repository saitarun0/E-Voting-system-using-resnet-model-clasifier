
from flask import Flask, render_template, request, redirect, url_for, session,flash
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from flask_session import Session 
import sqlite3
from cv2 import *
import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
import tensorflow
import dlib
import random
import smtplib
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from tqdm import tqdm
from sklearn.utils import shuffle
from encrypt import encrypt
from encrypt import decrypt

app = Flask(__name__)
app.secret_key = "12345678989"
access=["no"]
PRIMARY_DATABASE="primary2.db"
DATABASE = 'secondary2.db'

def create_table():
    try:
        conn = sqlite3.connect(DATABASE)    
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT NOT NULL ,
                          email TEXT NOT NULL,
                          voter_id TEXT NOT NULL)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS login (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT NOT NULL ,
                          email TEXT NOT NULL,
                          voter_id TEXT NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS nominees (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                nominee_name TEXT,zone TEXT,
                                voter_id INTEGER ) ''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS  shedule1 (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                voter_id TEXT,
                                 shedule INTEGER ) ''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS votes (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                nominee_id INTEGER,
                                FOREIGN KEY (nominee_id) REFERENCES nominees (id))''')
        conn.commit()
        conn = sqlite3.connect(PRIMARY_DATABASE)    
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT NOT NULL ,
                          email TEXT NOT NULL,
                          voter_id TEXT NOT NULL)''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print("Error occurred:", e)
    finally:
        if conn:
            conn.close()
create_table()






    


@app.route('/')
@app.route('/index')
def index():
        return render_template('index.html')
@app.route('/user', methods=['GET', 'POST'])
def user():
        return render_template('user.html')
@app.route('/success')
def success():
        return render_template('success.html')
@app.route('/complete')
def complete():
        return render_template('complete.html')



password = '123456'
user=[]
voter_id=[]
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username=request.form['username']
        user.append(username)
        voter = request.form['voter']
        voter_id .append(voter)
        v1=str(voter)
        if len(v1)!=6:
            return  " please enter valid voter id "
        email = request.form['email']
        confirm_email = request.form['confirm_email']
        if email != confirm_email:
            return " your email does not match"
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE email=?", (email,))
        registered = cursor.fetchall()
        cursor.execute("SELECT voter_id  FROM users WHERE voter_id=?", (voter,))
        registered1 = cursor.fetchall()
        if registered:
            return " your email already registered"
        elif registered1:
            return  "your voter_id  already registered"
        else:
            username1 = encrypt(username, password)
            voter1=encrypt(voter, password)
            email1=encrypt(email, password)
            cursor.execute("INSERT INTO users (username, email,voter_id ) VALUES (?, ?,?)", (username, email,voter))
            conn.commit()
            conn_primary = sqlite3.connect(PRIMARY_DATABASE)
            cursor_primary = conn_primary.cursor()
            cursor_primary.execute("INSERT INTO users (username, email, voter_id) VALUES (?, ?, ?)", (username1, email1, voter1))
            conn_primary.commit()
            conn_primary.close()
            return render_template('data.html')
    return render_template('register.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        username = user[-1]
        voter_id1 = voter_id[-1]
        cam = cv2.VideoCapture(0) 

        if not cam.isOpened():
            print("Failed to open webcam.")
        time.sleep(5)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()

            if not ret:
                #print("Failed to capture frame from webcam.")
                return "Failed to capture frame from webcam."

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1

                # Save the image in a folder named after the voter_id
                folder_path = os.path.join('TrainingImages', str(voter_id1))
                os.makedirs(folder_path, exist_ok=True)
                image_path = os.path.join(folder_path, f"{voter_id1}_{sampleNum}.jpg")
                cv2.imwrite(image_path, gray[y:y + h, x:x + w])

                cv2.imshow('frame', img)

            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 30:
                break

        cam.release()
        cv2.destroyAllWindows()
        
    return render_template('success.html')

@app.route('/train', methods=['GET'])
def train():
    return render_template('train.html')

def recognize_faces():
    detector = dlib.get_frontal_face_detector()
    training_data_folder = 'TrainingImages'
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    face_encodings = []
    labels = []
    for person_name in os.listdir(training_data_folder):
        person_folder = os.path.join(training_data_folder, person_name)
        if os.path.isdir(person_folder):
            person_id = int(person_name.replace('person', ''))          
            for filename in os.listdir(person_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_folder, filename)
                    image = cv2.imread(image_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)

                    for face in faces:
                        shape = predictor(gray, face)
                        face_encoding = face_recognizer.compute_face_descriptor(image, shape)
                        face_encodings.append(face_encoding)
                        labels.append(person_id)
    labels = np.array(labels)
    face_encodings = np.array(face_encodings)
    return "succuess fully train"

@app.route('/training', methods=['POST'])
def training():
    recognize_faces()
    return render_template('complete.html')



u=[]
p=[]
a=[]
l=[]
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if access[-1]=="yes":                                        
                username = request.form['username']
                u.append(username)      
                email = request.form['email']
                p.append(email)
                voter_id  = request.form['voter']
                #print(voter_id ,":voter_id ")
                a.append(voter_id )
                conn = sqlite3.connect(DATABASE)
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM login WHERE voter_id =? AND email=?", (voter_id ,email))
                user1 = cursor.fetchone()
                if user1:
                    return display_popup("You have already voted  Don't cheat")
                else:
                    cursor.execute("SELECT * FROM users WHERE voter_id =? AND email=?", (voter_id , email))
                    user = cursor.fetchone()
                    if user:
                        detector = dlib.get_frontal_face_detector()
                        training_data_folder = 'TrainingImages'
                        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                        face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
                        face_encodings = []
                        labels = []
                        for person_name in os.listdir(training_data_folder):
                            person_folder = os.path.join(training_data_folder, person_name)
                            if os.path.isdir(person_folder):
                                person_id = int(person_name.replace('person', ''))          
                                for filename in os.listdir(person_folder):
                                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                                        image_path = os.path.join(person_folder, filename)
                                        image = cv2.imread(image_path)
                                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                        faces = detector(gray)

                                        for face in faces:
                                            shape = predictor(gray, face)
                                            face_encoding = face_recognizer.compute_face_descriptor(image, shape)
                                            face_encodings.append(face_encoding)
                                            labels.append(person_id)
                        labels = np.array(labels)
                        face_encodings = np.array(face_encodings)
                        cap = cv2.VideoCapture(0)

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = detector(gray)
                            for face in faces:
                                shape = predictor(gray, face)
                                face_encoding = face_recognizer.compute_face_descriptor(frame, shape)
                                distances = np.linalg.norm(face_encodings - face_encoding, axis=1)
                                min_distance_idx = np.argmin(distances)
                                min_distance = distances[min_distance_idx]
                                #print(min_distance)
                                if min_distance < 0.5:
                                        label = labels[min_distance_idx]
                                        name1=str(label)
                                        voter_id1=str(voter_id)
                                        #print(voter_id1)
                                        if (name1==voter_id1):
                                            #print("after",name1,voter_id1)
                                            cursor.execute("SELECT email FROM users WHERE voter_id =? AND email=?", (voter_id , email))
                                            user2 = cursor.fetchone() 
                                            #print(user2)
                                            smtp_server = 'smtp.example.com'
                                            smtp_port = 587
                                            sender_email = 'diwa.2801@gmail.com'
                                            sender_password = 'furgqbokcooqfjkf'
                                            receiver_email = user2
                                            otp = ""
                                            for _ in range(6):
                                                otp += str(random.randint(0, 9))
                                                l.append(otp)
                                            print(l[-1])
                                            host = "smtp.gmail.com"
                                            mmail = "diwa.2801@gmail.com"        
                                            hmail = user2[0]
                                            receiver_name = username
                                            sender_name= "election commision "
                                            msg = MIMEMultipart()
                                            subject = "YOUR OTP CODE"
                                            text =  f'Your OTP code is: {otp}'
                                            msg = MIMEText(text, 'plain')
                                            msg['To'] = formataddr((receiver_name, hmail))
                                            msg['From'] = formataddr((sender_name, mmail))
                                            msg['Subject'] = 'Hello  ' + receiver_name
                                            server = smtplib.SMTP(host, 587)
                                            server.ehlo()
                                            server.starttls()
                                            password = " furgqbokcooqfjkf"
                                            server.login(mmail, password)
                                            server.sendmail(mmail, [hmail], msg.as_string())
                                            server.quit()
                                            #print('send')
                                            return redirect(url_for('OTP'))
                                            cam.release()
                                            cv2.destroyAllWindows()
                                            break
                                        else: 
                                            return display_popup("User face mismatch")
                                else:
                                       print("outof range")
                    else:
                        return display_popup("Not registered")
        else:
             return display_popup( "web not working")
    return render_template('login.html')




def display_popup2(message):
    flash(message)
    return redirect(url_for('index'))
def display_popup1(message):
    flash(message)
    return redirect(url_for('register'))
def display_popup(message):
    flash(message)
    return redirect(url_for('login'))

def display_popup3(message):
    flash(message)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
                return render_template('dashboard.html')


@app.route('/OTP', methods=['GET', 'POST'])
def OTP():
    if request.method == 'POST':
        otp1 = request.form['otp']
        if otp1==l[-1]:
            l.clear()
            return render_template('dashboard.html')
        else:
            print("not valid")
    return render_template('OTP.html')

@app.route('/shedule', methods=['GET', 'POST'])
def shedule():
    if request.method == 'POST':
        time = request.form["choices"]
        voter_id = int(a[0])
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT shedule  FROM shedule1 WHERE voter_id=?', (voter_id,))
        user = cursor.fetchone()
        #print(user)
        if user:
            a.clear()
            return display_popup2("Your schedule is already registered")
        else:
            cursor.execute('SELECT COUNT(shedule) FROM shedule1 WHERE shedule=?', (time,))
            user = cursor.fetchone()
            if user[0]>=5:
                    return display_popup3("already more than 5 users selected this shedule ")
            else:
                    cursor.execute('INSERT INTO shedule1 (voter_id, shedule) VALUES (?, ?)', (voter_id, time))
                    conn.commit()
                    a.clear()
                    return display_popup2("Your schedule has been successfully registered")

  


@app.route('/voting', methods=['GET', 'POST'])
def voting_form():
    username=u[0]
    email=p[0]
    voter_id =a[0]
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT shedule  FROM shedule1 WHERE voter_id=?', (voter_id,))
    data=cursor.fetchone()
    if data:
        cursor.execute('SELECT * FROM nominees')
        nominees = cursor.fetchall()

        cursor.execute('SELECT nominee_id, COUNT(id) as vote_count FROM votes GROUP BY nominee_id')
        votes = {nominee_id: vote_count for nominee_id, vote_count in cursor.fetchall()}
        if request.method == 'POST':
            nominee_id = int(request.form['nominee'])
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO votes (nominee_id) VALUES (?)', (nominee_id,))
            cursor.execute("INSERT INTO login (username,email,voter_id ) VALUES (?, ?,?)", (username, email,voter_id ))
            conn.commit()
            conn.close()
            p.clear()   
            u.clear()
            a.clear()
            return display_popup2(" your vote has been sucessfully registered")
        return render_template('voting_form.html', nominees=nominees, votes=votes)
    else:
        return display_popup3(" you are not registered schedule ")


@app.route('/result1')
def result1():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT nominees.nominee_name, COUNT(votes.id) as vote_count
        FROM nominees LEFT JOIN votes ON nominees.id = votes.nominee_id
        GROUP BY nominees.nominee_name
        ORDER BY vote_count DESC
    ''')
    result = cursor.fetchall()
    conn.close()
    #print(result)

    max_value = max(result, key=lambda x: x[1])[1]

    winners = [tuple for tuple in result if tuple[1] == max_value]

    if len(winners) > 1:
        winner="The Election is Draw"
        #print(winner)
    else:
        winner=winners[0][0]
        #print(winner)   
    return render_template('result1.html', result=result, winner=winner)


ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin'

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            return redirect('/details')
    return render_template('admin.html')


#print(a)
@app.route('/details',methods=['GET', 'POST'])
def details():
    if request.method == 'POST':
        agree = request.form.get('agree')
        #print(agree)
        access.append(agree)
        #print(access)
    
    return render_template('details.html')


def get_table_data3():
    conn = sqlite3.connect(PRIMARY_DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    table_data = cursor.fetchall()
    conn.close()
    return table_data

@app.route('/table')
def table():
    table_data = get_table_data3()
    return render_template('table.html', table_data=table_data)





def get_table_data1():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nominees")
    table_data = cursor.fetchall()
    conn.close()
    return table_data

@app.route('/nomine_list')
def nomine_list():
    table_data = get_table_data1()
    return render_template('nomine_list.html', table_data=table_data)


def get_table_data5():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM login")
    table_data = cursor.fetchall()
    conn.close()
    return table_data

@app.route('/voted_ist')
def voted_ist():
    table_data = get_table_data5()
    return render_template('voted_ist.html', table_data=table_data)




@app.route('/nomine', methods=['GET', 'POST'])
def nominee_form():
    if request.method == 'POST':
        nominee_name = request.form['nominee']
        zone = request.form['zone']
        voter_id = request.form['voter']
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO nominees (nominee_name,zone,voter_id) VALUES (?,?,?)', (nominee_name,zone,voter_id))
        conn.commit()
        conn.close()
        return redirect(url_for('nominee_form'))
    return render_template('nominee_form.html')



@app.route('/recieve', methods=['GET', 'POST'])
def recieve():
    if request.method == 'POST':
        id1= request.form['number']
        password= request.form['pass']
        conn = sqlite3.connect(PRIMARY_DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users where id=?",(id1,))
        table_data1= cursor.fetchone()
        name=decrypt(table_data1[1], password)
        email=decrypt(table_data1[2], password)
        voter_id=decrypt(table_data1[3], password)
        return render_template("decrypt.html",id1=table_data1[0],name=name,email=email,voter_id=voter_id)
        
        



    

if __name__ == '__main__':
    app.run(debug=False,port=300)
