import cv2
from gpiozero import Buzzer, MotionSensor, LED
from flask import Flask, render_template_string
import threading
import time

motion_sensor = MotionSensor(16)
status_led = LED(21)            
buzzerPin = Buzzer(20)         


app = Flask(__name__)
seat_status = "자리 비움"
drowsy_count = 0

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>열람실 좌석 모니터링 시스템</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="2">
</head>
<body style="text-align: center; font-family: Arial, sans-serif; margin-top: 50px;">
    <h1>시험기간 열람실 좌석 실시간 대시보드</h1>
    <hr width="50%">
    <div style="margin: 30px; padding: 20px; border: 2px solid #ccc; display: inline-block; border-radius: 10px;">
        <h2>현재 좌석 상태: <span style="color: blue;">{{ status }}</span></h2>
        <h2>실시간 졸음 감지 횟수: <span style="color: red;">{{ count }}회</span></h2>
    </div>
    <p>※ 본 페이지는 실시간으로 열람실 좌석, 집중도를 모니터링합니다.</p>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, status=seat_status, count=drowsy_count)

def monitor_system():
    global seat_status, drowsy_count
    
    camera = cv2.VideoCapture(-1)
    camera.set(3, 640)
    camera.set(4, 480)
    
    face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_xml = cv2.data.haarcascades + 'haarcascade_eye.xml'
    face_cascade = cv2.CascadeClassifier(face_xml)
    eye_cascade = cv2.CascadeClassifier(eye_xml)
    
    print("시스템 모니터링 루프를 시작합니다.")
    
    try:
        while camera.isOpened():
            if not motion_sensor.motion_detected:
                seat_status = "자리 비움"
                status_led.off()
                buzzerPin.off()
                
                _, image = camera.read()
                cv2.imshow('Library Seat Monitor', image)
                if cv2.waitKey(1) == ord('q'):
                    break
                time.sleep(0.5)
                continue
            
            seat_status = "이용 중"
            
            _, image = camera.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE)
            print("faces detected Number: " + str(len(faces)))

            if len(faces):
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    

                    face_gray = gray[y:y+h, x:x+w]
                    face_color = image[y:y+h, x:x+w]
                    
                    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)
                   
                    if len(eyes) <= 1:
                        buzzerPin.on()
                        status_led.on()
                        seat_status = "졸음 및 집중력 저하 감지!"
                        drowsy_count += 1
                    else:
                        buzzerPin.off()
                        status_led.off()
                    
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            else:
                buzzerPin.off()
                status_led.off()
            
            cv2.imshow('Library Seat Monitor', image)
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"에러 발생: {e}")
        
    finally:
        camera.release()
        cv2.destroyAllWindows()
        buzzerPin.off()
        status_led.off()
        print("모니터링 시스템이 종료되었습니다.")

if __name__ == '__main__':
    video_thread = threading.Thread(target=monitor_system)
    video_thread.daemon = True
    video_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False)