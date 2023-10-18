from flask import Flask, render_template, Response, request
from SASL_model import generate_frames

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed_route():
    return video_feed()

if __name__ == '__main__':
    app.run(debug=True)
