from flask import Flask, request, render_template, send_file, url_for, redirect, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from werkzeug.utils import secure_filename
import time
import sqlite3
from datetime import datetime, timedelta

app = Flask(__name__)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('parking.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spot_id TEXT NOT NULL,
            amount REAL NOT NULL,
            duration INTEGER NOT NULL,
            payment_time DATETIME NOT NULL,
            card_number TEXT NOT NULL,
            card_holder TEXT NOT NULL,
            transaction_id TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Call init_db when the application starts
init_db()

# Load the trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Dictionary to track parking spot states and their durations
parking_status = {}

# Time interval for tracking (10 seconds)
TIME_INTERVAL = 10

# Rate for billing (per session)
BILL_RATE = 20

# Payment status tracking
payment_status = {}

# Define image preprocessing function
def preprocess_image(image):
    image_resized = cv2.resize(image, (128, 128))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_normalized = image_gray / 255.0
    image_input = np.expand_dims(image_normalized, axis=-1)
    return np.expand_dims(image_input, axis=0)

# Prediction function
def predict_image(image):
    image = np.array(image)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return 'parked' if prediction > 0.5 else 'empty'

# Function to process image and predict parking spots
def extract_and_predict(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    cell_width = 100
    cell_height = 60

    columns_of_interest = [2, 3]

    overlay = image.copy()

    for y in range(0, height, cell_height):
        for x in range(0, width, cell_width):
            col_label = (x // cell_width) + 1
            if col_label in columns_of_interest:
                cropped_image = image[y:y + cell_height, x:x + cell_width]
                prediction = predict_image(cropped_image)
                color = (0, 255, 0) if prediction == 'empty' else (0, 0, 255)
                cv2.rectangle(overlay, (x, y), (x + cell_width, y + cell_height), color, -1)

    # Blend overlay with original image
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

    output_path = "static/highlighted_parking_image.png"
    cv2.imwrite(output_path, image)

    return output_path

@app.route('/')
def index():
    return render_template('index.html', active_page='home')

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/predict')
def predict():
    return render_template('predict.html', active_page='predict')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return redirect(url_for('predict'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('predict'))
    
    # Save the uploaded image
    filename = secure_filename(file.filename)  
    image_path = os.path.join('static', 'uploads', filename)
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    file.save(image_path)
    
    # Process the image and get predictions
    result_path = extract_and_predict(image_path)
    
    return render_template('predict.html', 
                         active_page='predict',
                         original_image=image_path,
                         processed_image=result_path)

@app.route("/download")
def download():
    return send_file("static/highlighted_parking_image.png", as_attachment=True)

@app.route('/video-analysis')
def video_analysis():
    return render_template('video_analysis.html', active_page='video')

@app.route('/initiate-payment/<spot_id>')
def initiate_payment(spot_id):
    # Get billing info for the spot
    spot_info = None
    for entry in payment_status.get(spot_id, []):
        if not entry.get('paid', False):
            spot_info = entry
            break
    
    if not spot_info:
        return jsonify({'error': 'No pending payment found for this spot'}), 404
    
    return render_template('payment.html', 
                         spot_id=spot_id,
                         duration=spot_info['duration'],
                         amount=spot_info['amount'],
                         active_page='video')

@app.route('/process-payment', methods=['POST'])
def process_payment():
    data = request.get_json()
    spot_id = data.get('spot_id')
    card_number = data.get('card_number')
    expiry = data.get('expiry')
    cvv = data.get('cvv')
    name = data.get('name')
    
    # Validate payment data
    if not all([spot_id, card_number, expiry, cvv, name]):
        return jsonify({'success': False, 'error': 'Missing payment information'})
    
    # Get billing info for the spot
    spot_info = None
    for entry in payment_status.get(spot_id, []):
        if not entry.get('paid', False):
            spot_info = entry
            break
    
    if not spot_info:
        return jsonify({'success': False, 'error': 'No pending payment found'})
    
    try:
        # Generate a unique transaction ID
        transaction_id = f"TXN-{int(time.time())}-{spot_id}"
        
        # Store payment in database
        conn = sqlite3.connect('parking.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO payments (
                spot_id, amount, duration, payment_time, 
                card_number, card_holder, transaction_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            spot_id,
            spot_info['amount'],
            spot_info['duration'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            card_number[-4:],  # Only store last 4 digits
            name,
            transaction_id
        ))
        conn.commit()
        conn.close()
        
        # Update payment status
        spot_info['paid'] = True
        spot_info['payment_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        spot_info['transaction_id'] = transaction_id
        
        return jsonify({
            'success': True,
            'message': 'Payment processed successfully',
            'receipt': {
                'transaction_id': transaction_id,
                'spot_id': spot_id,
                'amount': spot_info['amount'],
                'duration': spot_info['duration'],
                'payment_time': spot_info['payment_time']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected video'}), 400
    
    # Save the uploaded video
    video_filename = secure_filename(video_file.filename)
    video_path = os.path.join('static', 'uploads', video_filename)
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    video_file.save(video_path)
    
    # Process the video
    cap = cv2.VideoCapture(video_path)
    text_output = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % (TIME_INTERVAL * int(cap.get(cv2.CAP_PROP_FPS))) == 0:
            height, width = frame.shape[:2]
            cell_width = 100
            cell_height = 60
            rows = 'abcdefgh'
            current_status = []
            
            for y in range(0, height, cell_height):
                for x in range(0, width, cell_width):
                    col_label = (x // cell_width) + 1
                    row_label = rows[y // cell_height]
                    spot_label = f"{row_label}{col_label}"
                    
                    cropped_image = frame[y:y+cell_height, x:x+cell_width]
                    processed_image = preprocess_image(cropped_image)
                    prediction = model.predict(processed_image)
                    status = 'parked' if prediction > 0.5 else 'empty'
                    current_status.append(f"{spot_label}:{status}")
                    
                    # Track status and update time
                    if spot_label not in parking_status:
                        parking_status[spot_label] = {'state': status, 'time': time.time()}
                    else:
                        if parking_status[spot_label]['state'] == 'parked' and status == 'empty':
                            duration = int(time.time() - parking_status[spot_label]['time'])
                            bill = BILL_RATE
                            bill_text = f"{spot_label} {parking_status[spot_label]['state']}:{duration} Bill={bill} rs"
                            text_output.append(bill_text)
                            
                            # Track payment status
                            if spot_label not in payment_status:
                                payment_status[spot_label] = []
                            payment_status[spot_label].append({
                                'duration': duration,
                                'amount': bill,
                                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'paid': False
                            })
                            
                            parking_status[spot_label] = {'state': status, 'time': time.time()}
                        elif parking_status[spot_label]['state'] != status:
                            parking_status[spot_label] = {'state': status, 'time': time.time()}
            
            text_output.append(f"t={frame_count // (TIME_INTERVAL * int(cap.get(cv2.CAP_PROP_FPS)))} " + ', '.join(current_status))
        
        frame_count += 1
    
    cap.release()
    
    # Save the analysis results
    results_path = os.path.join('static', 'results', f'analysis_{video_filename}.txt')
    os.makedirs(os.path.join('static', 'results'), exist_ok=True)
    with open(results_path, 'w') as f:
        f.write('\n'.join(text_output))
    
    return jsonify({
        'success': True,
        'results_file': results_path,
        'text_output': text_output,
        'payment_status': payment_status
    })

@app.route('/payment-history')
def payment_history():
    conn = sqlite3.connect('parking.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    c = conn.cursor()
    
    c.execute('SELECT * FROM payments ORDER BY payment_time DESC')
    payments = [dict(row) for row in c.fetchall()]
    
    conn.close()
    return render_template('payment_history.html', payments=payments, active_page='payment_history')

@app.route('/api/payments')
def get_payments():
    spot = request.args.get('spot', '').lower()
    date_filter = request.args.get('date', 'all')
    
    conn = sqlite3.connect('parking.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    query = 'SELECT * FROM payments WHERE 1=1'
    params = []
    
    if spot:
        query += ' AND LOWER(spot_id) LIKE ?'
        params.append(f'%{spot}%')
    
    current_date = datetime.now().date()
    if date_filter == 'today':
        query += ' AND DATE(payment_time) = DATE(?)'
        params.append(current_date.isoformat())
    elif date_filter == 'week':
        week_ago = current_date - timedelta(days=7)
        query += ' AND DATE(payment_time) >= DATE(?)'
        params.append(week_ago.isoformat())
    elif date_filter == 'month':
        month_ago = current_date - timedelta(days=30)
        query += ' AND DATE(payment_time) >= DATE(?)'
        params.append(month_ago.isoformat())
    
    query += ' ORDER BY payment_time DESC'
    
    c.execute(query, params)
    payments = [dict(row) for row in c.fetchall()]
    
    conn.close()
    return jsonify({'payments': payments})

if __name__ == "__main__":
    app.run(debug=True)
