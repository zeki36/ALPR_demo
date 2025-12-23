import cv2
import os
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import time


ISLEM_MODU = 'IMAGE' # ÇALIŞTIRMADAN ÖNCE BAKKKKKKK (IMAGE/VIDEO)

TEST_KLASORU = '/Users/iremsen/Desktop/ALPR_DEMO/TEST'
SONUC_KLASORU = '/Users/iremsen/Desktop/ALPR_DEMO/sonuclar'
VIDEO_GIRIS_YOLU = '/Users/iremsen/Desktop/ALPR_DEMO/trafik_video.mp4'
VIDEO_CIKIS_YOLU = '/Users/iremsen/Desktop/ALPR_DEMO/sonuclar/islenmis_video.mp4'
MODEL_YOLU = 'plate_detector.pt'

PADDING = 10
MIN_TEXT_RATIO = 0.6 

# Klasörü oluştur
os.makedirs(SONUC_KLASORU, exist_ok=True)

model = YOLO(MODEL_YOLU)
reader = easyocr.Reader(['en'], gpu=True)

dict_int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
dict_char_to_int = {'O': '0', 'D': '0', 'I': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'E': '3'}

def clean_plate(text):
    return re.sub(r'[^A-Z0-9]', '', text)

def remove_tr_prefix(text):
    return re.sub(r'^(TR|T|R)(?=\d)', '', text)

def preprocess_plate(img):
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    return gray

def smart_fix_plate(text):
    text = remove_tr_prefix(text)
    text_list = list(text)
    n = len(text_list)
    if n < 5: return text

    for i in range(2):
        if i < n and text_list[i] in dict_char_to_int:
            text_list[i] = dict_char_to_int[text_list[i]]
    for i in range(n - 2, n):
        if i >= 0 and text_list[i] in dict_char_to_int:
            text_list[i] = dict_char_to_int[text_list[i]]

    return "".join(text_list)

def process_frame(frame):
    """
    Bu fonksiyon bir resim karesi alır, plakaları bulur, okur, çizer ve geri döndürür.
    """
    h_img, w_img = frame.shape[:2]
    
    # YOLO Tespiti
    results = model(frame, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(w_img, x2 + PADDING)
            y2 = min(h_img, y2 + PADDING)
            
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0: continue

            processed_plate = preprocess_plate(plate_crop)
            
            ocr_result = reader.readtext(processed_plate, detail=1, 
                                         allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
            if ocr_result:
                max_h = 0
                for (bbox, text, conf) in ocr_result:
                    (tl, tr, br, bl) = bbox
                    h = bl[1] - tl[1]
                    if h > max_h: max_h = h
                
                text_parts = []
                conf_scores = []
                
                for (bbox, text, conf) in ocr_result:
                    (tl, tr, br, bl) = bbox
                    h = bl[1] - tl[1]
                    # Ana yazı boyutunun %60'ından büyükleri al
                    if h > max_h * MIN_TEXT_RATIO:
                        text_parts.append(text)
                        conf_scores.append(conf)

                raw_text = "".join(text_parts).upper().replace(" ", "")
                clean_text = clean_plate(raw_text)
                
                if conf_scores:
                    avg_conf = sum(conf_scores) / len(conf_scores)
                else:
                    avg_conf = 0.0
                
                final_text = smart_fix_plate(clean_text)
                
                color = (0, 255, 0) if avg_conf > 0.5 else (0, 255, 255)
                label = f"{final_text} %{avg_conf*100:.1f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w_text, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                
    return frame

#-----------------------------------------

if ISLEM_MODU == 'IMAGE':
    print(f"--- FOTOĞRAF MODU ---")
    print(f"Klasör: {TEST_KLASORU}")
    
    count = 0
    for filename in os.listdir(TEST_KLASORU):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(TEST_KLASORU, filename)
            frame = cv2.imread(path)
            
            if frame is None: continue
            
            # İşlemi yap
            processed_frame = process_frame(frame)
            
            # Kaydet
            save_path = os.path.join(SONUC_KLASORU, f"sonuc_{filename}")
            cv2.imwrite(save_path, processed_frame)
            print(f"Kaydedildi: {filename}")
            count += 1
            
    print(f"\nToplam {count} fotoğraf işlendi.")

elif ISLEM_MODU == 'VIDEO':
    print(f"--- VİDEO MODU ---")
    print(f"Video: {VIDEO_GIRIS_YOLU}")
    
    cap = cv2.VideoCapture(VIDEO_GIRIS_YOLU)
    if not cap.isOpened():
        print("HATA: Video dosyası açılamadı.")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Video kaydedici
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_CIKIS_YOLU, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        processed_frame = process_frame(frame)
        
        out.write(processed_frame)
        
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"İşlenen Kare: {frame_count} (Geçen süre: {elapsed:.1f} sn)")
            
    cap.release()
    out.release()
    print(f"\nVideo işlemi tamamlandı.")
    print(f"Dosya şuraya kaydedildi: {VIDEO_CIKIS_YOLU}")

else:
    print("HATA: ISLEM_MODU sadece 'IMAGE' veya 'VIDEO' olabilir.")
