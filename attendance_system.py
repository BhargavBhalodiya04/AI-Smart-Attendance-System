# attendance_system.py

import cv2
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known embeddings
def load_known_embeddings(data_dir='registered_faces'):
    known_embeddings = []
    known_names = []

    for name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, name)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face, _ = mtcnn(img_rgb, return_prob=True)
            if face is not None:
                embedding = resnet(face.to(device)).detach().cpu().numpy()
                known_embeddings.append(embedding)
                known_names.append(name)

    return np.vstack(known_embeddings), known_names

known_embeddings, known_names = load_known_embeddings()

# Save attendance to Excel
def save_attendance(marked_names):
    if not marked_names:
        return "No known faces detected."

    date_str = datetime.now().strftime('%d-%m-%Y')
    time_str = datetime.now().strftime('%H:%M:%S')
    er_numbers, clean_names = [], []

    for full_name in marked_names:
        if '_' in full_name:
            er, name = full_name.split('_', 1)
        else:
            er, name = "Unknown", full_name
        er_numbers.append(er)
        clean_names.append(name)

    df = pd.DataFrame({
        'ER Number': er_numbers,
        'Name': clean_names,
        'Date': [date_str] * len(clean_names),
        'Time': [time_str] * len(clean_names)
    })

    attendance_path = f'Attendance/attendance_{date_str}.xlsx'
    if os.path.exists(attendance_path):
        os.remove(attendance_path)

    df.to_excel(attendance_path, index=False)
    return f"Attendance marked for: {clean_names}"

# 1. Attendance from image(s)
def mark_attendance_live():
    marked_names = []
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(frame_rgb)
        faces = mtcnn(frame_rgb)

        if faces is not None and boxes is not None:
            for face_tensor, box in zip(faces, boxes):
                embedding = resnet(face_tensor.unsqueeze(0).to(device)).detach().cpu().numpy()
                distances = np.linalg.norm(known_embeddings - embedding, axis=1)
                min_index = np.argmin(distances)
                if distances[min_index] < 0.9:
                    name = known_names[min_index]
                    if name not in marked_names:
                        marked_names.append(name)

                    x1, y1, x2, y2 = [int(i) for i in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return save_attendance(marked_names)
# 2. Attendance from image(s)
def mark_attendance_from_images(image_paths):
    marked_names = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(img_rgb)
        faces = mtcnn(img_rgb)

        if faces is not None and boxes is not None:
            for face_tensor, box in zip(faces, boxes):
                embedding = resnet(face_tensor.unsqueeze(0).to(device)).detach().cpu().numpy()
                distances = np.linalg.norm(known_embeddings - embedding, axis=1)
                min_index = np.argmin(distances)
                if distances[min_index] < 0.9:
                    name = known_names[min_index]
                    if name not in marked_names:
                        marked_names.append(name)

    return save_attendance(marked_names)



from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os

def generate_attendance_pdf(attendance_list, filename="attendance_report.pdf"):
    pdf_path = os.path.join("reports", filename)
    os.makedirs("reports", exist_ok=True)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Attendance Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawString(50, height - 100, "Name")
    c.drawString(250, height - 100, "Enrollment No")
    c.drawString(450, height - 100, "Status")

    y = height - 120
    for name, er_no in attendance_list:
        c.drawString(50, y, name)
        c.drawString(250, y, er_no)
        c.drawString(450, y, "Present")
        y -= 20

    c.save()
    return pdf_path
