import os
import cv2
import face_recognition
import torch
import numpy as np


# 1. Load the YOLOv5 Model for Person Detection

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
model.conf = 0.3


# 2. Build the Authorized Personnel Database



authorized_encodings = {}  # Dictionary: {name: face_encoding}
authorized_faces_dir = 'Dataset'  # Folder with authorized personnel images

for filename in os.listdir(authorized_faces_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(authorized_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            name = os.path.splitext(filename)[0]
            authorized_encodings[name] = encodings[0]
            print(f"Loaded face encoding for: {name}")  # ✅ Debugging print
        else:
            print(f"⚠️ No face found in {filename}, skipping.")  # ✅ Debugging print

print(f"Total authorized faces loaded: {len(authorized_encodings)}")  # ✅ Debugging print


# 3. Set Up the Registry for Counted Persons



counted_persons = set()


# 4. Initialize Video Capture (Camera or Video File)



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
# 5. Process Each Frame in a Loop


while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received; exiting.")
        break

    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

    # Run YOLOv5 inference on the frame to detect objects.
    # Convert frame to RGB for YOLOv5
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # <-- ADD THIS LINE
    results = model(frame_rgb)  # <-- USE RGB FRAME

    detections = results.xyxy[0].cpu().numpy()

    print(f"Total objects detected: {len(detections)}")  # ✅ Debugging print

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        print(f"Detected object - Class: {int(cls)}, Confidence: {conf}")  # ✅ Debugging print

        if int(cls) != 0:
            continue
            # Skip small ROIs (e.g., distant people)

        height = y2 - y1
        width = x2 - x1
        if height < 20 or width < 20:  # <-- ADD THIS CHECK
         continue

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


        # 6. Face Detection & Recognition Within the Person ROI


        person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
        person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        if person_roi.size == 0:
            continue  # Skip if ROI is empty

        face_landmarks_list = face_recognition.face_locations(person_roi, model="hog")
        print(f"Faces found: {len(face_landmarks_list)}")  # ✅ Debugging print
        print("Face landmarks list:", face_landmarks_list)

        if face_landmarks_list:
            face_encodings = face_recognition.face_encodings(person_roi_rgb, face_landmarks_list)

            # Handle encoding extraction failures
            if not face_encodings:
                print("⚠️ Face detected but encoding failed.")
                continue

            # Process all faces in the ROI
            for (top, right, bottom, left), face_encoding in zip(face_landmarks_list, face_encodings):  # <-- LOOP HERE
                # Use face distances instead of boolean matches
                face_distances = face_recognition.face_distance(
                    list(authorized_encodings.values()), face_encoding
                )

                # Find the best match
                best_match_idx = np.argmin(face_distances)
                if face_distances[best_match_idx] < 0.5:  # Adjust threshold as needed
                    name = list(authorized_encodings.keys())[best_match_idx]
                else:
                    name = "Unknown"

                # Check if authorized person is new
                if name != "Unknown" and name not in counted_persons:
                    counted_persons.add(name)
                    print(f"✅ Authorized person counted: {name}")

                # Draw annotations (coordinates already adjusted in your code)
                # ... (your existing drawing code remains here)
            else:
                print("❌ No match found.")


                # Check if this authorized person has already been counted.
                if name not in counted_persons:
                    counted_persons.add(name)
                    print(f"Authorized person counted: {name}")


            # 7. Annotate the Frame with Recognition Results


            # Draw face bounding boxes and label with the person's name.
            for (top, right, bottom, left) in face_landmarks_list:
                # Adjust the coordinates from the ROI to the full frame.
                top_global = int(y1) + top
                right_global = int(x1) + right
                bottom_global = int(y1) + bottom
                left_global = int(x1) + left
                cv2.rectangle(frame, (left_global, top_global), (right_global, bottom_global), (0, 255, 0), 2)
                cv2.putText(frame, name, (left_global, top_global - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # 8. Display the Current Head Count on the Frame


    head_count = len(counted_persons)
    cv2.putText(frame, f"Head Count: {head_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the processed video frame.
    cv2.imshow("Authorized Personnel Head Count", frame)

    # Break the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# 9. Cleanup Resources

cap.release()
cv2.destroyAllWindows()