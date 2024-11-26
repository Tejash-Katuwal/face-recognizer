from ran_face_recognizer import FaceDetector, FaceEncoder, FaceRecognizer
import cv2

# Initialize components
detector = FaceDetector()
encoder = FaceEncoder()
recognizer = FaceRecognizer(threshold=0.45)

# Step 1: Process a folder of images to generate known encodings
folder_path = 'D:\\Programming\\ran-face-recognizer\\ran-face-recognizer\\images'
known_encodings = encoder.encode_faces_from_folder(folder_path)

# Step 2: Start real-time video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    for face in faces:
        # Generate encoding for the detected face
        encoding = encoder.encode_face(frame, face)

        # Compare the encoding with known encodings
        match, name = recognizer.compare_faces(known_encodings, encoding)

        # Draw rectangle around face
        (top, right, bottom, left) = (face.top(), face.right(), face.bottom(), face.left())
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the name if matched, else "Unknown"
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
