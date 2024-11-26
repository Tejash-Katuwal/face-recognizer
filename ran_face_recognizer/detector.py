import dlib
import cv2

class FaceDetector:
    def __init__(self):
        # Initialize dlib's face detector
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, image):
        """
        Detect faces in the given image.
        
        Args:
            image (numpy array): Input image (from OpenCV).
        
        Returns:
            list: A list of dlib rectangles representing faces.
        """
        # Convert the image to grayscale because dlib works with grayscale images
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray_image)
        return faces
