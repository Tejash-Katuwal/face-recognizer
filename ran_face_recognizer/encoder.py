import os
import cv2
import dlib
import numpy as np
import requests
import bz2
import shutil

class FaceEncoder:
    def __init__(self, model_path="dlib_face_recognition_resnet_model_v1.dat", shape_predictor_path="shape_predictor_68_face_landmarks.dat"):
        model_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define paths for the models
        model_path = os.path.join(model_dir, model_path)
        shape_predictor_path = os.path.join(model_dir, shape_predictor_path)

        # Check if the recognition model exists, if not, download it
        if not os.path.exists(model_path):
            print("Face recognition model not found locally. Downloading and decompressing...")
            self.download_and_decompress_model(model_path, "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
        
        # Check if the shape predictor exists, if not, download it
        if not os.path.exists(shape_predictor_path):
            print("Shape predictor model not found locally. Downloading and decompressing...")
            self.download_and_decompress_model(shape_predictor_path, "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        
        # Initialize the recognition model and face detector
        self.recognition_model = dlib.face_recognition_model_v1(model_path)
        self.detector = dlib.get_frontal_face_detector()

        # Initialize the shape predictor
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def download_and_decompress_model(self, model_path, url):
        """
        Downloads and decompresses a model from the provided URL if it's not found locally.
        
        Args:
            model_path (str): Path where the model should be saved.
            url (str): The URL to download the model from.
        """
        try:
            # Check if the compressed model already exists
            compressed_model_path = model_path + ".bz2"
            
            # Download the model if it doesn't exist
            if not os.path.exists(compressed_model_path):
                print(f"Downloading model from {url}...")
                response = requests.get(url)
                if response.status_code == 200:
                    with open(compressed_model_path, "wb") as file:
                        file.write(response.content)
                    print("Model downloaded successfully.")
                else:
                    raise Exception(f"Failed to download the model. Status code: {response.status_code}")

            # Decompress the downloaded model
            self.decompress_model(compressed_model_path, model_path)
            os.remove(compressed_model_path)  # Remove the compressed file after extraction
            print(f"Model saved to {model_path} and ready for use.")

        except Exception as e:
            print(f"Error downloading or decompressing model: {e}")
            raise

    def decompress_model(self, compressed_model_path, model_path):
        """Decompresses the downloaded model from bz2 format to the original model path."""
        try:
            with bz2.BZ2File(compressed_model_path) as f_in, open(model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            print(f"Error decompressing model: {e}")
            raise

    def encode_face(self, image, face):
        """
        Generate a 128-dimensional encoding for a detected face.
        
        Args:
            image (numpy array): The input image.
            face (dlib.rectangle): The bounding box of the detected face.
        
        Returns:
            numpy array: A 128-dimensional encoding of the face.
        """
        shape = self.shape_predictor(image, face)
        # Pass both the face rectangle (bounding box) and the shape (landmarks)
        full_face = dlib.full_object_detection(face, shape.parts())
        return np.array(self.recognition_model.compute_face_descriptor(image, full_face))

    def encode_faces(self, image, faces):
        """
        Generate encodings for multiple faces in an image.
        
        Args:
            image (numpy array): The input image.
            faces (list): A list of detected faces (bounding boxes).
        
        Returns:
            list: A list of 128-dimensional encodings.
        """
        return [self.encode_face(image, face) for face in faces]

    def encode_faces_from_folder(self, folder_path):
        """
        Process a folder of images, detect faces, and generate encodings.
        
        Args:
            folder_path (str): The path to the folder containing images.
        
        Returns:
            list: A list of tuples (encoding, name), where 'encoding' is a 128-dimensional numpy array
                  and 'name' is the name associated with the image.
        """
        known_encodings = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                faces = self.detector(image)

                encodings = self.encode_faces(image, faces)
                name = os.path.splitext(filename)[0]

                for encoding in encodings:
                    known_encodings.append((encoding, name))

        return known_encodings
