import numpy as np

class FaceRecognizer:
    def __init__(self, threshold=0.6):
        """
        Initialize the recognizer with a customizable matching threshold.
        
        Args:
            threshold (float): The threshold for face recognition, lower values make it more strict.
        """
        self.threshold = threshold

    def compare_faces(self, known_encodings, unknown_encoding):
        """
        Compare the known face encodings with an unknown encoding.
        
        Args:
            known_encodings (list): List of tuples (encoding, name) for known faces.
            unknown_encoding (numpy array): Encoding of the unknown face.
        
        Returns:
            bool, int: Whether a match was found and the name of the matched person.
        """
        # Calculate the Euclidean distance between the unknown encoding and each known encoding
        distances = np.linalg.norm([encoding[0] for encoding in known_encodings] - unknown_encoding, axis=1)
        min_distance = np.min(distances)
        match_index = np.argmin(distances)

        # Check if the minimum distance is below the threshold
        if min_distance < self.threshold:
            return True, known_encodings[match_index][1]  # Return the name of the matched person
        else:
            return False, "Unknown"
