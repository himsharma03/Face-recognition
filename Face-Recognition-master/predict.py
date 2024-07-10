import os
import face_recognition_api
import pickle
import numpy as np
import pandas as pd

def get_prediction_images(prediction_dir):
    files = [x[2] for x in os.walk(prediction_dir)][0]
    l = []
    exts = [".jpg", ".jpeg", ".png"]
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in exts:
            l.append(os.path.join(prediction_dir, file))
    return l

def predict_faces_in_images(prediction_dir, classifier_file):
    with open(classifier_file, 'rb') as f:
        (le, clf) = pickle.load(f)

    for image_path in get_prediction_images(prediction_dir):
        print('\x1b[6;30;42m' + f"=====Predicting faces in '{image_path}'=====" + '\x1b[0m')

        img = face_recognition_api.load_image_file(image_path)
        X_faces_loc = face_recognition_api.face_locations(img)

        faces_encodings = face_recognition_api.face_encodings(img, known_face_locations=X_faces_loc)
        print(f"Found {len(faces_encodings)} faces in the image")

        if len(faces_encodings) == 0:
            print("No faces found in the image. Skipping.")
            continue

        closest_distances = clf.kneighbors(faces_encodings, n_neighbors=1)
        is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(X_faces_loc))]

        predictions = clf.predict(faces_encodings)

        # Ensure predictions are made as a 1-dimensional array
        if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
            predictions = predictions.ravel()

        try:
            predictions = [(le.inverse_transform([int(pred)])[0].title(), loc) if rec else ("Unknown", loc)
                           for pred, loc, rec in zip(predictions, X_faces_loc, is_recognized)]
        except ValueError as e:
            print(f"Error in inverse_transform: {e}")

        print(predictions)
        print()

if __name__ == "__main__":
    prediction_dir = './test-images'
    classifier_file = "./classifier.pkl"
    predict_faces_in_images(prediction_dir, classifier_file)
