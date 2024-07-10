import dlib
import numpy as np
from PIL import Image

# Initialize dlib face detector, shape predictor, and face recognition model
face_detector = dlib.get_frontal_face_detector()

# Path to shape predictor model
predictor_model = 'C:/Users/LENOVO/Downloads/Face-Recognition-master/Face-Recognition-master/models/shape_predictor_68_face_landmarks.dat'
pose_predictor = dlib.shape_predictor(predictor_model)

# Path to face recognition model
face_recognition_model = 'C:/Users/LENOVO/Downloads/Face-Recognition-master/Face-Recognition-master/models/dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def _rect_to_tuple(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _tuple_to_rect(rect):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    """
    return dlib.rectangle(rect[3], rect[0], rect[1], rect[2])


def _trim_rect_tuple_to_bounds(rect, image_shape):
    """
    Ensure a tuple in (top, right, bottom, left) order is within the bounds of the image.
    """
    return max(rect[0], 0), min(rect[1], image_shape[1]), min(rect[2], image_shape[0]), max(rect[3], 0)


def load_image_file(filename, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array
    """
    img = np.array(Image.open(filename))

    # Resize large images
    if img.shape[0] > 800 or img.shape[1] > 800:
        baseheight = 500
        w = baseheight / max(img.shape[:2])
        new_shape = (int(img.shape[1] * w), int(img.shape[0] * w))
        img = np.array(Image.fromarray(img).resize(new_shape))

    return img


def _raw_face_locations(img, number_of_times_to_upsample=1):
    """
    Returns bounding boxes of human faces in an image
    """
    return face_detector(img, number_of_times_to_upsample)


def face_locations(img, number_of_times_to_upsample=1):
    """
    Returns bounding boxes of human faces in an image as tuples (top, right, bottom, left)
    """
    return [_trim_rect_tuple_to_bounds(_rect_to_tuple(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample)]


def _raw_face_landmarks(face_image, face_locations=None):
    """
    Returns facial landmarks from an image
    """
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_tuple_to_rect(face_location) for face_location in face_locations]

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_landmarks(face_image, face_locations=None):
    """
    Returns facial landmarks as a dictionary for each face in an image
    """
    landmarks = _raw_face_landmarks(face_image, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]


def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    """
    Returns 128-dimension face encodings for each face in an image
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)

    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


def face_distance(face_encodings, face_to_compare):
    """
    Calculate Euclidean distances between face encodings
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
