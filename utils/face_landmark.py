from PIL import Image
import numpy as np
import torch

from diffusers import StableDiffusionPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from typing import Mapping
import mediapipe as mp

class LandmarkAnnotator:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_detection = mp.solutions.face_detection  # Only for counting faces.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
        self.mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS
        self.mp_body_connections = mp.solutions.pose_connections.POSE_CONNECTIONS

        self.DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
        self.PoseLandmark = mp.solutions.drawing_styles.PoseLandmark

        self.f_thick = 2
        self.f_rad = 1
        self.right_iris_draw = self.DrawingSpec(color=(10, 200, 250), thickness=self.f_thick, circle_radius=self.f_rad)
        self.right_eye_draw = self.DrawingSpec(color=(10, 200, 180), thickness=self.f_thick, circle_radius=self.f_rad)
        self.right_eyebrow_draw = self.DrawingSpec(color=(10, 220, 180), thickness=self.f_thick, circle_radius=self.f_rad)
        self.left_iris_draw = self.DrawingSpec(color=(250, 200, 10), thickness=self.f_thick, circle_radius=self.f_rad)
        self.left_eye_draw = self.DrawingSpec(color=(180, 200, 10), thickness=self.f_thick, circle_radius=self.f_rad)
        self.left_eyebrow_draw = self.DrawingSpec(color=(180, 220, 10), thickness=self.f_thick, circle_radius=self.f_rad)
        self.mouth_draw = self.DrawingSpec(color=(10, 180, 10), thickness=self.f_thick, circle_radius=self.f_rad)
        self.head_draw = self.DrawingSpec(color=(10, 200, 10), thickness=self.f_thick, circle_radius=self.f_rad)

        # mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
        self.face_connection_spec = {}
        for edge in self.mp_face_mesh.FACEMESH_FACE_OVAL:
            self.face_connection_spec[edge] = self.head_draw
        for edge in self.mp_face_mesh.FACEMESH_LEFT_EYE:
            self.face_connection_spec[edge] = self.left_eye_draw
        for edge in self.mp_face_mesh.FACEMESH_LEFT_EYEBROW:
            self.face_connection_spec[edge] = self.left_eyebrow_draw
        # for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
        #    face_connection_spec[edge] = left_iris_draw
        for edge in self.mp_face_mesh.FACEMESH_RIGHT_EYE:
            self.face_connection_spec[edge] = self.right_eye_draw
        for edge in self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
            self.face_connection_spec[edge] = self.right_eyebrow_draw
        # for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
        #    face_connection_spec[edge] = right_iris_draw
        for edge in self.mp_face_mesh.FACEMESH_LIPS:
            self.face_connection_spec[edge] = self.mouth_draw
        self.iris_landmark_spec = {468: self.right_iris_draw, 473: self.left_iris_draw}


    def draw_pupils(self, image, landmark_list, drawing_spec, halfwidth: int = 2):
        """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
        landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
        if len(image.shape) != 3:
            raise ValueError("Input image must be H,W,C.")
        image_rows, image_cols, image_channels = image.shape
        if image_channels != 3:  # BGR channels
            raise ValueError('Input image must contain three channel bgr data.')
        for idx, landmark in enumerate(landmark_list.landmark):
            if (
                    (landmark.HasField('visibility') and landmark.visibility < 0.9) or
                    (landmark.HasField('presence') and landmark.presence < 0.5)
            ):
                continue
            if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
                continue
            image_x = int(image_cols*landmark.x)
            image_y = int(image_rows*landmark.y)
            draw_color = None
            if isinstance(drawing_spec, Mapping):
                if drawing_spec.get(idx) is None:
                    continue
                else:
                    draw_color = drawing_spec[idx].color
            elif isinstance(drawing_spec, self.DrawingSpec):
                draw_color = drawing_spec.color
            image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color


    def reverse_channels(self, image):
        """Given a np array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
        # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
        # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
        return image[:, :, ::-1]


    def generate_annotation(
            self,
            input_image: Image.Image,
            max_faces: int,
            min_face_size_pixels: int = 0,
            return_annotation_data: bool = False
    ):
        """
        Find up to 'max_faces' inside the provided input image.
        If min_face_size_pixels is provided and nonzero it will be used to filter faces that occupy less than this many
        pixels in the image.
        If return_annotation_data is TRUE (default: false) then in addition to returning the 'detected face' image, three
        additional parameters will be returned: faces before filtering, faces after filtering, and an annotation image.
        The faces_before_filtering return value is the number of faces detected in an image with no filtering.
        faces_after_filtering is the number of faces remaining after filtering small faces.

        :return:
        If 'return_annotation_data==True', returns (np array, np array, int, int).
        If 'return_annotation_data==False' (default), returns a np array.
        """
        with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
        ) as facemesh:
            img_rgb = np.asarray(input_image)
            results = facemesh.process(img_rgb).multi_face_landmarks

            faces_found_before_filtering = len(results)

            # Filter faces that are too small
            filtered_landmarks = []
            for lm in results:
                landmarks = lm.landmark
                face_rect = [
                    landmarks[0].x,
                    landmarks[0].y,
                    landmarks[0].x,
                    landmarks[0].y,
                ]  # Left, up, right, down.
                for i in range(len(landmarks)):
                    face_rect[0] = min(face_rect[0], landmarks[i].x)
                    face_rect[1] = min(face_rect[1], landmarks[i].y)
                    face_rect[2] = max(face_rect[2], landmarks[i].x)
                    face_rect[3] = max(face_rect[3], landmarks[i].y)
                if min_face_size_pixels > 0:
                    face_width = abs(face_rect[2] - face_rect[0])
                    face_height = abs(face_rect[3] - face_rect[1])
                    face_width_pixels = face_width * input_image.size[0]
                    face_height_pixels = face_height * input_image.size[1]
                    face_size = min(face_width_pixels, face_height_pixels)
                    if face_size >= min_face_size_pixels:
                        filtered_landmarks.append(lm)
                else:
                    filtered_landmarks.append(lm)

            faces_remaining_after_filtering = len(filtered_landmarks)

            # Annotations are drawn in BGR for some reason, but we don't need to flip a zero-filled image at the start.
            empty = np.zeros_like(img_rgb)

            # Draw detected faces:
            for face_landmarks in filtered_landmarks:
                self.mp_drawing.draw_landmarks(
                    empty,
                    face_landmarks,
                    connections=self.face_connection_spec.keys(),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.face_connection_spec
                )
                self.draw_pupils(empty, face_landmarks, self.iris_landmark_spec, 2)

            # Flip BGR back to RGB.
            empty = self.reverse_channels(empty)

            # We might have to generate a composite.
            if return_annotation_data:
                # Note that we're copying the input image AND flipping the channels so we can draw on top of it.
                annotated = self.reverse_channels(np.asarray(input_image)).copy()
                for face_landmarks in filtered_landmarks:
                    self.mp_drawing.draw_landmarks(
                        empty,
                        face_landmarks,
                        connections=self.face_connection_spec.keys(),
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.face_connection_spec
                    )
                    self.draw_pupils(empty, face_landmarks, self.iris_landmark_spec, 2)
                annotated = self.reverse_channels(annotated)

            if not return_annotation_data:
                return empty
            else:
                return empty, annotated, faces_found_before_filtering, faces_remaining_after_filtering
            
    def get_annotation(
                    self, 
                    image: str, 
                    resolution=(512, 512),
                    save_path: str=None
                    ):
        face_img = Image.open(image).convert("RGB").resize(resolution)
        face_landmark = self.generate_annotation(
            input_image=face_img,
            max_faces=1,
        )
        if save_path is not None:
            Image.fromarray(face_landmark).save(save_path)

        return face_landmark
