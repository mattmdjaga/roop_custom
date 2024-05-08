import threading
from typing import Any, Optional, List
import insightface
import numpy
import os
import sys

import onnxruntime


def encode_execution_providers(execution_providers):
    return [
        execution_provider.replace("ExecutionProvider", "").lower()
        for execution_provider in execution_providers
    ]


def decode_execution_providers(execution_providers):
    return [
        provider
        for provider, encoded_execution_provider in zip(
            onnxruntime.get_available_providers(),
            encode_execution_providers(onnxruntime.get_available_providers()),
        )
        if any(
            execution_provider in encoded_execution_provider
            for execution_provider in execution_providers
        )
    ]


def suggest_execution_providers():
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        return 8
    return 1


def update_status(message: str, scope: str = "ROOP.CORE") -> None:
    print(f"[{scope}] {message}")
    if not roop.globals.headless:
        ui.update_status(message)

current_dir = os.path.dirname(__file__)
parent_parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_parent_dir)

import roop.globals
from roop.typing import Frame, Face

from roop.utilities import normalize_output_path
roop.globals.many_faces = False 
roop.globals.reference_face_position = 0
roop.globals.reference_frame_number = 0
roop.globals.similar_face_distance = 0.85
roop.globals.execution_providers = decode_execution_providers(
    ["cpu"]
)
roop.globals.execution_threads = (
    suggest_execution_threads()
)
roop.globals.headless = True

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


def clear_face_analyser() -> Any:
    global FACE_ANALYSER

    FACE_ANALYSER = None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        return get_face_analyser().get(frame)
    except ValueError:
        print('No face detected in the frame.')
        return None


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
