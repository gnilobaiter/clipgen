"""Child-process entry point for the sound-event classifier (see sound_classifier.py for why it is isolated).

Usage: python -m src.core.sound_worker MODEL.onnx AUDIO.npy SCORES_OUT.npy
"""

import sys

import numpy as np

from src.core import sound_classifier


def main(argv):
    model_path, audio_path, scores_path = argv
    session = sound_classifier.create_session(model_path, use_gpu=True)
    audio = np.load(audio_path, mmap_mode="r")
    np.save(scores_path, sound_classifier.classify_frames(audio, session))
    print(f"provider={session.get_providers()[0]}")


if __name__ == "__main__":
    main(sys.argv[1:])
