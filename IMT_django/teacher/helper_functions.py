import numpy as np
import os


def pad_audio(audio_in, length=40000):
    trail_space = max(length - audio_in.shape[0], 0)

    print(trail_space)
    audio_padded = np.pad(audio_in, (0, trail_space))

    return audio_padded


def calculate_final_score(request):
    total_score = np.sum(request.session['score'])
    final_score = total_score / len(request.session['score'])

    return final_score







