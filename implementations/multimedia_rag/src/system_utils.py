"""System utilities for path alignment and GPU memory reporting."""

import os

import torch


def get_aligned_paths(video_dir, audio_dir, caption_dir):
    """
    Find and align media files based on their shared filenames.

    This function scans video, audio, and caption directories and keeps
    only the files that have a matching filename in all three locations.

    Parameters
    ----------
    video_dir : str
        Path to the directory containing video files (.mp4).
    audio_dir : str
        Path to the directory containing audio files (.wav).
    caption_dir : str
        Path to the directory containing caption files (.srt).

    Returns
    -------
    video_paths : list of str
        The list of absolute paths to the aligned video files.
    audio_paths : list of str
        The list of absolute paths to the aligned audio files.
    caption_paths : list of str
        The list of absolute paths to the aligned caption files.
    """
    video_paths = sorted(os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4"))

    audio_paths = sorted(os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav"))

    caption_paths = sorted(os.path.join(caption_dir, f) for f in os.listdir(caption_dir) if f.endswith(".srt"))

    video_ids = {os.path.splitext(os.path.basename(p))[0] for p in video_paths}
    audio_ids = {os.path.splitext(os.path.basename(p))[0] for p in audio_paths}
    caption_ids = {os.path.splitext(os.path.basename(p))[0] for p in caption_paths}

    common_ids = sorted(video_ids & audio_ids & caption_ids)

    video_paths = [os.path.join(video_dir, f"{file_id}.mp4") for file_id in common_ids]
    audio_paths = [os.path.join(audio_dir, f"{file_id}.wav") for file_id in common_ids]
    caption_paths = [os.path.join(caption_dir, f"{file_id}.srt") for file_id in common_ids]

    return video_paths, audio_paths, caption_paths


def print_gpu_memory():
    """
    Print the current GPU memory statistics to the console.

    Calculates and displays the amount of allocated and reserved memory on
    the primary CUDA device in gigabytes (GB). If no GPU is available,
    it does nothing.

    Returns
    -------
    None
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
