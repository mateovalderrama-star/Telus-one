"""Utilities for processing media files (video and audio) using ffmpeg."""

import os
import subprocess


def get_duration(file_path):
    """
    Calculate the duration of a media file.

    Parameters
    ----------
    file_path : str
        The path to the video or audio file.

    Returns
    -------
    float
        The duration of the file in seconds.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        text=True,
    )
    return float(result.stdout.strip())


def list_video_durations(folder_path, threshold_seconds=300):
    """
    Scan a folder and summarize the durations of all video files.

    It lists all videos and highlights those that fall below a specific
    time threshold.

    Parameters
    ----------
    folder_path : str
        The directory containing video files.
    threshold_seconds : int, default 300
        The duration limit in seconds for identifying short videos.

    Returns
    -------
    None
    """
    video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".m4a", ".webm")

    durations = {}
    total_videos = 0
    below_threshold_files = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(video_extensions):
            total_videos += 1
            full_path = os.path.join(folder_path, filename)

            try:
                duration = get_duration(full_path)
                durations[filename] = duration

                if duration < threshold_seconds:
                    below_threshold_files.append((filename, duration))

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Print all durations sorted by duration (ascending)
    print("\nAll videos sorted by duration:")
    for filename, duration in sorted(durations.items(), key=lambda x: x[1]):
        print(f"{filename}: {duration} seconds")

    if total_videos > 0:
        percentage = (len(below_threshold_files) / total_videos) * 100
        print(f"\n{percentage:.2f}% of videos are less than {threshold_seconds} seconds.")

        if below_threshold_files:
            print("\nFiles below threshold (sorted by duration):")
            for filename, duration in sorted(below_threshold_files, key=lambda x: x[1]):
                print(f"- {filename}: {duration} seconds")
        else:
            print("\nNo files below threshold.")
    else:
        print("\nNo video files found.")


def process_video(video_dir, process_dir, max_time=60):
    """
    Standardize video files in a directory to a consistent format.

    Uses ffmpeg to copy video streams into a standardized container.
    Videos exceeding the duration limit are skipped.

    Parameters
    ----------
    video_dir : str
        Directory containing the source video files.
    process_dir : str
        Directory where the processed videos will be saved.
    max_time : int, default 60
        The maximum allowed duration in seconds.

    Returns
    -------
    None
    """
    # Define supported video formats
    video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    output_dir = process_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process each video file in the directory
    for filename in os.listdir(video_dir):
        if filename.lower().endswith(video_extensions):
            input_path = os.path.join(video_dir, filename)
            duration = get_duration(input_path)
            if max_time is not None and duration > max_time:
                print(f"Skipping {filename} ({duration:.2f}s > {max_time}s)")
                continue
            output_path = os.path.join(output_dir, filename)
            print(f"Processing: {filename}")

            # Copy video streams without re-encoding
            command = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-map",
                "0",
                "-c",
                "copy",
                output_path,
            ]

            subprocess.run(command, check=True)


def process_audio(audio_dir, process_dir, max_time=60.00000):
    """
    Convert audio files from M4A to a standardized WAV format.

    Uses ffmpeg to perform the conversion with PCM encoding.
    Files exceeding the duration limit are skipped.

    Parameters
    ----------
    audio_dir : str
        Directory containing the source M4A audio files.
    process_dir : str
        Directory where the converted WAV files will be saved.
    max_time : float, default 60.0
        The maximum allowed duration in seconds.

    Returns
    -------
    None
    """
    output_dir = process_dir
    os.makedirs(output_dir, exist_ok=True)

    # Convert M4A files to WAV format
    for filename in os.listdir(audio_dir):
        if filename.lower().endswith(".m4a"):
            input_path = os.path.join(audio_dir, filename)
            duration = get_duration(input_path)
            if max_time is not None and duration > max_time:
                print(f"Skipping {filename} ({duration:.2f}s > {max_time}s)")
                continue
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(output_dir, output_filename)
            print(f"Converting: {filename} -> {output_filename}")

            # Convert to PCM WAV format
            command = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-acodec",
                "pcm_s16le",
                output_path,
            ]

            subprocess.run(command, check=True)
