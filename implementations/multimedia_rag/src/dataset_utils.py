"""Utilities for loading, validating, and processing dataset files."""

import json
import os


def extract_id(filename):
    """
    Clean a filename to create a consistent identifier.

    Parameters
    ----------
    filename : str
        The original filename string.

    Returns
    -------
    str
        The cleaned identifier without extension and with standardized underscores.
    """
    base = os.path.splitext(filename)[0]  # remove extension
    return base.replace("__", "_")


def check_dataset_integrity(  # noqa: PLR0912
    root_path, vid_dir="video", aud_dir="audio", cap_dir="caption"
):
    """
    Validate that every sample has its corresponding media and text files.

    Scans the dataset layout and identifies any missing triplets
    (video, audio, and caption) for each content ID.

    Parameters
    ----------
    root_path : str
        The directory containing the topic folders.
    vid_dir : str, default "video"
        Name of the subfolder containing video files.
    aud_dir : str, default "audio"
        Name of the subfolder containing audio files.
    cap_dir : str, default "caption"
        Name of the subfolder containing caption files.

    Returns
    -------
    None
    """
    # List all main folders in the root directory, excluding hidden ones.
    main_folders = [
        f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f)) and not f.startswith(".")
    ]

    # Iterate through each folder and check its contents.
    for folder in sorted(main_folders):
        print(f"\nChecking: {folder}")
        folder_path = os.path.join(root_path, folder)

        # Define paths for video, audio, and caption subdirectories.
        video_dir = os.path.join(folder_path, vid_dir)
        audio_dir = os.path.join(folder_path, aud_dir)
        caption_dir = os.path.join(folder_path, cap_dir)

        # Initialize sets to store unique IDs for videos, audios, and captions.
        video_ids = set()
        audio_ids = set()
        caption_ids = set()

        # Collect video IDs from the video directory.
        if os.path.exists(video_dir):
            for f in os.listdir(video_dir):
                full_path = os.path.join(video_dir, f)
                if os.path.isfile(full_path) and not f.startswith("."):
                    video_ids.add(extract_id(f))

        # Collect audio IDs from the audio directory.
        if os.path.exists(audio_dir):
            for f in os.listdir(audio_dir):
                full_path = os.path.join(audio_dir, f)
                if os.path.isfile(full_path) and not f.startswith("."):
                    audio_ids.add(extract_id(f))

        # Collect caption IDs from the caption directory.
        if os.path.exists(caption_dir):
            for f in os.listdir(caption_dir):
                full_path = os.path.join(caption_dir, f)
                if os.path.isfile(full_path) and not f.startswith("."):
                    caption_ids.add(extract_id(f))

        # Combine all unique IDs from videos, audios, and captions.
        all_ids = sorted(video_ids | audio_ids | caption_ids)

        # Initialize counters for matched and mismatched triplets.
        matched_count = 0
        mismatch_count = 0

        # Check each ID for the presence of video, audio, and caption files.
        for id_ in all_ids:
            has_video = id_ in video_ids
            has_audio = id_ in audio_ids
            has_caption = id_ in caption_ids

            if has_video and has_audio and has_caption:
                matched_count += 1  # Increment matched triplet count.
            else:
                mismatch_count += 1  # Increment mismatch count.
                print(f"  ID {id_}: video={has_video}, audio={has_audio}, caption={has_caption}")

        # Print summary of the integrity check for the current folder.
        print(f"  Total unique IDs: {len(all_ids)}")
        print(f"  Matched triplets: {matched_count}")
        print(f"  Mismatches: {mismatch_count}")

        # If no mismatches are found, print a success message.
        if mismatch_count == 0:
            print("  All video-audio-caption triplets match.")


def extract_video_number(filename):
    """
    Extract the numeric video ID from a filename string.

    Handles various formats including prefixes and segmentation IDs.

    Parameters
    ----------
    filename : str
        The media filename to parse.

    Returns
    -------
    str or None
        The numeric representation as a string if found, otherwise None.
    """
    base = os.path.splitext(filename)[0]

    # Remove segmentation part if present (002__000 -> 002)
    base = base.split("__")[0]

    # Remove optional prefix (video_002 -> 002)
    if "_" in base:
        base = base.split("_")[-1]

    return base if base.isdigit() else None


def filter_json_by_existing_videos(json_path, video_folder, output_path=None):
    """
    Synchronize a JSON dataset with the actual files on disk.

    Removes any questions or metadata entries from the JSON list if the
    corresponding video file is missing from the specified folder.

    Parameters
    ----------
    json_path : str
        Path to the source JSON dataset.
    video_folder : str
        Directory to check for existing video files.
    output_path : str, optional
        Target path for the filtered JSON output.

    Returns
    -------
    None
    """
    # Load JSON data from the specified file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Collect available video numbers from the video folder
    available_video_numbers = set()
    for fname in os.listdir(video_folder):
        vid_num = extract_video_number(fname)
        if vid_num:
            available_video_numbers.add(vid_num)

    print(f"Found {len(available_video_numbers)} videos in folder.")

    # Filter entries in the JSON file to include only those with matching video numbers
    original_count = len(data.get("entries", []))
    filtered_entries = [
        entry for entry in data.get("entries", []) if entry.get("video_number") in available_video_numbers
    ]
    filtered_count = len(filtered_entries)

    print(f"Original entries: {original_count}")
    print(f"Filtered entries: {filtered_count}")

    # Update the JSON data with the filtered entries
    data["entries"] = filtered_entries
    data["num_entries"] = filtered_count

    # Save the filtered JSON data to the specified output path, if provided
    if output_path:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Filtered JSON saved to: {output_path}")

    # The function does not return anything; it modifies the JSON file in place
    # if output_path is provided.


def simplify_mcq_json(data):
    """
    Extract a cleaner, flatter version of MCQ data from a raw JSON.

    Parameters
    ----------
    data : dict
        The raw dataset dictionary.

    Returns
    -------
    list of dict
        A list of entries containing only the most important fields for evaluation.
    """
    simplified_entries = []

    # Iterate over all entries in the "entries" list of the input data
    for entry in data.get("entries", []):
        # Extract the index of the correct answer
        answer_index = entry.get("answer_index")
        # Extract the list of options
        options = entry.get("options", [])

        # Determine the correct answer text based on the answer index
        correct_answer_text = (
            options[answer_index] if isinstance(answer_index, int) and answer_index < len(options) else None
        )

        # Create a simplified representation of the MCQ entry
        simplified_entry = {
            "video_id": entry.get("video_id"),  # ID of the video
            "video_number": entry.get("video_number"),  # Video number
            "segment": entry.get("segment"),  # Segment information
            "question": entry.get("question"),  # The question text
            "options": options,  # List of answer options
            "answer_index": answer_index,  # Index of the correct answer
            "answer_letter": entry.get("answer_letter"),  # Letter of the selected answer
            "correct_answer_letter": entry.get("answer_letter"),  # Letter of the correct answer
            "correct_answer_text": correct_answer_text,  # Text of the correct answer
            "rationale": entry.get("rationale"),  # Explanation or rationale for the answer
        }

        # Add the simplified entry to the result list
        simplified_entries.append(simplified_entry)

    return simplified_entries


def rename_media_files(parent_dir: str):
    """
    Remove verbose prefixes from media filenames in a directory.

    Parameters
    ----------
    parent_dir : str
        The topic folder containing 'video', 'audio', and 'caption' subdirectories.

    Returns
    -------
    None
    """

    def rename_in_folder(folder, prefix):
        """Rename files in a specific subfolder with a given prefix."""
        folder_path = os.path.join(parent_dir, folder)

        if not os.path.exists(folder_path):
            return

        for filename in os.listdir(folder_path):
            if not filename.startswith(prefix):
                continue

            old_path = os.path.join(folder_path, filename)

            # Remove prefix
            new_name = filename[len(prefix) :]
            new_path = os.path.join(folder_path, new_name)

            if os.path.exists(new_path):
                print(f"Skipping (exists): {new_name}")
                continue

            os.rename(old_path, new_path)
            print(f"Renamed → {new_name}")

    rename_in_folder("video", "video_")
    rename_in_folder("audio", "audio_")
    rename_in_folder("caption", "caption_")

    print("\nRenaming complete.")


def srt_time_to_seconds(t):
    """
    Convert a standard SRT timestamp string to a numeric second value.

    Parameters
    ----------
    t : str
        Timestamp in 'HH:MM:SS,mmm' format.

    Returns
    -------
    float
        The absolute time in seconds.
    """
    hms, ms = t.split(",")  # Split into hours:minutes:seconds and milliseconds.
    h, m, s = hms.split(":")  # Extract hours, minutes, and seconds.
    return (
        int(h) * 3600  # Convert hours to seconds.
        + int(m) * 60  # Convert minutes to seconds.
        + int(s)  # Add seconds.
        + int(ms) / 1000.0  # Add milliseconds as a fraction of a second.
    )


def parse_srt_with_timestamps(srt_path):
    """
    Read an SRT file and convert it into structured data objects.

    Parameters
    ----------
    srt_path : str
        Path to the subtitle file.

    Returns
    -------
    list of dict
        Entries containing 'start', 'end', and 'text' as parsed keys.
    """
    entries = []

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().split("\n\n")  # Split blocks by double newlines.

    for block in lines:
        parts = block.strip().split("\n")
        if len(parts) < 3:  # Skip blocks with insufficient lines.
            continue

        time_line = parts[1]  # Extract the timestamp line.
        text = " ".join(parts[2:])  # Combine all text lines.

        start, end = time_line.split(" --> ")  # Split start and end times.

        entries.append(
            {
                "start": srt_time_to_seconds(start),
                "end": srt_time_to_seconds(end),
                "text": text,
            }
        )

    return entries


def seconds_to_srt(seconds):
    """
    Convert a numeric second value into a formatted SRT timestamp.

    Parameters
    ----------
    seconds : float
        The time in seconds to format.

    Returns
    -------
    str
        The formatted 'HH:MM:SS,mmm' string.
    """
    ms = int((seconds - int(seconds)) * 1000)  # Extract milliseconds.
    s = int(seconds)  # Extract whole seconds.
    h = s // 3600  # Calculate hours.
    m = (s % 3600) // 60  # Calculate minutes.
    s = s % 60  # Calculate remaining seconds.
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"  # Format as HH:MM:SS,mmm.
