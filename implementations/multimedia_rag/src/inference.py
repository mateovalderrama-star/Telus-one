"""Inference utilities for frame-level evidence analysis and question answering."""

import os
import tempfile
import time

import decord
import matplotlib.pyplot as plt
import torch
import torchaudio


decord.bridge.set_bridge("torch")
from imagebind.models.imagebind_model import ModalityType  # noqa: E402
from src.model.avrag import (  # noqa: E402
    encode_frames_with_imagebind,
    sample_audio_windows,
    sample_frames,
)


def run_frame_evidence_analysis(rag, j_res, t_embed, video_paths, audio_paths, m=16, topk_frames=5, gamma=10.0):
    """
    Apply frame-level evidence analysis using Sequential Feature Selection (SFS).

    This function samples frames and audio windows from retrieved videos,
    calculates their relevance to the queries, and visualizes the results.

    Parameters
    ----------
    rag : object
        The RAG model instance used for encoding and selection.
    j_res : list of dict
        Retrieval results containing the top-1 video IDs for each query.
    t_embed : dict
        Text embeddings of the input queries.
    video_paths : list of str
        Full file paths for all available videos.
    audio_paths : list of str
        Full file paths for all available audios.
    m : int, default 16
        The total number of frames to sample from each video.
    topk_frames : int, default 5
        The number of highest-scoring frames to select.
    gamma : float, default 10.0
        Hyperparameter controlling the diversity in SFS selection.

    Returns
    -------
    None
    """
    print("\n================ Frame-Level Evidence Analysis ================")

    for q_idx, result in enumerate(j_res):
        query_key = list(result.keys())[0]
        top1_video_id = result[query_key][0]["file"]

        video_matches = [p for p in video_paths if top1_video_id in os.path.basename(p)]
        audio_matches = [p for p in audio_paths if top1_video_id in os.path.basename(p)]

        if not video_matches or not audio_matches:
            print("Matching segment not found.")
            continue

        top1_path = video_matches[0]
        audio_path = audio_matches[0]

        print(f"\nQuery: {query_key}")
        print(f"Top-1 video: {top1_video_id}")

        # ---- Sample frames ----
        frames, frame_indices = sample_frames(top1_path, m=m)
        vision_embed = encode_frames_with_imagebind(rag, frames)

        vr = decord.VideoReader(top1_path)
        video_fps = vr.get_avg_fps()
        audio_clips = sample_audio_windows(audio_path, frame_indices, video_fps)

        # Save temporary audio clips
        tmp_audio_dir = tempfile.mkdtemp()
        audio_paths_tmp = []

        for i, clip in enumerate(audio_clips):
            path = os.path.join(tmp_audio_dir, f"{i}.wav")
            torchaudio.save(path, clip, 16000)
            audio_paths_tmp.append(path)

        audio_embed = rag.encode(audio_paths_tmp, ModalityType.AUDIO)["embeddings"]

        # ---- Fuse AV per frame ----
        z = vision_embed * audio_embed

        # ---- Naive Top-K ----
        q_embed = t_embed["embeddings"][q_idx : q_idx + 1].to(rag.device)
        z_norm = torch.nn.functional.normalize(z, dim=-1)
        q_norm = torch.nn.functional.normalize(q_embed, dim=-1)

        sim_scores = (q_norm @ z_norm.T).squeeze(0)
        naive_topk = torch.topk(sim_scores, k=topk_frames).indices.cpu().tolist()

        print("Naive Top-K frames:", frame_indices[naive_topk].tolist())

        # ---- SFS ----
        selected = rag.sfs(z, k=topk_frames, gamma=gamma)
        print("SFS Selected frames:", frame_indices[selected].tolist())

        # ---- Visualization ----
        plt.figure(figsize=(12, 3))
        for i, idx in enumerate(selected):
            plt.subplot(1, len(selected), i + 1)
            plt.imshow(frames[idx].numpy())
            plt.axis("off")
            plt.title(f"{frame_indices[idx].item()}")
        plt.suptitle("SFS Selected Frames")
        plt.tight_layout()
        plt.show()

        # ---- Q Matrix ----
        q_matrix = rag.build_sfs_Q(z, gamma=gamma).cpu().numpy()  # noqa: N806

        plt.figure(figsize=(6, 5))
        plt.imshow(q_matrix)
        plt.colorbar()
        plt.title(f"Q Matrix (gamma={gamma})")
        plt.show()


def process_retrieved_files(
    retrieved_files,
    question,
    root_data_dir,
    segment_suffix,
    model,
    bsz,
    default_topic=None,
):
    """
    Generate answers for a set of retrieved video segments.

    This function locates the physical files for each retrieved segment ID,
    prepares them for the model, and runs inference to obtain text answers.

    Parameters
    ----------
    retrieved_files : list
        Identifiers for the videos retrieved for a given question.
    question : str
        The user's question to be answered by the model.
    root_data_dir : str
        The base directory where all topic folders are stored.
    segment_suffix : str
        The duration suffix for segmented directories (e.g., '30s').
    model : object
        The multimodal LLM used to generate answers.
    bsz : int
        Processing batch size.
    default_topic : str, optional
        Topic name to use if the ID doesn't contain one. Required for local IDs.

    Returns
    -------
    dict
        A mapping of file identifiers to their generated text responses.

    Raises
    ------
    ValueError
        If a local ID is provided without a `default_topic`.
    """
    agent_answers = {}

    for retrieved_item in retrieved_files:
        retrieved_file = retrieved_item["file"] if isinstance(retrieved_item, dict) else retrieved_item

        # ---- Detect global vs local ----
        parts = retrieved_file.split("__")

        if len(parts) >= 3:
            # Global format: Topic__002__000
            topic = parts[0]
            segment_name = "__".join(parts[1:])
        else:
            # Local format: 002__000
            if not default_topic:
                raise ValueError(
                    "Local retrieved_file format '002__000' requires a non-empty "
                    "default_topic, but none was provided. Either supply a default "
                    "topic or use the global format 'Topic__002__000'."
                )
            topic = default_topic
            segment_name = retrieved_file

        video_path = os.path.join(
            root_data_dir,
            topic,
            f"segment-video_{segment_suffix}",
            f"{segment_name}.mp4",
        )

        audio_path = os.path.join(
            root_data_dir,
            topic,
            f"segment-audio_{segment_suffix}",
            f"{segment_name}.wav",
        )

        if not os.path.exists(video_path):
            print(f"[ERROR] Missing video: {video_path}")
            continue

        if not os.path.exists(audio_path):
            print(f"[ERROR] Missing audio: {audio_path}")
            continue

        inputs = [{"text": question, "video": video_path, "audio": audio_path}]

        inputs = model.prepare_input(inputs)

        start_time = time.time()
        text, _ = model.generate(inputs)
        end_time = time.time()

        print(f"[{retrieved_file}] Inference: {end_time - start_time:.2f}s")

        agent_answers[retrieved_file] = text

        torch.cuda.empty_cache()

    return agent_answers


def process_question(source, root_data_dir, segment_suffix, model, bsz, topic):
    """
    Handle the logical flow for answering a single question.

    Aggregates the retrieved segments for a question and initiates the
    inference process to gather individual agent responses.

    Parameters
    ----------
    source : dict
        A dictionary containing the question string and its retrieved segments.
    root_data_dir : str
        Base directory containing the multimedia data.
    segment_suffix : str
        Directory suffix for segment storage.
    model : object
        The model used for answer generation.
    bsz : int
        Batch size for inference.
    topic : str
        Topic name for the question.

    Returns
    -------
    dict
        The updated `source` dictionary containing an additional 'agent_answers' key.
    """
    question = source["question"]
    retrieved_files = source["retrieved_file"]

    if isinstance(retrieved_files, str):
        retrieved_files = [retrieved_files]

    agent_answers = process_retrieved_files(
        retrieved_files=retrieved_files,
        question=question,
        root_data_dir=root_data_dir,
        segment_suffix=segment_suffix,
        model=model,
        bsz=bsz,
        default_topic=topic,
    )

    source["agent_answers"] = agent_answers
    return source
