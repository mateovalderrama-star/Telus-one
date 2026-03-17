"""Audio-Visual Retrieval-Augmented Generation (AV-RAG) model implementation."""

import math
import os
import tempfile

import decord
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F  # noqa: N812
import torchaudio
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from PIL import Image


decord.bridge.set_bridge("torch")


def get_first_k(dir_path, ext, k):
    """Return the first k sorted file paths in dir_path matching the given extension."""
    return sorted(os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(ext))[:k]


class AVRAG:
    """
    Audio-Visual Retrieval-Augmented Generation retrieval engine.

    Uses ImageBind to create cross-modal embeddings for video, audio, and
    text. This engine manages the alignment and scoring of multi-modal
    segments against user queries.
    """

    def __init__(self, model_path=None, bsz=128):
        """Initialize AVRAG with an optional pretrained model path and batch size."""
        self.bsz = bsz
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Instantiate model
        if model_path:
            self.model = imagebind_model.imagebind_huge(pretrained=False)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:  # download pretrained model automatically
            self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        self.load_and_transform_func = {
            ModalityType.TEXT: data.load_and_transform_text,
            ModalityType.AUDIO: data.load_and_transform_audio_data,
            ModalityType.VISION: [
                data.load_and_transform_vision_data,
                data.load_and_transform_video_data,
            ],
        }

    @torch.no_grad()
    def encode(self, input_paths, data_type, cache=False, save_cache=False, cache_path=None) -> dict:  # noqa: PLR0912
        """
        Convert diverse input media into numerical embeddings.

        Parameters
        ----------
        input_paths : str or list of str
            File paths or raw text strings to be encoded.
        data_type : ModalityType
            The modality of the input (VISION, AUDIO, or TEXT).
        cache : bool, default False
            If True, load existing embeddings from disk instead of re-encoding.
        save_cache : bool, default False
            If True, write the computed embeddings to a file for later use.
        cache_path : str, optional
            Target file path for saving/loading the cache.

        Returns
        -------
        dict
            A dictionary containing 'filename' and 'embeddings' tensors.
        """
        if cache:
            assert input_paths.endswith(".pt")
            return torch.load(input_paths)

        if isinstance(input_paths, str):
            if os.path.isdir(input_paths):
                if data_type == ModalityType.VISION:
                    exts = (".mp4", ".jpg", ".png")
                elif data_type == ModalityType.AUDIO:
                    exts = (".wav", ".m4a")
                else:
                    exts = ()

                input_paths = sorted(
                    [os.path.join(input_paths, f) for f in os.listdir(input_paths) if f.endswith(exts)]
                )
            else:
                input_paths = [input_paths]

        all_batches = []

        for start in range(0, len(input_paths), self.bsz):
            end = start + self.bsz
            input_batch = input_paths[start:end]

            if data_type == ModalityType.VISION:
                indice = 1 if input_batch[0].endswith(".mp4") else 0
                inputs = {
                    data_type: self.load_and_transform_func[data_type][indice](input_batch, self.device),
                }
            else:
                inputs = {
                    data_type: self.load_and_transform_func[data_type](input_batch, self.device),
                }

            # KEEP ON GPU
            embedding_batch = self.model(inputs)[data_type]
            all_batches.append(embedding_batch)

        embeddings = torch.cat(all_batches, dim=0)

        # ---- Safer filename handling ----
        if data_type != ModalityType.TEXT:
            filenames = [os.path.splitext(os.path.basename(path))[0] for path in input_paths]
        else:
            # Avoid using raw query string as filename
            filenames = [f"text_{i}" for i in range(len(input_paths))]

        result = {"filename": filenames, "embeddings": embeddings}

        # Save cache on CPU only when explicitly requested
        if save_cache and data_type != ModalityType.TEXT:
            if cache_path is None:
                cache_path = os.path.join(
                    os.path.dirname(os.path.dirname(input_paths[0])),
                    f"{'video' if data_type == ModalityType.VISION else 'audio'}_embeddings.pt",
                )
            torch.save(
                {"filename": filenames, "embeddings": embeddings.cpu()},
                cache_path,
            )

        return result

    def _parse_srt(self, srt_path: str) -> str:
        """
        Extract the plain text from an SRT file, ignoring timestamps.

        Parameters
        ----------
        srt_path : str
            The path to the source .srt file.

        Returns
        -------
        str
            The combined text contents of all subtitle blocks.
        """
        lines = []
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.isdigit():
                    continue
                if "-->" in line:
                    continue
                lines.append(line)
        return " ".join(lines)

    @torch.no_grad()
    def encode_srt_dir(self, srt_dir: str, cache: bool = False) -> dict:
        """
        Process a directory of SRT files into a text-based vocabulary.

        Parameters
        ----------
        srt_dir : str
             Directory containing .srt files.
        cache : bool, default False
             Whether to load from a pre-computed .pt file.

        Returns
        -------
        dict
            A dictionary with IDs and text embeddings.
        """
        if cache:
            # caller passes .pt file path in srt_dir in this mode
            assert srt_dir.endswith(".pt")
            return torch.load(srt_dir)

        srt_paths = sorted(os.path.join(srt_dir, f) for f in os.listdir(srt_dir) if f.endswith(".srt"))
        if len(srt_paths) == 0:
            raise ValueError(f"No .srt files found in {srt_dir}")

        texts = [self._parse_srt(p) for p in srt_paths]

        # Important: encode expects list[str] for TEXT
        cap_embed = self.encode(texts, ModalityType.TEXT, cache=False)

        # Replace filenames (currently texts) with the .srt basenames for alignment
        cap_embed["filename"] = [os.path.splitext(os.path.basename(p))[0] for p in srt_paths]

        # Optional cache next to srt_dir
        torch.save(cap_embed, os.path.join(os.path.dirname(srt_dir), "caption_embeddings.pt"))
        return cap_embed

    def topk(self, queries, vocabs, k=1, log=True):
        """
        Identify the most similar items in a vocabulary for a set of queries.

        Parameters
        ----------
        queries : torch.Tensor
            Embeddings of shape (n, d) representing the queries.
        vocabs : torch.Tensor
            Embeddings of shape (m, d) representing the vocabulary candidates.
        k : int, default 1
            Number of top results to retrieve.
        log : bool, default True
            If True, prints the results to the console.

        Returns
        -------
        indices : torch.Tensor
             The indices of the top-k items.
        values : torch.Tensor
             The similarity scores for the top-k items.
        """
        q = torch.nn.functional.normalize(queries, dim=-1)
        v = torch.nn.functional.normalize(vocabs, dim=-1)

        scores = q @ v.T
        values, indices = torch.topk(scores, k=k, dim=-1)

        if log:
            for q_idx in range(values.shape[0]):
                print(f"\nQuery {q_idx}:")
                for rank in range(k):
                    print(
                        f"  Rank {rank + 1} | "
                        f"Index: {indices[q_idx, rank].item()} | "
                        f"Score: {values[q_idx, rank].item():.6f}"
                    )

        return indices, values

    def pair_rag(self, query=None, vocab=None, k=1):
        """
        Perform simple retrieval by matching queries to a single modality.

        Parameters
        ----------
        query : dict
            Dictionary containing query names and embeddings.
        vocab : dict
            Dictionary containing item names and their embeddings.
        k : int, default 1
            Number of results to return.

        Returns
        -------
        list of dict
            A list of retrieval results formatted for ranking.
        """
        topk_indices, topk_values = self.topk(query["embeddings"], vocab["embeddings"], k=k)

        topk_files = []

        for q_idx, topi_indices in enumerate(topk_indices):
            results = []

            for rank, vocab_idx in enumerate(topi_indices):
                results.append(
                    {
                        "rank": rank + 1,
                        "file": vocab["filename"][vocab_idx],
                        "score": float(topk_values[q_idx, rank]),
                    }
                )

            topk_files.append({query["filename"][q_idx]: results})

        return topk_files

    def joint_rag(self, query, vocab_vision, vocab_audio, vocab_caption, k=1, mode="0"):
        """
        Execute joint multi-modal retrieval (AV-RAG).

        Combines information from video, audio, and captions to find the
        best matching video segments for a given text query.

        Parameters
        ----------
        query : dict
            The input query embeddings.
        vocab_vision : dict
            Source video segment embeddings.
        vocab_audio : dict
            Source audio segment embeddings.
        vocab_caption : dict
            Source caption (SRT) embeddings.
        k : int, default 1
            The number of top segments to return.
        mode : str, default '0'
            The retrieval strategy version.

        Returns
        -------
        list of dict
            Ranked segments aligned by filename.
        """
        if mode != "0":
            raise NotImplementedError("Only mode '0' supported.")

        # ----- Align by filename -----
        mv = dict(zip(vocab_vision["filename"], vocab_vision["embeddings"]))
        ma = dict(zip(vocab_audio["filename"], vocab_audio["embeddings"]))
        mc = dict(zip(vocab_caption["filename"], vocab_caption["embeddings"]))

        common = sorted(set(mv) & set(ma) & set(mc))
        if len(common) == 0:
            raise ValueError("No overlapping filenames between vocabs.")

        # Move to GPU for scoring
        v = torch.stack([mv[f] for f in common], dim=0).to(self.device)
        a = torch.stack([ma[f] for f in common], dim=0).to(self.device)
        c = torch.stack([mc[f] for f in common], dim=0).to(self.device)
        q = query["embeddings"].to(self.device)

        # ----- Hadamard AV fusion -----
        e_av = torch.nn.functional.normalize(v * a, dim=-1)
        c = torch.nn.functional.normalize(c, dim=-1)
        q = torch.nn.functional.normalize(q, dim=-1)

        # ----- Similarities -----
        s_av = q @ e_av.T
        s_cap = q @ c.T
        s_final = (s_av + s_cap) / 2

        values, indices = torch.topk(s_final, k=k, dim=-1)

        # Move back to CPU for output formatting
        values = values.cpu()
        indices = indices.cpu()

        topk_files = []

        for q_idx in range(indices.shape[0]):
            results = []
            for rank in range(indices.shape[1]):
                idx = indices[q_idx, rank].item()
                results.append(
                    {
                        "rank": rank + 1,
                        "file": common[idx],
                        "score": float(values[q_idx, rank]),
                    }
                )
            topk_files.append({query["filename"][q_idx]: results})

        return topk_files

    # For ablation studies
    def compute_scores_av_only(self, q, V, A):  # noqa: N803
        """Compute audio-visual similarity scores via Hadamard fusion."""
        E_av = torch.nn.functional.normalize(V * A, dim=-1)  # noqa: N806
        q = torch.nn.functional.normalize(q, dim=-1)
        return q @ E_av.T

    def compute_scores_caption_only(self, q, C):  # noqa: N803
        """Compute caption-only similarity scores."""
        q = torch.nn.functional.normalize(q, dim=-1)
        C = torch.nn.functional.normalize(C, dim=-1)  # noqa: N806
        return q @ C.T

    def compute_scores_joint(self, q, V, A, C):  # noqa: N803
        """Compute joint AV and caption similarity scores."""
        S_av = self.compute_scores_av_only(q, V, A)  # noqa: N806
        S_cap = self.compute_scores_caption_only(q, C)  # noqa: N806
        return (S_av + S_cap) / 2

    # Salient frame selector (SFS)
    # -----------------------------
    # 1) Paper SFS DP (Algorithm 1)
    # -----------------------------
    def sfs_select_indices(self, Q: torch.Tensor, k: int) -> list[int]:  # noqa: N803
        """
        Solve the SFS optimization problem using dynamic programming.

        Parameters
        ----------
        Q : torch.Tensor
            The square quality matrix of shape (m, m).
        k : int
            The number of indices to select.

        Returns
        -------
        list of int
            The optimal set of indices chosen from the candidate set.
        """
        assert Q.dim() == 2 and Q.shape[0] == Q.shape[1], "Q must be square (m x m)"
        m = Q.shape[0]
        assert 1 <= k <= m, f"k must be in [1, m], got k={k}, m={m}"

        # C[i][j] = min cost to end at i selecting j frames
        C = torch.full((m + 1, k + 1), float("inf"), device=Q.device)  # noqa: N806
        back = torch.full((m + 1, k + 1), -1, dtype=torch.long, device=Q.device)

        C[0, 0] = 0.0

        for j in range(1, k + 1):
            for i in range(j, m + 1):
                # transition from p to i: p in [j-1 .. i-1]
                best_cost = float("inf")
                best_p = -1
                for p in range(j - 1, i):
                    prev = C[p, j - 1]
                    if torch.isinf(prev):
                        continue
                    # Q is 0-indexed for candidates: candidate idx = i-1, p-1
                    # For p==0 (no previous), we define 0 transition cost.
                    trans = 0.0 if p == 0 else Q[p - 1, i - 1].item()
                    cost = prev.item() + trans
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p

                C[i, j] = best_cost
                back[i, j] = best_p

        # backtrack from i=m (paper uses i<-m); you can also choose argmin_i C[i,k]
        i = m
        j = k
        result = []
        while j > 0:
            result.append(i - 1)  # candidate index
            i = back[i, j].item()
            j -= 1

        result.reverse()
        return result

    # ---------------------------------------
    # 2) Build Q = cosine_sim + temporal_penalty
    # ---------------------------------------
    def build_sfs_Q(self, z: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:  # noqa: N802
        """
        Construct the SFS quality matrix Q.

        The matrix balances visual similarity with a temporal diversity
        penalty to ensure the selected frames are both relevant and
        spread through the clip.

        Parameters
        ----------
        z : torch.Tensor
            Embeddings of shape (m, d) for candidate frames.
        gamma : float, default 10.0
            Weight of the temporal penalty term.

        Returns
        -------
        torch.Tensor
            The resulting (m, m) quality matrix.
        """
        assert z.dim() == 2, "z must be (m, d)"
        m = z.shape[0]

        z = F.normalize(z, dim=-1)
        Gamma = z @ z.T  # cosine similarity matrix (m x m)  # noqa: N806

        # temporal penalty Δ_ab = γ * ( 1 / sin(pi/2 * |a-b|) + 1 - 1 )
        # which simplifies to γ * (1 / sin(pi/2 * |a-b|)) for |a-b|>0, and 0 on diag.
        idx = torch.arange(m, device=z.device)
        dist = (idx[:, None] - idx[None, :]).abs().float()

        Delta = torch.zeros((m, m), device=z.device)  # noqa: N806
        nonzero = dist > 0
        # avoid division by zero by masking dist>0
        denom = torch.sin((math.pi / 2.0) * dist[nonzero])
        Delta[nonzero] = gamma * (1.0 / denom)

        return Gamma + Delta

    # ---------------------------------------
    # 3) Wrapper: given candidate embeddings -> selected indices
    # ---------------------------------------
    def sfs(self, candidate_z: torch.Tensor, k: int, gamma: float = 10.0) -> list[int]:
        """
        Select salient frames from video embeddings (SFS).

        Parameters
        ----------
        candidate_z : torch.Tensor
            Available frame embeddings.
        k : int
            Number of frames to output.
        gamma : float, default 10.0
            Temporal penalty factor.

        Returns
        -------
        list of int
            List of the selected frame indices.
        """
        Q = self.build_sfs_Q(candidate_z, gamma=gamma)  # noqa: N806
        return self.sfs_select_indices(Q, k=k)


def sample_frames(video_path, m=16):
    """
    Sample a fixed set of frames spaced evenly across a video.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    m : int, default 16
        The number of frames to extract.

    Returns
    -------
    frames : torch.Tensor
        The sampled frames with shape (m, H, W, C).
    indices : torch.Tensor
        The original indices of the sampled frames.
    """
    vr = decord.VideoReader(video_path)
    total = len(vr)

    # Sample frames uniformly across the video duration
    indices = torch.linspace(0, total - 1, steps=m).long()
    frames = vr.get_batch(indices)

    return frames, indices


def encode_frames_with_imagebind(rag, frames):
    """
    Generate vision embeddings for individual video frames.

    Parameters
    ----------
    rag : AVRAG
        The AVRAG engine instance.
    frames : torch.Tensor
        The frame tensors to encode.

    Returns
    -------
    torch.Tensor
        The vision embeddings for the input frames.
    """
    # Create temporary directory for frame images and ensure cleanup afterwards
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []

        # Save each frame as a temporary image file
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame.numpy())
            path = os.path.join(tmp_dir, f"{i}.jpg")
            img.save(path)
            paths.append(path)

        # Encode frames using ImageBind vision encoder
        return rag.encode(paths, ModalityType.VISION)["embeddings"]


def sample_audio_windows(audio_path, frame_indices, video_fps, window_sec=2.0):
    """
    Extract audio clips aligned with specific video frame indices.

    Parameters
    ----------
    audio_path : str
        Path to the source audio file.
    frame_indices : torch.Tensor
        The indices of the video frames focused on.
    video_fps : float
        The playback rate of the video.
    window_sec : float, default 2.0
        The width of the audio window in seconds centered on each frame.

    Returns
    -------
    list of torch.Tensor
        A list of audio waveform clips.
    """
    # Load audio waveform
    waveform, sr = torchaudio.load(audio_path)
    windows = []

    # Minimum audio length required (0.5 seconds)
    min_required = int(0.5 * sr)

    for frame_idx in frame_indices:
        # Convert frame index to time in seconds
        center_time = frame_idx.item() / video_fps

        # Calculate audio window boundaries
        start = int(max(0, (center_time - window_sec / 2) * sr))
        end = int(min(waveform.shape[1], (center_time + window_sec / 2) * sr))

        # Extract audio clip
        clip = waveform[:, start:end]

        # Pad clip if it's shorter than minimum required length
        if clip.shape[1] < min_required:
            pad_amount = min_required - clip.shape[1]
            clip = torch.nn.functional.pad(clip, (0, pad_amount))

        windows.append(clip)

    return windows


if __name__ == "__main__":
    # -------------------------------------------------
    # Configuration
    # -------------------------------------------------
    BASE_DIR = "<BASE_FOLDER>/data/Customer_Service_Interactions"

    VIDEO_DIR = os.path.join(BASE_DIR, "process-video")
    AUDIO_DIR = os.path.join(BASE_DIR, "process-audio")
    CAPTION_DIR = os.path.join(BASE_DIR, "caption")

    TOP_K = 3
    NUM_FILES = 8
    NUM_FRAMES = 16
    SFS_K = 5
    GAMMA = 0.0

    # -------------------------------------------------
    # Initialize RAG
    # -------------------------------------------------
    rag = AVRAG(model_path="./checkpoints/imagebind_huge.pth", bsz=16)

    # -------------------------------------------------
    # Example Queries
    # -------------------------------------------------
    queries = [
        "Find My iPhone",
        "Despite the customer's claims of being full after finishing a second serving of noodles, the chef insists on preparing a final rice dish, emphasizing it's part of 'Japanese culture.'",
        "Anna's money transfer",
    ]

    # -------------------------------------------------
    # Collect Media
    # -------------------------------------------------
    video_paths = get_first_k(VIDEO_DIR, ".mp4", NUM_FILES)
    audio_paths = get_first_k(AUDIO_DIR, ".wav", NUM_FILES)
    caption_paths = get_first_k(CAPTION_DIR, ".srt", NUM_FILES)

    print("Videos:", video_paths)
    print("Audios:", audio_paths)
    print("Captions:", caption_paths)

    # -------------------------------------------------
    # Encode Modalities
    # -------------------------------------------------
    print("\nEncoding text queries...")
    t_embed = rag.encode(queries, ModalityType.TEXT)

    print("Encoding videos...")
    v_embed = rag.encode(video_paths, ModalityType.VISION)

    print("Encoding audios...")
    a_embed = rag.encode(audio_paths, ModalityType.AUDIO)

    print("Encoding captions...")
    caption_texts = [rag._parse_srt(p) for p in caption_paths]
    c_embed = rag.encode(caption_texts, ModalityType.TEXT)
    c_embed["filename"] = [os.path.splitext(os.path.basename(p))[0] for p in caption_paths]

    # -------------------------------------------------
    # Joint Retrieval
    # -------------------------------------------------
    print("\nRunning Joint AV-RAG retrieval...")
    joint_results = rag.joint_rag(t_embed, v_embed, a_embed, c_embed, k=TOP_K)

    print("\n================ Joint AV-RAG Results ================")
    for result in joint_results:
        print(result)

    # -------------------------------------------------
    # Salient Frame Selection (SFS)
    # -------------------------------------------------
    print("\n================ Running SFS on Top-1 Videos ================")

    for q_idx, result in enumerate(joint_results):
        query_key = list(result.keys())[0]
        top1_video_id = result[query_key][0]["file"]

        top1_video_path = next(p for p in video_paths if top1_video_id in p)
        top1_audio_path = next(p for p in audio_paths if top1_video_id in p)

        print(f"\nQuery {q_idx}: {query_key}")
        print(f"Top-1 video: {top1_video_id}")

        # ---- Sample frames ----
        frames, frame_indices = sample_frames(top1_video_path, m=NUM_FRAMES)

        # ---- Encode vision ----
        vision_embed = encode_frames_with_imagebind(rag, frames)

        # ---- Audio alignment ----
        vr = decord.VideoReader(top1_video_path)
        fps = vr.get_avg_fps()

        audio_clips = sample_audio_windows(top1_audio_path, frame_indices, fps)

        # Save temporary audio clips and ensure cleanup afterwards
        with tempfile.TemporaryDirectory() as tmp_audio_dir:
            tmp_audio_paths = []

            for i, clip in enumerate(audio_clips):
                tmp_path = os.path.join(tmp_audio_dir, f"{i}.wav")
                torchaudio.save(tmp_path, clip, 16000)
                tmp_audio_paths.append(tmp_path)

            audio_embed = rag.encode(tmp_audio_paths, ModalityType.AUDIO)["embeddings"]

        # ---- Fuse AV ----
        z = vision_embed * audio_embed

        # ---- Run SFS ----
        selected_indices = rag.sfs(z, k=SFS_K, gamma=GAMMA)

        print("Candidate frame indices:", frame_indices.tolist())
        print("Selected SFS indices:", selected_indices)
        print("Selected frame numbers:", frame_indices[selected_indices].tolist())

        # ---- Visualize Q Matrix ----
        Q_mat = rag.build_sfs_Q(z, gamma=GAMMA).detach().cpu().numpy()  # noqa: N806

        plt.figure(figsize=(6, 5))
        plt.imshow(Q_mat, cmap="viridis")
        plt.colorbar()
        plt.title(f"Q Matrix - Video {top1_video_id}")
        plt.xlabel("Frame index")
        plt.ylabel("Frame index")
        plt.tight_layout()

        save_path = f"tmp_Q_{top1_video_id}_q{q_idx}.jpg"
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"Saved Q heatmap to {save_path}")
