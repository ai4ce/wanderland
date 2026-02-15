#!/usr/bin/env python3
"""
Render first-person navigation videos for UrbanSim scenes using GSplat checkpoints.

This script keeps the pipeline minimal:
  * load navigation episodes (VLN-CE style) for a scene;
  * generate in-memory camera pose sequences using generate_cam_poses.generate_cam_sequence;
  * load a single GSplat checkpoint once and extract splat tensors;
  * interpolate the camera trajectory and call gsplat.rendering.rasterization directly;
  * save RGB-only MP4 videos for each episode.

Optionally, it can also call Gemini (google-generativeai) to generate VLN-style
instructions per episode and write them back into episodes.json.

No training data, dataset parser, or Runner class dependencies are required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

import imageio
import cv2
import numpy as np
import torch
import tqdm

try:
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    genai = None

# Newer Gemini SDK (preferred/required for some Gemini 3 preview models).
# Library name on PyPI: google-genai
try:
    from google import genai as genai2  # type: ignore
    from google.genai import types as genai2_types  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    genai2 = None
    genai2_types = None

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader

from navigation.generate_cam_poses import generate_cam_sequence


def _load_module(rel_path: str, name: str):
    """Dynamically load a Python module from the repository."""
    module_path = PROJECT_ROOT / rel_path
    loader = SourceFileLoader(name, str(module_path))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


_traj_module = _load_module("reconstruction/gsplat/examples/datasets/traj.py", "_traj")
generate_interpolated_path = _traj_module.generate_interpolated_path


INSTRUCTION_DIR_NAME = "instructions"
DEFAULT_MODEL_NAME = "gemini-2.5-pro"
DEFAULT_INSTRUCTION_RETRIES = 3
DEFAULT_RETRY_DELAY = 5.0  # seconds

DEFAULT_PROMPT_TEMPLATE = Template(
    """You are a VLN-CE (Vision-and-Language Navigation in Continuous Environments) instruction generator.\n"
    "A first-person (ego-view) video is provided, showing a complete walking trajectory from a fixed start point to a destination.\n\n"
    "Context:\n"
    "- Scene id: $scene_id\n"
    "- Episode id: $episode_id\n"
    "- Reference path length: $ref_len waypoints\n"
    "- Turn summary: $turn_summary\n\n"
    "On-screen guidance overlays are present for reference in the video frames:\n"
    "- START / END mark the beginning and completion of the trajectory.\n"
    "- \"Forward\" indicates translational motion segments.\n"
    "- \"<<<\" / \">>>\" indicate in-place turns (the total angle is shown).\n"
    "Use these overlays only as hints to understand motion phases. Do not rely solely on the text or transcribe it; base descriptions on the visual scene itself.\n\n"
    "Objective:\n"
    "- Produce a concise, executable, and reproducible navigation instruction so an agent at the same start pose can reach the same destination using only this instruction.\n"
    "- Base descriptions strictly on what is observable in the video; avoid speculation or outside knowledge.\n\n"
    "Style and Constraints (critical):\n"
    "- Use first-person phrasing (e.g., 'walk forward', 'turn left', 'go through the door', 'follow the hallway').\n"
    "- Structure the instruction into steps; each step should include 'action + landmark/reference + distance'.\n"
    "- Provide conservative hints for turning (left/right with an approximate angle) and rough distance (short/medium/long or about N steps/meters).\n"
    "- IMPORTANT: The very first step must establish the initial facing direction using a unique visual reference in the START frame (e.g., 'At START, face the red EXIT sign / the double doors / the bright hallway').\n"
    "- Do NOT start with 'walk forward' or 'turn X degrees' unless you also state what you are aligning to (what you will face after the turn).\n"
    "- Include a clear stop condition (e.g., stop at a distinctive landmark or when facing a particular direction).\n"
    "- No hallucinations: do not mention spaces/landmarks not visible in the video. If uncertain, use cautious wording (e.g., 'short distance', 'about N steps').\n\n"
    "Landmark Uniqueness (important):\n"
    "- Make each landmark description discriminative to reduce ambiguity.\n"
    "- Prefer distinctive attributes: color, shape, material/texture, printed text/symbols/icons, lighting patterns, and count.\n"
    "- For repeated objects (cars/benches/posters/doors), prefer explicit counts/ordinals (e.g., 'pass 3 benches', 'stop by the 6th parked car', 'the second poster on the right') instead of vague plurals ('a row of cars', 'a series of benches').\n"
    "- Add relative position to the path or scene: 'on the right wall', 'at the corner', 'beside the doorway', 'ahead at the T-junction'.\n"
    "- Combine at least two cues when possible (e.g., color + shape, or color + location).\n"
    "- Avoid generic nouns alone ('the trash bin'); prefer descriptive variants ('the red square trash bin').\n"
    "- Disambiguate duplicates (e.g., 'the second blue poster on the right').\n"
    "- Prefer static fixtures over transient objects; only reference moving items if clearly stable throughout the clip.\n\n"
    "Output format (must be valid JSON only):\n"
    "{\n"
    "  \"navigation_style\": \"VLN-CE\",\n"
    "  \"goal\": string,\n"
    "  \"steps\": [\n"
    "    {\n"
    "      \"action\": string,\n"
    "      \"landmark\": string,\n"
    "      \"hint\": string\n"
    "    }\n"
    "  ],\n"
    "  \"key_landmarks\": [\n"
    "    {\n"
    "      \"name\": string,\n"
    "      \"type\": string\n"
    "    }\n"
    "  ],\n"
    "  \"stop_condition\": string,\n"
    "  \"instruction\": string\n"
    "}\n\n"
    "Return strictly valid JSON only, with no explanations or markdown fences."""
)


def load_prompt_template(prompt_file: Optional[str]) -> Optional[str]:
    if not prompt_file:
        return None
    path = Path(prompt_file)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_prompt(
    scene_id: str,
    episode_id: int,
    ref_len: int,
    turn_summary: str,
    template_text: Optional[str],
) -> str:
    template = Template(template_text) if template_text else DEFAULT_PROMPT_TEMPLATE
    return template.safe_substitute(
        scene_id=scene_id,
        episode_id=episode_id,
        ref_len=ref_len,
        turn_summary=turn_summary,
    )


def build_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Construct a simple pinhole intrinsics matrix."""
    fx = (width * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
    fy = fx
    cx = width * 0.5
    cy = height * 0.5
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] -> rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def frames_to_camtoworld(frames: Sequence[Dict]) -> np.ndarray:
    """Convert a list of pose dicts into [N, 4, 4] cam-to-world matrices."""
    N = len(frames)
    camtoworld = np.zeros((N, 4, 4), dtype=np.float32)
    for i, frame in enumerate(frames):
        R = quat_xyzw_to_rotmat(np.asarray(frame["quaternion_xyzw"], dtype=np.float32))
        R[:, 1] *= -1.0
        R[:, 2] *= -1.0
        t = np.asarray(frame["position"], dtype=np.float32)
        camtoworld[i, :3, :3] = R
        camtoworld[i, :3, 3] = t
        camtoworld[i, 3, 3] = 1.0
    return camtoworld


def _compute_yaw_sequence(c2w_seq: np.ndarray) -> List[float]:
    """Compute yaw (rad) per pose from cam-to-world matrices.

    Yaw is computed from forward vector f = -R[:,2] using atan2(fx, -fz).
    """
    yaws: List[float] = []
    for i in range(c2w_seq.shape[0]):
        R = c2w_seq[i, :3, :3]
        fwd = -R[:, 2]
        yaw = math.atan2(float(fwd[0]), float(-fwd[2]))
        yaws.append(yaw)
    return yaws


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _delta_yaw(a_new: float, a_old: float) -> float:
    return _wrap_to_pi(a_new - a_old)


def _label_from_block(c2w_block: np.ndarray) -> str:
    """Create a human-friendly ASCII label for a block.

    Use only ASCII to ensure OpenCV Hershey fonts render correctly.
    """
    if c2w_block.shape[0] <= 1:
        return "Forward"
    yaws = _compute_yaw_sequence(c2w_block)
    dyaw = _delta_yaw(yaws[-1], yaws[0])
    deg = abs(dyaw) * 180.0 / math.pi
    if abs(deg) < 1e-2:
        return "Forward"
    # Two-line label: first line arrow, second line text (ASCII only)
    return (f"<<<\nTurning Left {deg:.0f} deg" if dyaw > 0 else f">>>\nTurning Right {deg:.0f} deg")


def _annotate_frame(frame_u8: np.ndarray, text: str) -> np.ndarray:
    """Overlay centered text (supports \n for multiline) with white fill + red outline.

    Convert RGB->BGR for OpenCV drawing and convert back to keep colors correct.
    """
    H, W = frame_u8.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thick_fill = 2
    thick_outline = thick_fill + 2
    color_fill = (255, 255, 255)   # white fill (BGR)
    color_outline = (0, 0, 255)    # red outline (BGR)

    lines = str(text).splitlines() if text else [""]
    # Measure total height to vertically place near top with small margin
    line_sizes = [cv2.getTextSize(line, font, font_scale, thick_fill)[0] for line in lines]
    line_height = max((sz[1] for sz in line_sizes), default=20)
    margin_top = 20
    y = margin_top + line_height

    bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
    for line, (tw, th) in zip(lines, line_sizes):
        x = max(10, (W - tw) // 2)
        # Outline then fill for each line
        cv2.putText(bgr, line, (x, y), font, font_scale, color_outline, thick_outline, cv2.LINE_AA)
        cv2.putText(bgr, line, (x, y), font, font_scale, color_fill, thick_fill, cv2.LINE_AA)
        y += line_height + 4  # small line spacing

    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb


def load_splats_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[Dict[str, torch.Tensor], Optional[int]]:
    """Load Gaussian splats from a GSplat checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "splats" not in ckpt:
        raise RuntimeError(f"Invalid checkpoint: 'splats' key missing in {ckpt_path}")
    splats_raw = ckpt["splats"]

    means = splats_raw["means"].to(device)  # [N, 3]
    quats = splats_raw["quats"].to(device)  # [N, 4]
    scales = torch.exp(splats_raw["scales"]).to(device)  # [N, 3]
    opacities = torch.sigmoid(splats_raw["opacities"]).to(device)  # [N]
    sh0 = splats_raw["sh0"].to(device)  # [N, 1, 3]
    shN = splats_raw["shN"].to(device)  # [N, K-1, 3]

    colors = torch.cat([sh0, shN], dim=1)  # [N, K, 3]
    K = colors.shape[1]
    root = int(round(math.sqrt(K)))
    sh_degree = root - 1 if root * root == K else None

    splats = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }
    return splats, sh_degree


def interpolate_camtoworld(camtoworld_4x4: np.ndarray, factor: int) -> np.ndarray:
    """Interpolate camera trajectory using B-spline interpolation."""
    if factor <= 1 or camtoworld_4x4.shape[0] <= 1:
        return camtoworld_4x4
    path_3x4 = camtoworld_4x4[:, :3, :]
    interp_3x4 = generate_interpolated_path(path_3x4, n_interp=factor)
    pad = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (len(interp_3x4), 1))
    camtoworld_interp = np.concatenate([interp_3x4, pad[:, None, :]], axis=1)
    return camtoworld_interp


def _is_turn_note(note: str) -> bool:
    return note in {"key_pre_turn", "key_turn_step", "key_post_turn", "start_look_next"}


def _strip_initial_start_frame(frames: Sequence[Dict]) -> List[Dict]:
    """Drop the start_given frame when a start_look_next immediately follows.

    The initial start_given quaternion can face the opposite way from the intended
    forward motion; keeping only the start_look_next frame ensures the first video
    frame already looks down the reference path.
    """
    if (
        len(frames) >= 2
        and frames[0].get("note") == "start_given"
        and frames[1].get("note") == "start_look_next"
    ):
        return list(frames[1:])
    return list(frames)


def _segment_frames(frames: Sequence[Dict]) -> List[tuple[str, List[Dict]]]:
    """Split frames into move/turn blocks based on note semantics.

    - turn block: key_* notes and the initial start_look_next (and its preceding start_given).
    - move block: everything else (including solitary start_given).
    """
    blocks: List[tuple[str, List[Dict]]] = []
    i = 0
    n = len(frames)
    while i < n:
        note = frames[i].get("note", "")
        # Treat start_given + immediate start_look_next as a turn block together
        if note == "start_given" and i + 1 < n and frames[i + 1].get("note", "") == "start_look_next":
            j = i + 2
            # include any subsequent turn steps just in case (unlikely at head)
            while j < n and _is_turn_note(frames[j].get("note", "")):
                j += 1
            blocks.append(("turn", frames[i:j]))
            i = j
            continue

        seg_type = "turn" if _is_turn_note(note) else "move"
        j = i + 1
        while j < n:
            next_note = frames[j].get("note", "")
            if seg_type == "turn":
                if not _is_turn_note(next_note):
                    break
            else:
                if _is_turn_note(next_note):
                    break
            j += 1
        blocks.append((seg_type, frames[i:j]))
        i = j
    return blocks


def _c2w_from_frames(frames: Sequence[Dict]) -> np.ndarray:
    return frames_to_camtoworld(frames)


def _interpolate_move_block(c2w_block: np.ndarray, interp_factor: int) -> np.ndarray:
    """Interpolate positions for move block while keeping orientation fixed."""
    if interp_factor <= 1 or c2w_block.shape[0] < 2:
        return c2w_block
    # Keep a constant orientation from the first frame in the block
    R_fixed = c2w_block[0, :3, :3].copy()
    path_3x4 = c2w_block[:, :3, :]
    try:
        interp_3x4 = generate_interpolated_path(path_3x4, n_interp=interp_factor)
        # Rebuild 4x4 and overwrite rotation with R_fixed
        T = len(interp_3x4)
        pad = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (T, 1))
        c2w_interp = np.concatenate([interp_3x4, pad[:, None, :]], axis=1)
        c2w_interp[:, :3, :3] = R_fixed[None, ...]
        return c2w_interp.astype(np.float32)
    except Exception:
        # Fallback to raw when spline fails (too few points, etc.)
        return c2w_block


def summarize_turns(frames: Sequence[Dict]) -> str:
    """Produce a concise textual summary of move/turn blocks for prompting."""
    frames = _strip_initial_start_frame(frames)
    blocks = _segment_frames(frames)
    labels: List[str] = []
    for seg_type, seg_frames in blocks:
        if seg_type == "move":
            labels.append("Forward")
        else:
            label = _label_from_block(_c2w_from_frames(seg_frames))
            labels.append(label.replace("\n", " "))
    return " -> ".join(labels) if labels else "Forward"


def compute_geodesic_distance(frames: Sequence[Dict]) -> float:
    """Approximate geodesic distance by summing Euclidean steps along the camera sequence."""
    if not frames:
        return 0.0
    dist = 0.0
    prev = np.asarray(frames[0]["position"], dtype=np.float64)
    for frame in frames[1:]:
        cur = np.asarray(frame["position"], dtype=np.float64)
        dist += float(np.linalg.norm(cur - prev))
        prev = cur
    return dist


def wait_for_active(file_obj: Any, timeout_s: float = 900.0, poll_s: float = 2.0) -> Any:
    """Poll generative file upload until ACTIVE or failure."""
    if genai is None:
        return file_obj
    start = time.time()
    name = getattr(file_obj, "name", None)
    if not name:
        return file_obj
    while True:
        handle = genai.get_file(name)
        state = getattr(handle, "state", None)
        status = getattr(state, "name", None) if state else None
        if status == "ACTIVE":
            return handle
        if status == "FAILED":
            raise RuntimeError(f"File processing failed for {name} ({handle.state})")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for file to become ACTIVE: {name}")
        time.sleep(poll_s)


def _is_gemini3_model_name(model_name: str) -> bool:
    s = (model_name or "").strip().lower()
    s = s.removeprefix("models/")
    return s.startswith("gemini-3-")


def _normalize_model_name_for_genai2(model_name: str) -> str:
    # Keep for backwards compatibility; actual requests use a candidate list
    # to handle both "gemini-3-*" and "models/gemini-3-*" forms.
    return (model_name or "").strip()


def _genai2_model_candidates(model_name: str) -> List[str]:
    """Try both doc-style and list-style model ids for google-genai."""
    raw = (model_name or "").strip()
    if not raw:
        return []
    # The docs often use "gemini-3-pro-preview", while `client.models.list()`
    # returns "models/gemini-3-pro-preview". In practice, deployments vary.
    #
    # IMPORTANT: for video/multimodal calls, many doc snippets use the full
    # resource name (e.g. "models/gemini-3-pro-preview"), so we try that first.
    short = raw[len("models/") :] if raw.startswith("models/") else raw
    full = raw if raw.startswith("models/") else f"models/{raw}"
    cands: List[str] = [full, short]
    # Deduplicate while preserving order.
    seen = set()
    out: List[str] = []
    for c in cands:
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _get_api_key(args: argparse.Namespace) -> str:
    # Keep compatibility with existing scripts that export GOOGLE_API_KEY.
    # Also allow GEMINI_API_KEY (common in docs/snippets).
    return (
        (args.api_key or "").strip()
        or (os.environ.get("GOOGLE_API_KEY") or "").strip()
        or (os.environ.get("GEMINI_API_KEY") or "").strip()
    )


def _sanitize_ssl_env_for_sdk() -> None:
    """
    Some HPC environments export SSL_CERT_FILE/SSL_CERT_DIR pointing to host paths
    that don't exist inside the container, causing TLS initialization to crash
    (notably in `google-genai` which uses httpx/ssl.create_default_context()).
    """
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        if key in os.environ and os.environ.get(key):
            os.environ.pop(key, None)


def wait_for_active_genai2(
    client: Any, file_obj: Any, timeout_s: float = 900.0, poll_s: float = 2.0
) -> Any:
    """Poll google-genai file upload until ACTIVE or failure."""
    start = time.time()
    name = getattr(file_obj, "name", None)
    if not name:
        return file_obj
    while True:
        handle = client.files.get(name=name)
        state = getattr(handle, "state", None)
        status = getattr(state, "name", None) if state else None
        if status == "ACTIVE":
            return handle
        if status == "FAILED":
            raise RuntimeError(f"File processing failed for {name} ({handle.state})")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for file to become ACTIVE: {name}")
        time.sleep(poll_s)


def request_instruction_json_genai2(
    client: Any,
    model_name: str,
    video_path: Path,
    prompt_text: str,
    *,
    temperature: float,
    retries: int,
    retry_delay: float,
) -> Optional[Dict]:
    """Invoke Gemini via the newer google-genai SDK (best for Gemini 3)."""
    last_error: Optional[Exception] = None
    model_candidates = _genai2_model_candidates(model_name)
    if not model_candidates:
        print("[instruction] gemini3 requested but model name is empty; skipping.")
        return None
    for attempt in range(retries):
        file_obj = None
        uploaded = False
        step = "init"
        try:
            cfg = None
            if genai2_types is not None:
                cfg = genai2_types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                )
            response = None
            last_not_found = None

            # Prefer inline bytes for small videos (<20MB). This matches the official
            # "Video data inline" doc snippet:
            #   contents=types.Content(parts=[inline_data Blob, text Part])
            # and avoids Files API incompatibilities observed with some Gemini 3 previews.
            inline_limit_bytes = 18 * 1024 * 1024  # leave margin for request overhead
            inline_ok = False
            try:
                inline_ok = video_path.stat().st_size <= inline_limit_bytes
            except OSError:
                inline_ok = False

            if inline_ok and genai2_types is not None:
                step = "read_video_bytes"
                video_bytes = video_path.read_bytes()
                contents = genai2_types.Content(
                    parts=[
                        genai2_types.Part(
                            inline_data=genai2_types.Blob(data=video_bytes, mime_type="video/mp4")
                        ),
                        genai2_types.Part(text=prompt_text),
                    ]
                )
                step = "generate_content_inline"
                for cand in model_candidates:
                    try:
                        response = client.models.generate_content(
                            model=cand,
                            contents=contents,
                            config=cfg,
                        )
                        break
                    except Exception as exc:
                        msg = str(exc)
                        if "404" in msg and ("NOT_FOUND" in msg or "not found" in msg.lower()):
                            last_not_found = exc
                            continue
                        raise
            else:
                step = "upload_file"
                file_obj = client.files.upload(file=str(video_path))
                uploaded = True

                step = "wait_for_active"
                file_obj = wait_for_active_genai2(client, file_obj)

                step = "generate_content_file"
                for cand in model_candidates:
                    try:
                        response = client.models.generate_content(
                            model=cand,
                            contents=[file_obj, prompt_text],
                            config=cfg,
                        )
                        break
                    except Exception as exc:
                        msg = str(exc)
                        # Some deployments only accept one of the two ID formats.
                        if "404" in msg and ("NOT_FOUND" in msg or "not found" in msg.lower()):
                            last_not_found = exc
                            continue
                        raise
            if response is None:
                raise RuntimeError(f"model not found for any of: {model_candidates} ({last_not_found})")

            step = "parse_response"
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise ValueError("Empty response text from model")
            parsed = json.loads(text)
            if "instruction" not in parsed or not parsed["instruction"]:
                raise ValueError("Model response missing 'instruction' field")
            return parsed
        except Exception as exc:  # pragma: no cover - network/remote failures
            wrapped = RuntimeError(f"{step} failed: {exc}")
            last_error = wrapped
            if attempt + 1 >= retries:
                break
            delay = retry_delay * (2 ** attempt)
            print(f"[instruction] attempt {attempt + 1} failed: {wrapped}. Retrying in {delay:.1f}s")
            time.sleep(delay)
        finally:
            if uploaded and file_obj is not None:
                try:
                    client.files.delete(name=file_obj.name)
                except Exception:
                    pass
    if last_error:
        print(f"[instruction] giving up after {retries} attempts: {last_error}")
    return None


def request_instruction_json(
    model: Any,
    video_path: Path,
    prompt_text: str,
    retries: int,
    retry_delay: float,
) -> Optional[Dict]:
    """Invoke the Gemini model with the video+prompt, returning parsed JSON."""
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        file_obj = None
        step = "init"
        try:
            step = "upload_file"
            file_obj = genai.upload_file(path=str(video_path))

            step = "wait_for_active"
            file_obj = wait_for_active(file_obj)

            step = "generate_content"
            response = model.generate_content([file_obj, {"text": prompt_text}])

            step = "parse_response"
            text = (response.text or "").strip()
            if not text:
                raise ValueError("Empty response text from model")
            parsed = json.loads(text)
            if "instruction" not in parsed or not parsed["instruction"]:
                raise ValueError("Model response missing 'instruction' field")
            return parsed
        except Exception as exc:  # pragma: no cover - network/remote failures
            wrapped = RuntimeError(f"{step} failed: {exc}")
            last_error = wrapped
            if attempt + 1 >= retries:
                break
            delay = retry_delay * (2 ** attempt)
            print(
                f"[instruction] attempt {attempt + 1} failed: {wrapped}. Retrying in {delay:.1f}s"
            )
            time.sleep(delay)
        finally:
            if file_obj is not None:
                try:
                    genai.delete_file(file_obj.name)
                except Exception:
                    pass
    if last_error:
        print(f"[instruction] giving up after {retries} attempts: {last_error}")
    return None


def save_instruction_payload(directory: Path, episode_id: int, payload: Dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"episode_{episode_id:04d}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def apply_instructions_to_payload(
    payload: Dict[str, Any],
    per_episode: Dict[int, Dict[str, Any]],
) -> int:
    """Populate instruction fields inside episodes.json payload."""
    updated = 0
    for episode in payload.get("episodes", []):
        eid = episode.get("episode_id")
        if eid in per_episode:
            record = per_episode[eid]
            episode["instruction"] = record["instruction"]
            if "geodesic_distance" in record:
                episode["geodesic_distance"] = record["geodesic_distance"]
            if "goal_radius" in record:
                goals = episode.get("goals")
                if isinstance(goals, list) and goals:
                    goals[0]["radius"] = record["goal_radius"]
            updated += 1
    return updated


def rebuild_valid_episodes(
    payload: Dict[str, Any],
    valid_ids: Dict[int, Dict[str, Any]],
    scene_id: str,
) -> int:
    """Keep only episodes that obtained instructions, renumber ids, and sanitize scene ids."""
    new_episodes: List[Dict[str, Any]] = []
    next_id = 1
    for episode in payload.get("episodes", []):
        eid = episode.get("episode_id")
        if eid not in valid_ids:
            continue
        episode = dict(episode)
        episode["episode_id"] = next_id
        episode["scene_id"] = scene_id
        new_episodes.append(episode)
        next_id += 1
    payload["episodes"] = new_episodes
    return len(new_episodes)


@torch.no_grad()
def render_episode(
    splats: Dict[str, torch.Tensor],
    sh_degree: Optional[int],
    frames: Sequence[Dict],
    *,
    width: int,
    height: int,
    hfov_deg: float,
    fps: int,
    interp_factor: int,
    near_plane: float,
    far_plane: float,
    out_path: Path,
    device: torch.device,
    annotate: bool = True,
    yaw_thr_deg: float = 1.0,
) -> None:
    """Render a single episode to an MP4 file with segment-wise interpolation.

    - Move blocks: interpolate (positions only), keep orientation fixed.
    - Turn blocks: no interpolation, preserve per-frame rotations.
    """
    # Import here so instruction-only runs don't import/initialize gsplat CUDA bits.
    from gsplat.rendering import rasterization

    frames = _strip_initial_start_frame(frames)
    blocks = _segment_frames(frames)

    c2w_list: List[np.ndarray] = []
    for seg_type, seg_frames in blocks:
        c2w_block = _c2w_from_frames(seg_frames)
        if seg_type == "move":
            c2w_block = _interpolate_move_block(c2w_block, interp_factor)
        # else: turn block, keep as-is
        c2w_list.append(c2w_block)

    camtoworld = np.concatenate(c2w_list, axis=0) if c2w_list else np.zeros((0, 4, 4), np.float32)
    camtoworld_t = torch.from_numpy(camtoworld).to(device).float()
    viewmats = torch.linalg.inv(camtoworld_t)  # [T, 4, 4]

    K_np = build_intrinsics(width, height, hfov_deg)
    K_t = torch.from_numpy(K_np).to(device).float()[None]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps)

    # Prepare per-frame labels from block semantics
    labels_per_frame: List[str] = []
    for (seg_type, seg_frames), c2w_block in zip(blocks, c2w_list):
        if seg_type == "move":
            label_text = "Forward"
        else:
            label_text = _label_from_block(c2w_block)
        labels_per_frame.extend([label_text] * c2w_block.shape[0])

    for i in tqdm.trange(viewmats.shape[0], desc=f"Render {out_path.name}", leave=False):
        vm = viewmats[i : i + 1]
        rgb, _, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=splats["colors"],
            viewmats=vm,
            Ks=K_t,
            width=width,
            height=height,
            sh_degree=sh_degree,
            render_mode="RGB",
            near_plane=near_plane,
            far_plane=far_plane,
            camera_model="pinhole",
            packed=True,
        )
        frame = torch.clamp(rgb[..., :3], 0.0, 1.0).squeeze(0).cpu().numpy()
        frame_u8 = (frame * 255).astype(np.uint8)
        if annotate:
            label = labels_per_frame[i] if i < len(labels_per_frame) else ""
            if i == 0:
                label = "START"
            # mark the final frame as END for clear termination
            if i == len(labels_per_frame) - 1:
                label = "END"
            frame_u8 = _annotate_frame(frame_u8, label)
        writer.append_data(frame_u8)
    writer.close()


def collect_episode_frames(
    episodes_json: Path,
    *,
    cam_height: float,
    key_radius: float,
    max_turn_deg: float,
    max_nav_height_above_floor_m: float,
    episode_ids: Optional[Sequence[int]] = None,
) -> List[Dict]:
    """Load episodes and generate pose sequences in memory."""
    payload = json.loads(episodes_json.read_text())
    keep: List[Dict] = []
    target = set(episode_ids) if episode_ids else None
    skipped = 0
    skipped_high = 0

    # Some scenes have navmesh artifacts where "ceiling" surfaces become walkable.
    # These episodes tend to have start/end points significantly above the dominant
    # ground plane in the scene (note: this dataset uses -Y as up).
    floor_y = None
    if max_nav_height_above_floor_m > 0:
        ys: List[float] = []
        for ep in payload.get("episodes", []):
            for p in (ep.get("gt_locations") or []):
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        ys.append(float(p[1]))
                    except (TypeError, ValueError):
                        continue
        if len(ys) >= 10:
            floor_y = float(np.median(np.asarray(ys, dtype=np.float64)))

    for episode in payload.get("episodes", []):
        eid = episode.get("episode_id")
        if target is not None and eid not in target:
            continue

        if floor_y is not None:
            locs = episode.get("gt_locations") or []
            if isinstance(locs, list) and len(locs) >= 2:
                try:
                    start_y = float(locs[0][1])
                    end_y = float(locs[-1][1])
                    min_above = min(floor_y - start_y, floor_y - end_y)
                except Exception:
                    min_above = None
                if min_above is not None and min_above > max_nav_height_above_floor_m:
                    skipped += 1
                    skipped_high += 1
                    continue

        seq = generate_cam_sequence(
            episode,
            cam_height=cam_height,
            key_radius=key_radius,
            max_turn_deg=max_turn_deg,
        )
        if not seq:
            skipped += 1
            continue
        ref_len = len(episode.get("reference_path", []))
        keep.append({"episode_id": eid, "frames": seq, "ref_len": ref_len})

    if not keep:
        # If the caller asked for specific episode_ids, it is possible that all
        # requested episodes were filtered (e.g., ceiling-navmesh artifacts).
        # In that case, return an empty list so the caller can decide what to do.
        if target is not None:
            if skipped_high:
                print(f"[episodes] usable=0, skipped={skipped} (high_paths={skipped_high})")
            else:
                print(f"[episodes] usable=0, skipped={skipped}")
            return []
        raise RuntimeError(f"No usable episodes found in {episodes_json} (skipped {skipped}).")

    if skipped_high:
        print(f"[episodes] usable={len(keep)}, skipped={skipped} (high_paths={skipped_high})")
    else:
        print(f"[episodes] usable={len(keep)}, skipped={skipped}")
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Render UrbanSim navigation videos.")
    parser.add_argument("--scene", required=True, help="Scene id (e.g., 1EoD__...)")
    parser.add_argument(
        "--paths-root",
        default=os.environ.get("PATHS_ROOT") or str(PROJECT_ROOT / "paths"),
        help="Root containing per-scene nav outputs.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to GSplat checkpoint (.pt). Required unless --instructions-only.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write MP4 videos.",
    )
    parser.add_argument("--episode-ids", nargs="+", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render at most N episodes (stop once N videos exist/are produced).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing episode_*.mp4 videos under --output-dir (default: skip).",
    )
    parser.add_argument(
        "--generate-instructions",
        action="store_true",
        help="If set, call Gemini to generate per-episode instructions and write them back to episodes.json.",
    )
    parser.add_argument(
        "--instructions-only",
        action="store_true",
        help="Skip rendering/ckpt loading and only generate instructions using existing episode_*.mp4 files.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google AI Studio API key (fallback: env GOOGLE_API_KEY). Required when --generate-instructions.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Gemini model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature for instruction generation.")
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional path to a custom prompt template (Python Template syntax).",
    )
    parser.add_argument(
        "--instruction-retries",
        type=int,
        default=DEFAULT_INSTRUCTION_RETRIES,
        help="Number of retries for Gemini API calls per episode.",
    )
    parser.add_argument(
        "--instruction-retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help="Initial delay (seconds) between Gemini retries (exponential backoff).",
    )
    parser.add_argument(
        "--save-instruction-json",
        action="store_true",
        help="Save raw Gemini JSON per episode under nav/instructions/.",
    )
    parser.add_argument(
        "--instruction-dir-name",
        default=INSTRUCTION_DIR_NAME,
        help=f"Directory name under nav/ for saved instruction JSON (default: {INSTRUCTION_DIR_NAME}).",
    )
    parser.add_argument(
        "--overwrite-instructions",
        action="store_true",
        help="Regenerate instructions even if episodes.json already has one.",
    )
    parser.add_argument(
        "--episodes-out",
        default=None,
        help="If set, write the instruction-updated episodes JSON to this path instead of overwriting nav/episodes.json.",
    )
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--interp-factor", type=int, default=4)
    parser.add_argument("--cam-height", type=float, default=1.2)
    parser.add_argument("--key-radius", type=float, default=0.75)
    parser.add_argument(
        "--max-turn-deg",
        type=float,
        default=10.0,
        help="Maximum yaw change per pose at keypoints (degrees). Smaller values yield smoother turns.",
    )
    parser.add_argument(
        "--max-nav-height-above-floor-m",
        type=float,
        default=1.8,
        help=(
            "Filter out episodes whose start/end navmesh points are far above the scene's dominant ground plane "
            "(helps remove ceiling/navmesh artifacts). Set <=0 to disable."
        ),
    )
    parser.add_argument("--near", type=float, default=1e-3)
    parser.add_argument("--far", type=float, default=1e4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodes_json = Path(args.paths_root) / args.scene / "nav" / "episodes.json"
    if not episodes_json.is_file():
        raise FileNotFoundError(f"episodes.json not found: {episodes_json}")

    episodes_payload = json.loads(episodes_json.read_text())

    episodes = collect_episode_frames(
        episodes_json,
        cam_height=args.cam_height,
        key_radius=args.key_radius,
        max_turn_deg=args.max_turn_deg,
        max_nav_height_above_floor_m=float(args.max_nav_height_above_floor_m),
        episode_ids=args.episode_ids,
    )
    if not episodes:
        print("[episodes] No usable episodes after filtering; nothing to do.")
        return

    generate_instructions = bool(args.generate_instructions)
    instructions_only = bool(args.instructions_only)
    if instructions_only and not generate_instructions:
        raise RuntimeError("--instructions-only requires --generate-instructions.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splats: Optional[Dict[str, torch.Tensor]] = None
    sh_degree: Optional[int] = None
    if not instructions_only:
        if not args.ckpt:
            raise RuntimeError("--ckpt is required unless --instructions-only is set.")
        splats, sh_degree = load_splats_from_ckpt(Path(args.ckpt), device=device)

    model = None
    instruction_backend: str = ""
    instruction_model_name: str = ""
    instruction_dir = None
    prompt_template_text = None
    existing_instruction_ids: set[int] = set()

    if generate_instructions:
        _sanitize_ssl_env_for_sdk()
        api_key = _get_api_key(args)
        if not api_key:
            raise RuntimeError(
                "API key missing for instruction generation. Provide --api-key or set GOOGLE_API_KEY/GEMINI_API_KEY."
            )

        prompt_template_text = load_prompt_template(args.prompt_file)
        instruction_dir = Path(args.paths_root) / args.scene / "nav" / str(args.instruction_dir_name)
        existing_instruction_ids = {
            int(ep.get("episode_id", 0))
            for ep in episodes_payload.get("episodes", [])
            if isinstance(ep, dict) and ep.get("instruction")
        }

        # NOTE: We support *two* SDKs:
        # - google-generativeai (older): works for gemini-2.x/2.5 in our cluster tests.
        # - google-genai (newer): required for some Gemini 3 preview models (old SDK can 404).
        instruction_model_name = str(args.model).strip() or DEFAULT_MODEL_NAME
        if _is_gemini3_model_name(instruction_model_name):
            if genai2 is None:
                raise RuntimeError(
                    "Gemini 3 requested but google-genai is not installed. Install it (e.g., `uv pip install google-genai`)."
                )
            instruction_backend = "genai2"
            # Use the SDK default API version to match doc examples and avoid
            # deployment-specific mismatches (video calls in particular).
            model = genai2.Client(api_key=api_key)
        else:
            if genai is None:
                raise RuntimeError(
                    "google-generativeai is not installed. Install it (e.g., `uv pip install google-generativeai`) "
                    "or run without --generate-instructions."
                )
            instruction_backend = "genai1"
            genai.configure(api_key=api_key)
            # `google-generativeai` model names are usually full resource names (e.g., "models/gemini-2.5-pro").
            # We accept shorthand like "gemini-2.5-pro" and normalize it here.
            model_name = instruction_model_name
            if model_name and "/" not in model_name:
                model_name = f"models/{model_name}"
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": args.temperature,
                    "response_mime_type": "application/json",
                },
            )

    video_cap = args.limit if args.limit and args.limit > 0 else None
    videos_ready = 0
    videos_rendered = 0
    video_failures = 0
    skipped_existing_videos = 0

    instructions_written = 0
    instruction_failures = 0
    skipped_existing_instructions = 0
    per_episode_records: Dict[int, Dict[str, Any]] = {}

    # Priority sort within the scene: ep#1 first, then 3..8 ref points, then others.
    def _priority(item: Dict) -> tuple:
        eid = int(item.get("episode_id", 0))
        rlen = int(item.get("ref_len", 0))
        p0 = 0 if eid == 1 else (1 if 3 <= rlen <= 8 else 2)
        return (p0, eid)

    episodes = sorted(episodes, key=_priority)
    for episode in episodes:
        if video_cap is not None and videos_ready >= video_cap:
            break
        eid = episode["episode_id"]
        out_path = out_dir / f"episode_{eid:04d}.mp4"
        video_ok = out_path.is_file() and out_path.stat().st_size > 0

        if instructions_only:
            if not video_ok:
                video_failures += 1
                print(f"[render] episode={eid} missing video for instructions-only: {out_path}")
                continue
            skipped_existing_videos += 1
        elif video_ok and not args.overwrite:
            print(f"[render] episode={eid} exists -> skip (use --overwrite to rerender): {out_path}")
            skipped_existing_videos += 1
        else:
            print(f"[render] episode={eid} -> {out_path}")
            try:
                if splats is None:
                    raise RuntimeError("Internal error: splats not initialized for rendering.")
                render_episode(
                    splats,
                    sh_degree,
                    episode["frames"],
                    width=args.width,
                    height=args.height,
                    hfov_deg=args.hfov,
                    fps=args.fps,
                    interp_factor=args.interp_factor,
                    near_plane=args.near,
                    far_plane=args.far,
                    out_path=out_path,
                    device=device,
                )
            except Exception as exc:
                video_failures += 1
                print(f"[render] episode={eid} FAILED: {exc}")
                continue

            video_ok = out_path.is_file() and out_path.stat().st_size > 0
            if not video_ok:
                video_failures += 1
                print(f"[render] episode={eid} FAILED: output missing/empty: {out_path}")
                continue
            videos_rendered += 1

        videos_ready += 1

        if not generate_instructions:
            continue

        if (not args.overwrite_instructions) and eid in existing_instruction_ids:
            print(
                f"[instruction] episode={eid} already has instruction – skipping (use --overwrite-instructions to regenerate)."
            )
            skipped_existing_instructions += 1
            continue

        # `prompt_template_text` is allowed to be None (falls back to DEFAULT_PROMPT_TEMPLATE).
        if model is None:
            raise RuntimeError("Internal error: instruction generation requested but model not initialized.")

        turn_summary = summarize_turns(episode["frames"])
        prompt_text = build_prompt(
            scene_id=args.scene,
            episode_id=eid,
            ref_len=episode.get("ref_len", 0),
            turn_summary=turn_summary,
            template_text=prompt_template_text,
        )

        if instruction_backend == "genai2":
            instruction_payload = request_instruction_json_genai2(
                client=model,
                model_name=instruction_model_name,
                video_path=out_path,
                prompt_text=prompt_text,
                temperature=float(args.temperature),
                retries=max(1, args.instruction_retries),
                retry_delay=max(1e-3, args.instruction_retry_delay),
            )
        else:
            instruction_payload = request_instruction_json(
                model,
                out_path,
                prompt_text,
                retries=max(1, args.instruction_retries),
                retry_delay=max(1e-3, args.instruction_retry_delay),
            )

        if not instruction_payload:
            instruction_failures += 1
            continue

        instruction_text = instruction_payload.get("instruction")
        if not instruction_text:
            instruction_failures += 1
            continue

        geodesic_distance = compute_geodesic_distance(episode["frames"])
        if geodesic_distance <= 3.0:
            print(f"[instruction] episode={eid} geodesic distance too small; skipping.")
            instruction_failures += 1
            continue

        record = {
            "instruction": instruction_text,
            "geodesic_distance": geodesic_distance,
        }
        if geodesic_distance < 9.0:
            record["goal_radius"] = max(geodesic_distance / 3, 1.0)

        per_episode_records[eid] = record
        if args.save_instruction_json:
            if instruction_dir is None:
                raise RuntimeError("Internal error: instruction_dir missing.")
            save_instruction_payload(instruction_dir, eid, instruction_payload)

        print(f"[instruction] episode={eid} instruction captured (count={len(per_episode_records)})")

    if per_episode_records:
        applied = apply_instructions_to_payload(episodes_payload, per_episode_records)
        if applied:
            out_json = Path(args.episodes_out).expanduser() if args.episodes_out else episodes_json
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(
                json.dumps(episodes_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            instructions_written = applied
            print(f"[instruction] wrote instructions for {applied} episode(s) -> {out_json}")
        else:
            print("[instruction] No episodes updated; episodes.json left unchanged.")
    else:
        if generate_instructions:
            print("[instruction] No instructions captured; episodes.json left unchanged.")

    if videos_ready <= 0:
        raise RuntimeError(f"No videos rendered or found under: {out_dir}")

    print(
        "[done] "
        f"scene={args.scene} "
        f"videos_ready={videos_ready} "
        f"videos_rendered={videos_rendered} "
        f"video_failures={video_failures} "
        f"skipped_existing_videos={skipped_existing_videos} "
        f"instructions_written={instructions_written} "
        f"instruction_failures={instruction_failures} "
        f"skipped_existing_instructions={skipped_existing_instructions}"
    )


if __name__ == "__main__":
    main()
