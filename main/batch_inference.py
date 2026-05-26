#!/usr/bin/env python
"""Batch inference: load model + wav2vec2 once, iterate over a directory of wavs.

Output layout matches main/demo_npy.py:
    {demo_npy_save_folder}/{audio_name}/condition_{condition}_subject_{subject}.npy
"""
import os
import glob
import pickle
import re
import time
from collections import Counter, defaultdict

import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor

from base.utilities import get_parser
from models import get_model
from models.utils import init_biased_mask, PeriodicPositionalEncoding
from base.baseTrainer import load_state_dict

MAX_SEQ_LEN = 2048  # default 600 caps audio at ~20s; bump to allow longer clips

cfg = get_parser()


def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _prefix(stem):
    # TalkingHead-1KH pattern: {id}_{seg}_S{start}_E{end}_L..._T..._R..._B...
    m = re.search(r'_\d+_S\d+_E\d+', stem)
    if m:
        return stem[:m.start()]
    return re.sub(r'_\d+$', '', stem)


def main():
    device = _get_device()
    cfg.device = str(device)
    print(f"=> using device: {device}")

    model = get_model(cfg)
    model = model.to(device)

    if not os.path.isfile(cfg.model_path):
        raise RuntimeError(f"No checkpoint found at '{cfg.model_path}'")
    print(f"=> loading checkpoint '{cfg.model_path}'")
    checkpoint = torch.load(cfg.model_path, map_location=lambda s, l: s.cpu())
    load_state_dict(model, checkpoint['state_dict'], strict=False)

    # The default model caps sequence length at 600 frames (~20s at 30 fps).
    # Rebuild biased_mask and PPE for longer audio; both are deterministic
    # (sinusoidal/ALiBi), so it's safe to extend after loading the checkpoint.
    model.biased_mask = init_biased_mask(n_head=4, max_seq_len=MAX_SEQ_LEN, period=cfg.period)
    new_ppe = PeriodicPositionalEncoding(cfg.feature_dim, period=cfg.period, max_seq_len=MAX_SEQ_LEN).to(device)
    model.PPE = new_ppe

    model.eval()
    print(f"=> loaded checkpoint")

    save_folder = cfg.demo_npy_save_folder
    os.makedirs(save_folder, exist_ok=True)

    template_file = os.path.join(cfg.data_root, cfg.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    condition = cfg.condition
    subject = cfg.subject

    train_subjects_list = cfg.train_subjects.split(" ")
    one_hot_labels = np.eye(len(train_subjects_list))
    one_hot = one_hot_labels[train_subjects_list.index(condition)]
    one_hot = np.reshape(one_hot, (-1, one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device)

    template = templates[subject].reshape((-1))
    template = np.reshape(template, (-1, template.shape[0]))
    template = torch.FloatTensor(template).to(device)

    processor = Wav2Vec2Processor.from_pretrained(cfg.wav2vec2model_path)

    wav_dir = cfg.demo_wav_dir_path
    if not wav_dir or not os.path.isdir(wav_dir):
        raise RuntimeError(f"demo_wav_dir_path must point to a directory, got: {wav_dir!r}")

    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        print(f"No .wav files found in {wav_dir}")
        return
    print(f"Found {len(wav_files)} wav files in {wav_dir}")

    def out_path(stem):
        return os.path.join(save_folder, stem, f"condition_{condition}_subject_{subject}.npy")

    existing_prefix_counts = Counter()
    pending_by_prefix = defaultdict(list)
    skipped = 0
    for f in wav_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if os.path.isfile(out_path(stem)):
            existing_prefix_counts[_prefix(stem)] += 1
            skipped += 1
        else:
            pending_by_prefix[_prefix(stem)].append(f)

    ordered = []
    prefix_counts = dict(existing_prefix_counts)
    while pending_by_prefix:
        chosen = min(pending_by_prefix, key=lambda p: (prefix_counts.get(p, 0), p))
        ordered.append(pending_by_prefix[chosen].pop(0))
        prefix_counts[chosen] = prefix_counts.get(chosen, 0) + 1
        if not pending_by_prefix[chosen]:
            del pending_by_prefix[chosen]

    total = len(wav_files)
    processed = 0
    failed = 0
    file_times = []
    batch_start = time.monotonic()

    for i, wav_file in enumerate(ordered, start=1):
        stem = os.path.splitext(os.path.basename(wav_file))[0]
        target = out_path(stem)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        global_idx = skipped + i
        try:
            file_start = time.monotonic()
            speech_array, _ = librosa.load(wav_file, sr=16000)
            audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
            audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
            audio_feature = torch.FloatTensor(audio_feature).to(device)

            with torch.no_grad():
                prediction = model.predict(audio_feature, template, one_hot)
                prediction = prediction.squeeze()
                np.save(target, prediction.detach().cpu().numpy())

            elapsed = time.monotonic() - file_start
            file_times.append(elapsed)
            processed += 1

            avg = sum(file_times) / len(file_times)
            remaining = total - global_idx
            eta = int(avg * remaining)
            print(f"[{global_idx}/{total}] {stem} -> {target} ({elapsed:.1f}s, avg {avg:.1f}s, ETA {eta//60}m{eta%60:02d}s)")
        except Exception as e:
            failed += 1
            print(f"[{global_idx}/{total}] FAIL {stem}: {e}")

    total_elapsed = int(time.monotonic() - batch_start)
    print(f"\nDone — processed: {processed}, skipped: {skipped}, failed: {failed}, total: {total}")
    print(f"Total time: {total_elapsed//60}m{total_elapsed%60:02d}s")


if __name__ == '__main__':
    main()
