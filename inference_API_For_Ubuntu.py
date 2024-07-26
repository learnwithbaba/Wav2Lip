from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import subprocess
import torch
import face_detection
from models import Wav2Lip
import audio

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint_path = r"checkpoints/wav2lip_gan.pth"

def load_model(checkpoint_path):
    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()
    return model


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, batch_size, pads):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
    predictions = []
    for i in range(0, len(images), batch_size):
        predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    return results


def datagen(frames, mels, static, batch_size, img_size, box, face_det_batch_size, pads):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if box[0] == -1:
        if not static:
            face_det_results = face_detect(frames, face_det_batch_size, pads)
        else:
            face_det_results = face_detect([frames[0]], face_det_batch_size, pads)
    else:
        y1, y2, x1, x2 = box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch


import logging

# Setting up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/lip-sync', methods=['POST'])
def lip_sync():
    logging.info('Received request')
    #checkpoint_path = r"D:\Use-Case\Video_to_video_translation\LIPGAN\WAV2LIP_final\Wav2Lip\checkpoints\wav2lip.pth"
    face_path = request.files['face']
    audio_path = request.files['audio']
    outfile = request.form.get('outfile', 'results/result_voice.mp4')
    static = request.form.get('static', 'false').lower() == 'true'
    fps = float(request.form.get('fps', 25))
    pads = list(map(int, request.form.get('pads', '0 10 0 0').split()))
    face_det_batch_size = int(request.form.get('face_det_batch_size', 16))
    wav2lip_batch_size = int(request.form.get('wav2lip_batch_size', 128))
    resize_factor = int(request.form.get('resize_factor', 1))
    crop = list(map(int, request.form.get('crop', '0 -1 0 -1').split()))
    box = list(map(int, request.form.get('box', '-1 -1 -1 -1').split()))
    rotate = request.form.get('rotate', 'false').lower() == 'true'
    nosmooth = request.form.get('nosmooth', 'false').lower() == 'true'

    logging.info('Saving face and audio files')
    face_filename = secure_filename(face_path.filename)
    face_path.save(face_filename)

    audio_filename = secure_filename(audio_path.filename)
    audio_path.save(audio_filename)

    file_extension = face_filename.split('.')[-1].lower()
    if file_extension in ['jpg', 'png', 'jpeg']:
        static = True
    elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
        static = False
    else:
        logging.error('Unsupported file format for face input.')
        return jsonify({'error': 'Unsupported file format for face input.'}), 400

    if not os.path.exists('temp'):
        os.makedirs('temp')

    if not audio_filename.endswith('.wav'):
        logging.info('Converting audio to WAV')
        command = f'ffmpeg -y -i {audio_filename} -strict -2 temp/temp.wav'
        subprocess.call(command, shell=True)
        audio_filename = 'temp/temp.wav'

    logging.info('Loading and processing audio')
    wav = audio.load_wav(audio_filename, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        logging.error('Mel contains nan')
        return jsonify({
            'error': 'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again'
        }), 400

    mel_step_size = 16
    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    if not static:
        logging.info('Reading video frames')
        video_stream = cv2.VideoCapture(face_filename)
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)
    else:
        logging.info('Reading single image frame')
        full_frames = [cv2.imread(face_filename)]

    logging.info('Loading model')
    model = load_model(checkpoint_path)
    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    logging.info('Generating lip sync frames')
    gen = datagen(full_frames, mel_chunks, static, wav2lip_batch_size, 96, box, face_det_batch_size, pads)
    for img_batch, mel_batch, frames, coords in gen:
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    logging.info('Combining audio and video')
    command = f'ffmpeg -y -i {audio_filename} -i temp/result.avi -strict -2 -q:v 1 {outfile}'
    subprocess.call(command, shell=True)

    logging.info('Sending output file')
    return send_file(outfile, as_attachment=True)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 9999, debug=False)

