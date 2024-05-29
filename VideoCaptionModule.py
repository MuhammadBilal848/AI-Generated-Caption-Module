import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)


def video_to_caption(video_path):
    container = av.open(video_path)

    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    gen_kwargs = {
        "min_length": 10, 
        "max_length": 20, 
        "num_beams": 8,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return caption


def add_text_to_video(input_video_path, output_video_path, text, font=cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale=1, color=(255, 255, 255), thickness=2):
    cap = cv2.VideoCapture(input_video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = text_size[1] + 10  
    position = (text_x, text_y)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()