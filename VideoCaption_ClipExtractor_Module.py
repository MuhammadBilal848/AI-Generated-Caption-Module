import av
import cv2
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
 
def video_to_caption(video_path):
    '''Takes a video file path as input and return the caption of the video along with the timestamp of the frames selected for captioning.'''
    container = av.open(video_path)

    frame_rate = container.streams.video[0].average_rate
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames

    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    timestamps = []  

    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(frame.time)

    gen_kwargs = {
        "min_length": 10,
        "max_length": 20,
        "num_beams": 3,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

    words = caption.split()
    num_frames = len(timestamps)
    segment_length = 2  

    frame_word_map = {}
    for i, timestamp in enumerate(timestamps):
        start_index = i * segment_length
        end_index = start_index + segment_length
        if start_index < len(words):
            frame_word_map[timestamp] = ' '.join(words[start_index:end_index])

    return caption,frame_word_map

def add_text_to_video(input_video_path, output_video_path, text, font=cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale=1, color=(255, 255, 255), thickness=2):
    '''Takes a video file path, text to be added to the video, and the output video file path as input and adds the text to the video and saves it to the output path.'''
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

def extract_video_segment(video_path, start_time, end_time, output_path):
    '''Takes a video file path, start time, end time, and output video file path as input and extracts the video segment from the start time to the end time and saves it to the output path.'''
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ))

    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

def text_to_segment(video_path, user_query, frame_word_map, output_path):
    '''Takes a video file path, user query, frame-word map, and output video file path as input and extracts the video segment based on the user query and frame-word map and saves it to the output path.'''
    user_query_embedding = sentence_model.encode(user_query, convert_to_tensor=True)

    frame_word_embeddings = {timestamp: sentence_model.encode(phrase, convert_to_tensor=True) for timestamp, phrase in frame_word_map.items()}

    similarities = {}
    for timestamp, embedding in frame_word_embeddings.items():
        similarity = cosine_similarity(user_query_embedding.cpu().numpy().reshape(1, -1), embedding.cpu().numpy().reshape(1, -1))[0][0]
        similarities[timestamp] = similarity

    print(f"Similarities: {similarities}")

    filtered_similarities = {timestamp: similarity for timestamp, similarity in similarities.items() if similarity > 0.6}

    if filtered_similarities:
        best_timestamp = max(filtered_similarities, key=filtered_similarities.get)
        timestamps = sorted(filtered_similarities.keys())
        next_timestamp_index = timestamps.index(best_timestamp) + 1
        next_timestamp = timestamps[next_timestamp_index] if next_timestamp_index < len(timestamps) else None

        print(f"Best timestamp: {best_timestamp}")
        if next_timestamp:
            print(f"Next timestamp: {next_timestamp}")
        else:
            print("No next timestamp with similarity greater than 60%")

        start_time = best_timestamp
        end_time = next_timestamp if next_timestamp else start_time + 2  # assuming a default duration if next timestamp doesn't exist
        print(f"Extract frames from {start_time} to {end_time}")

        extract_video_segment(video_path, start_time, end_time, output_path)

        return best_timestamp, next_timestamp
    else:
        print("No timestamps with similarity greater than 60%")
        return None, None