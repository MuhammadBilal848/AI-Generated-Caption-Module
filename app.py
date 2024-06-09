import os
import cv2
import uvicorn
from fastapi import FastAPI ,UploadFile ,File
from VideoCaption_ClipExtractor_Module import video_to_caption ,  add_text_to_video, extract_video_segment , text_to_segment

app = FastAPI()

@app.get('/')
def index():
    return {'Welcome to AI Captioning & Clip Extracting API': 'Please use the http://127.0.0.1:8000/docs endpoint to post a video file for captioning & clip extracting.}'}
 
@app.post("/post-video_captioning/")
async def vid_caption(file: UploadFile = File(...)):
    inp_save_path = f"video_captioning/input_{file.filename}"
    out_save_path = f"video_captioning/output_{file.filename}"
    
    os.makedirs(os.path.dirname(inp_save_path), exist_ok=True)
    
    with open(inp_save_path, "wb") as video_file:
        contents = await file.read()
        video_file.write(contents)
    
    captions,_= video_to_caption(inp_save_path)
    
    add_text_to_video(inp_save_path, out_save_path, captions)
    
    return {"filename": file.filename, "captions": captions}

@app.post("/post-clip_extractor/")
async def vid_clips(activity:str , file: UploadFile = File(...)):
    inp_save_path = f"clip_extractor/input_{file.filename}"
    out_save_path = f"clip_extractor/output_{file.filename}"
    
    os.makedirs(os.path.dirname(inp_save_path), exist_ok=True)
    
    with open(inp_save_path, "wb") as video_file:
        contents = await file.read()
        video_file.write(contents)
    
    captions , frame_time = video_to_caption(inp_save_path)
    
    text_to_segment(inp_save_path, activity, frame_time, out_save_path)
    return {"filename": file.filename, "captions": captions,"TimeStamp":frame_time,"Activity":activity}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
