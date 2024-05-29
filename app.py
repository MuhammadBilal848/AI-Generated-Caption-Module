import os
import uvicorn
from fastapi import FastAPI ,UploadFile ,File
from VideoCaptionModule import video_to_caption , add_text_to_video
import cv2

app = FastAPI()

@app.get('/')
def index():
    return {'Welcome to AI Captioning API': 'Please use the http://127.0.0.1:8000/docs endpoint to post a video file for captioning.}'}

@app.post("/post-video_inp/")
async def video_notes(file: UploadFile = File(...)):
    inp_save_path = f"io_videos/input_{file.filename}"
    out_save_path = f"io_videos/output_{file.filename}"
    
    os.makedirs(os.path.dirname(inp_save_path), exist_ok=True)
    
    with open(inp_save_path, "wb") as video_file:
        contents = await file.read()
        video_file.write(contents)
    
    captions = video_to_caption(inp_save_path)
    
    add_text_to_video(inp_save_path, out_save_path, captions)
    
    return {"filename": file.filename, "captions": captions}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    