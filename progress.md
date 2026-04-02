# Basketball Analyzer — Daily Progress Log


## Day 0 — March 31, 2026
**Idea Decided**

The idea:
- User uploads a video of their basketball shot
- Tool analyzes shooting stance using pose estimation
- Tracks ball trajectory and tells if shot is flat, short, long, off left or off right
- Compares release stance with top NBA players like Curry or Durant
- Generates personalized AI coaching feedback using an LLM

---

## Day 1 — April 1, 2026
**Setup + Pose Estimation Working**

- Installed Cursor and set it up as the main code editor
- Installed OpenCV, MediaPipe, NumPy via pip
- Downloaded a test basketball video using yt-dlp
- Used Cursor AI to write and understand the pose estimation code
- Successfully ran MediaPipe on the test video
- Output video generated with full body skeleton drawn on the person
- Set up GitHub repository: basketball-analyzer
- Pushed first commit to GitHub

First milestone achieved — skeleton tracking working on a real video

---

## Day 2 — April 2, 2026
**Ball Detection + Combined Output Working**

- Tried YOLOv8n for ball detection — too weak, switched to YOLOv8x
- Implemented YOLO + CSRT tracker — YOLO finds ball, tracker follows it
- Discovered ball loses tracking mid-flight due to motion blur — release point frame is what actually matters for analysis
- Combined pose estimation and ball detection into one file — analyzer.py
- Output video now shows skeleton on person and green circle on ball simultaneously

Second milestone achieved — pose + ball detection working together 
