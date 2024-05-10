import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = r"D:\ComputerVision\multiPed.mp4"
cap = cv2.VideoCapture(video_path)

# Get frame count
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get frames per second (fps)
fps = cap.get(cv2.CAP_PROP_FPS)

# Get frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_annotate = cv2.VideoWriter_fourcc(*'mp4v')
out_annotate = cv2.VideoWriter(r"D:\ComputerVision\multiPed_annotate.mp4", 
                      video_annotate, 
                      fps, 
                      (frame_width, frame_height))

video_traj = cv2.VideoWriter_fourcc(*'mp4v')
out_traj = cv2.VideoWriter(r"D:\ComputerVision\multiPed_circle.mp4", 
                      video_traj, 
                      fps, 
                      (frame_width, frame_height))

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=0)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        for i, det in enumerate(results[0].boxes.xyxy):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, det[:4])
            center = (int((x1+x2)/2), y2)
            radius = 20
            color = (0,0,255)
            thickness = 2
            frame = cv2.circle(frame, center, radius, color, thickness)
        out_annotate.write(annotated_frame)
        out_traj.write(frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

out_annotate.release()
out_traj.release()
cap.release()







