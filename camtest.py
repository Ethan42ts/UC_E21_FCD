import cv2

# Open the default camera
cam = cv2.VideoCapture(2)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

TOP_CUT = 160
BOTTTOM_CUT = 0
LEFT_CUT = 500
RIGHT_CUT = 270

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width - LEFT_CUT - RIGHT_CUT, 
                                                   frame_height - TOP_CUT - BOTTTOM_CUT))


def crop_frame(frame):
    """Crop frame using global cut values."""
    h, w = frame.shape[:2]
    return frame[TOP_CUT:h-BOTTTOM_CUT, LEFT_CUT:w-RIGHT_CUT]

while True:
    ret, frame = cam.read()
    frame = crop_frame(frame)
    # Write the frame to the output file
    out.write(frame)
 
    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()