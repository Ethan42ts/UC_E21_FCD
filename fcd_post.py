import numpy as np
import scipy.signal.windows as sigWin
import cv2
from datetime import datetime
import os 

from getcarrier import Carrier, getcarrier
from findorthcarrierpks import findorthcarrierpks
from check_isclose import save_array
from utils.kvec import kvec
from fcd_dispfield import fcd_dispfield

# VIDEO_SRC = "./RawVid/Hair_Logi.mp4" # Relative file path
# VIDEO_SRC = "./RawVid/Soldering_Logi.mp4" # Relative file path
# VIDEO_SRC = "./RawVid/py_cam_soldering_1.mkv" # Relative file path
# VIDEO_SRC = "./RawVid/pi_cam_vid/ice_1.mkv" # Relative file path
# VIDEO_SRC = "./RawVid/site_tests/test60.mkv" # Relative file path
VIDEO_SRC = "./output_reference.avi" # Relative file path
VIDEO_OUT_DIR = "./ProcessedVid" # Directory - Filename will consit of runtime date & time
START_SKIP_SECONDS = 4


CAMERA_CALIBRATED = False # Only if camera calibration not obtained, otherwise set to false
CAMERA_CALIBRATION_MATRIX_SRC =  "./CameraCalibrationData/Logi_camera/camera_matrix.npy"
CAMERA_CALIBRATION_DISTORTION_COEFF_SRC = "./CameraCalibrationData/Logi_camera/distortion_coeff.npy"

ORIGINAL_FPS = 60 # fps
START_SKIP_FRAME = ORIGINAL_FPS * START_SKIP_SECONDS # Frames from recording start to determine Iref (start of FCD program)
END_SKIP_FRAME = 100000 # Frames from recording start to the last processed frame

# Cropping
TOP_CUT = 160 # pixels
BOTTTOM_CUT = 0 # pixels
LEFT_CUT = 500  # pixels
RIGHT_CUT = 270 # pixels

def load_camera_calibration(frame):
    if not CAMERA_CALIBRATED:
        raise ConnectionError("load_camera_calibration should not be called") 
    
    try:
        camera_matrix = np.loadtxt(CAMERA_CALIBRATION_MATRIX_SRC)
        distortion_coeff = np.loadtxt(CAMERA_CALIBRATION_DISTORTION_COEFF_SRC)
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 1, (w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, new_camera_matrix, (w,h), cv2.CV_16SC2)

    except:
        raise FileExistsError("Missing Camera Matrices and Distortion Files.")
    return camera_matrix, distortion_coeff, new_camera_matrix, roi



def analyse_Iref(Iref):
    rows, cols = Iref.shape
    kr, ku = findorthcarrierpks(Iref, 4 * np.pi /np.min(Iref.shape), np.inf)
    krad = np.sqrt(np.sum((kr-ku)**2))/2
    fIref = np.fft.fft2(Iref)   
    cr = getcarrier(fIref, kr, krad)
    cu = getcarrier(fIref, ku, krad)

    kxvec = np.fft.fftshift(kvec(cols))
    kyvec = np.fft.fftshift(kvec(rows))
    wr = sigWin.hann(rows, sym = False)
    wc = sigWin.hann(cols, sym = False)
    win2d = np.outer(wr, wc)

    fftIm = np.fft.fftshift(np.abs(np.fft.fft2((Iref-np.mean(Iref)) * win2d)))
    return cr, cu, kr



def main():
    start_time = datetime.now()
    cap = cv2.VideoCapture(VIDEO_SRC)

    if not cap.isOpened():
        raise FileExistsError("Can not open file")
    frame_count = 0 
    ret, frame = cap.read()

    file_name = f"{datetime.now().strftime("%m%d%H%M%S")}"+f"_{os.path.splitext(os.path.basename(VIDEO_SRC))[0]}.avi"
    
    frameh, framew, _ = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if ((ORIGINAL_FPS == 0) or (ORIGINAL_FPS == None)):
    #     ORIGINAL_FPS = fps
    #     START_SKIP_FRAME = ORIGINAL_FPS * START_SKIP_SECONDS
    writer = cv2.VideoWriter.fourcc(*"mp4v")

    output_frame = frame

    while frame_count <= START_SKIP_FRAME and ret:
        """ Skips unwanted frames """
        ret, frame = cap.read()
        frame_count += 1
        if frame_count == START_SKIP_FRAME:
            if CAMERA_CALIBRATED:
                camera_matrix, distortion_coeff, new_camera_matrix, roi = load_camera_calibration(frame)
                frame = cv2.undistort(frame, camera_matrix, distortion_coeff, None, new_camera_matrix)  # Undistort the image.
                x,y,w,h = roi
                frame = frame[y:y+h, x:x+w]
                frameh, framew, _ = frame.shape

                # print("Run 1")
            output = cv2.VideoWriter(
                    # f"{VIDEO_OUT_DIR}/{file_name}", writer, fps, (framew - LEFT_CUT - RIGHT_CUT, frameh-TOP_CUT-BOTTTOM_CUT))
                    f"{VIDEO_OUT_DIR}/{file_name}", writer, fps, (framew, frameh))
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cr, cu, kr = analyse_Iref(grayFrame)
    
            # resultFrame = cv2.

            min_disp = 0
            max_disp = (np.pi/np.sqrt(np.sum(kr**2)) / 600).astype(np.float64)
            disp_range = max_disp - min_disp


    ret, Idef = cap.read()
    while frame_count > START_SKIP_FRAME and frame_count <= END_SKIP_FRAME and ret:
        if CAMERA_CALIBRATED:
            Idef = cv2.undistort(Idef, camera_matrix, distortion_coeff, None, new_camera_matrix)  # Undistort the image.
            x,y,w,h = roi
            Idef = Idef[y:y+h, x:x+w]
            # print("Run 2")
        
       
        Idef = cv2.cvtColor(Idef, cv2.COLOR_BGR2GRAY)

        u, v = fcd_dispfield(np.fft.fft2(Idef), cr, cu, False)
        nrmU = np.sqrt(u**2 + v**2) # average of x and y diflections
        urmU_avg = np.average(nrmU)
        # nrmU /= disp_range
        # nrmU = cv2.medianBlur(nrmU.astype(np.float64), ksize=5)

        output_frame = nrmU
        # frame_avg = np.mean(output_frame)
        # output_frame = 75 + (output_frame * 127)   # now -1 -> 1 maps to [1,255]
        # output_frame = 0 + (output_frame * 127)   # now -1 -> 1 maps to [1,255]
        # output_frame = np.clip(output_frame, 0, 255).astype(np.uint8)

        eps = 1e-12
        
        output_frame = (np.log10(output_frame + eps) - np.log10(output_frame.min() + eps)) / \
             (np.log10(output_frame.max() + eps) - np.log10(output_frame.min() + eps))
        # output_frame =  (output_frame - output_frame.min()) / (output_frame.max() - output_frame.min())
        output_frame = (255 * output_frame).clip(0, 255).astype(np.uint8)
        output_frame = cv2.cvtColor(output_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # ensure 3 channels if writer expects color
        output.write(output_frame)
        ret, Idef = cap.read()
        frame_count += 1

    
    cap.release()
    output.release() 
    cv2.destroyAllWindows()

    end_time = datetime.now()
    processing_time = int((end_time - start_time).total_seconds()) # Seconds
    print(f"Processing time: {(processing_time//60):02d}:{(processing_time%60):02d} (m:s)")
    print(f"fps: {np.round(processing_time/frame_count, 5)}s")


if __name__ == "__main__":
    main()

