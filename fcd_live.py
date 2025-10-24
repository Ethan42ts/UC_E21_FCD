import numpy as np
import scipy.signal.windows as sigWin
import cv2
from datetime import datetime

from getcarrier import Carrier, getcarrier
from findorthcarrierpks import findorthcarrierpks
from check_isclose import save_array
from utils.kvec import kvec
from fcd_dispfield import fcd_dispfield

START_FRAME = 1
CAMERA_CALIBRATED = False
CAMERA_NUMBER = 2

TOP_CUT = 160
BOTTTOM_CUT = 0
LEFT_CUT = 500 
RIGHT_CUT = 270

def crop_frame(frame):
    """Crop frame using global cut values."""
    h, w = frame.shape[:2]
    return frame[TOP_CUT:h-BOTTTOM_CUT, LEFT_CUT:w-RIGHT_CUT]

def load_camera_calibration(frame):
    try:
        camera_matrix = np.loadtxt('./CameraCalibrationData/Logi_camera/camera_matrix.npy')
        distortion_coeff = np.loadtxt('./CameraCalibrationData/Logi_camera/distortion_coeff.npy')
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w,h), 1, (w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeff, None, new_camera_matrix, (w,h), cv2.CV_16SC2)

    except:
        raise FileExistsError("Missing Camera Matrices and Distortion Files.")

    return camera_matrix, distortion_coeff, new_camera_matrix, roi

def analyse_Iref(Iref):
        # grayFrame = cv2.cvtColor(Iref_colour, cv2.COLOR_BGR2GRAY)
        # print(grayFrame)
        # Iref = grayFrame
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
    frame_count = 0
    # cap = cv2.VideoCapture(2)
    cap = cv2.VideoCapture(CAMERA_NUMBER)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened(): 
        print("Cannot open camera")
        exit()

    while (frame_count <= START_FRAME):
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_count += 1
        if frame_count == START_FRAME:
            if CAMERA_CALIBRATED:
                camera_matrix, distortion_coeff, new_camera_matrix, roi = load_camera_calibration(frame)
                frame = cv2.undistort(frame, camera_matrix, distortion_coeff, None, new_camera_matrix)  # Undistort the image.
                x,y,w,h = roi
                frame = frame[y:y+h, x:x+w]
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayFrame = crop_frame(grayFrame)
            cr, cu, kr = analyse_Iref(grayFrame)
    
            # resultFrame = cv2.

            min_disp = 0
            max_disp = (np.pi/np.sqrt(np.sum(kr**2)) / 6).astype(np.float64)
            disp_range = max_disp - min_disp

    while True:
        start_time = datetime.now() 
        # Capture frame-by-frame
        ret, Idef = cap.read()
        # print(Idef.shape)
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        if CAMERA_CALIBRATED:
            Idef = cv2.undistort(Idef, camera_matrix, distortion_coeff, None, new_camera_matrix)  # Undistort the image.
            x,y,w,h = roi
            Idef = Idef[y:y+h, x:x+w]

        # Our operations on the frame come here
        Idef = cv2.cvtColor(Idef, cv2.COLOR_BGR2GRAY)
        Idef = crop_frame(Idef)
        # Display the resulting frame

        u, v = fcd_dispfield(np.fft.fft2(Idef), cr, cu, False)
        # save_array(kxvec) 

        nrmU = np.sqrt(u**2 + v**2)



        nrmU /= disp_range
        # print(type(nrmU))
        nrmU = cv2.medianBlur(nrmU.astype(np.float32), ksize=5)
        # nrmU = cv2.normalize(nrmU, None, 0, 255, cv2.NORM_MINMAX)
        # print(frame_count)
        end_time = datetime.now()
        FPS = (end_time - start_time).total_seconds() # Seconds
        print(1/FPS)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('frame', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('frame', nrmU)
        cv2.imshow('frame', nrmU)
        if cv2.waitKey(1) == ord('q'):
            # print(nrmU)
            break
        elif cv2.waitKey(1) == ord('r'):
            # print("TODO: implement reset of Iref")
            cr, cu, kr = analyse_Iref(Idef)
            print("Iref Reset")
        elif cv2.waitKey(1) == ord("o"):
            max_disp = max_disp / 1.25
            disp_range = max_disp - min_disp
            print(f"Reduced range: Max = {max_disp}")
        elif cv2.waitKey(1) == ord("p"):
            max_disp *= 1.25
            disp_range = max_disp - min_disp
            print(f"Increased range Max = {max_disp}")


if __name__ == "__main__":
    main()