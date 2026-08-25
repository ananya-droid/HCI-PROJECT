"""camera_test.py — verifies the webcam opens and returns frames."""
import cv2


def test_camera(index=0, seconds_hint=True):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ Could not open camera index {index}.")
        return False

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("❌ Camera opened but failed to return a frame.")
        return False

    print(f"✅ Camera OK. Frame shape: {frame.shape}")
    return True


if __name__ == "__main__":
    test_camera()
