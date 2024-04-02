import cv2
import os

def take_picture(output_folder, filename, camera):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Capture a frame
    ret, frame = camera.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Unable to capture frame.")
        return

    # Save the captured frame as an image with .jpg extension
    filepath = os.path.join(output_folder, filename + ".jpg")
    cv2.imwrite(filepath, frame)

    print(f"Picture saved as {filepath}")

if __name__ == "__main__":
    output_folder = "pictures"

    # Initialize the camera
    camera = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not camera.isOpened():
        print("Error: Unable to access the camera.")
    else:
        # Ask the user to enter the name of the image
        filename = input("Enter the name of the image: ")

        while True:
            # Capture a frame
            ret, frame = camera.read()

            # Display the frame
            cv2.imshow("Live Feed", frame)

            # Check for 'p' key press
            key = cv2.waitKey(1)
            if key == ord('p'):
                take_picture(output_folder, filename, camera)
                break
            elif key == 27:  # Press Esc key to exit
                break

        # Release the camera
        camera.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()
