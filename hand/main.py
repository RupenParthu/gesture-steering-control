from tracking import HandTracker
import cv2 as cv

# Create an instance of the tracker
tracker = HandTracker()

# Open the webcam   
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image for mirror effect
    frame = cv.flip(frame, 1)

    # Detect hands and draw landmarks
    frame = tracker.find_hands(frame)

    # Loop through both hands (hand 0 and hand 1)
    for hand_id in range(2):
        landmarks = tracker.get_landmark_positions(frame, hand_no=hand_id)

        if landmarks:
            # Draw circle on the index fingertip (ID 8)
            index_tip = landmarks[8]
            cv.circle(frame, (index_tip[1], index_tip[2]), 10, (0, 255, 0), cv.FILLED)

    # Show the frame
    cv.imshow("Hand Tracking", frame)

    # Break the loop on 'q' press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release and cleanup
cap.release()
cv.destroyAllWindows()
