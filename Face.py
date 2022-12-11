import face_recognition
import cv2
import numpy as np
import pickle

video_capture = cv2.VideoCapture(0)



all_face_encodings = {'santosh': ([-0.08620673,  0.0687653 , -0.00608819, -0.03519192, -0.09629069,
        0.02071193,  0.00855641, -0.0746007 ,  0.20025523, -0.19063942,
        0.23173384, -0.01975415, -0.19306482, -0.09184648, -0.02485181,
        0.12691274, -0.13205203, -0.23288512, -0.08447211, -0.04476186,
        0.0083975 , -0.0296244 , -0.0538367 ,  0.0996822 , -0.11422063,
       -0.39109814, -0.03008146, -0.08237354,  0.00964145, -0.10379514,
        0.01887341,  0.08326089, -0.2100599 , -0.04585231, -0.03458627,
        0.15673032, -0.04124431,  0.02592537,  0.22747248,  0.03232506,
       -0.23795892,  0.04985239, -0.0171667 ,  0.3682757 ,  0.11798345,
        0.05134935,  0.01320769, -0.0422096 ,  0.09669503, -0.21262598,
        0.07605252,  0.16217186,  0.11024585,  0.02930695, -0.01529235,
       -0.11060968, -0.05178311,  0.09883492, -0.19042166,  0.05386725,
        0.05107993, -0.08046649, -0.07053865, -0.08853689,  0.28919473,
        0.1427469 , -0.11267848, -0.14243214,  0.19903164, -0.20890479,
       -0.08105254,  0.07197691, -0.09797259, -0.14822087, -0.2282711 ,
        0.0713177 ,  0.44073999,  0.10898171, -0.16698462,  0.0021062 ,
       -0.05878554, -0.02249282,  0.10918788,  0.13914604, -0.01934218,
        0.0422359 , -0.05627656,  0.0509939 ,  0.21402936,  0.04171608,
       -0.08584393,  0.17579725,  0.03536662,  0.00843763,  0.05709684,
       -0.01475102, -0.03864821,  0.01387846, -0.18548468, -0.02710009,
        0.00405915, -0.01406179, -0.01352922,  0.06071987, -0.11533216,
        0.17780612,  0.00737498,  0.02268635,  0.05493347,  0.11715318,
       -0.12227686, -0.13134263,  0.15357852, -0.25482872,  0.18621401,
        0.14612481,  0.06547383,  0.13273469,  0.09938824,  0.08322583,
        0.02706247, -0.0312997 , -0.23157075, -0.05897343,  0.0765569 ,
       -0.02074298,  0.11898391,  0.00958223])}
# you can add more name to be recogised with their unique code in commas

known_face_encodings = list(all_face_encodings.values())


known_face_names =list(all_face_encodings.keys())

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
