import cv2

# TODO split video to images
def VideotoPicture(inputpath,outputpath):
    # build a VideoCapture object，read the video file
    cap = cv2.VideoCapture(inputpath)

    fps = cap.get(cv2.CAP_PROP_FPS)  # get the flame rate (The number of  flames in 1sec)
    print("flame rate (The number of  flames in 1sec):", fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # If the video is read successfully
    sucess = cap.isOpened()

    frame_count = 0
    img_name = 0
    while sucess:
        frame_count += 1
        # read each flame
        sucess, frame = cap.read()
        # TODO save one image each 20 flames
        if (frame_count % 20 == 0):
            img_name += 1
            #The final image names are: 1.jpg，2.jpg，3.jpg.................
            cv2.imwrite(outputpath+str(img_name)+'.jpg', frame)

    # release resource
    cap.release


def focal_length(sensor_size, image_size, focal_length_in_pixel):
    sensor_width_mm, sensor_height_mm = sensor_size
    image_width_px, image_height_px = image_size
    fc_x, fc_y = focal_length_in_pixel

    # transfer focal length in pixels into mm
    focal_length_mm_x = (fc_x * sensor_width_mm) / image_width_px
    focal_length_mm_y = (fc_y * sensor_height_mm) / image_height_px

    print(f"Focal Length（Horizontal）: {focal_length_mm_x} mm")
    print(f"Focal Length（Vertical): {focal_length_mm_y} mm")

    return fc_x, fc_y


# # princiapl point
# cc = [ 937.885559086665126 , 597.250118087286523 ]
# cc_ = [image_width_px/2 , image_height_px/2]
# print('shiftx: {}, shifty: {}'.format(cc[0] - cc_[0], cc[1] - cc_[1]))
# #shiftx: -22.114440913334874, shifty: 57.25011808728652

if __name__ == '__main__':
    # inputpath = '/homes/jh523/msc_project/Lap AR/Chessboard 2.mp4'
    # outputpath = '/homes/jh523/msc_project/Lap AR/chessboard2_list/camera_calib'
    # VideotoPicture(inputpath,outputpath)

    sensor_size = [36, 24]

    image_size = [1920, 1080]

    focal_length_in_pixel =  [1068.172863525421917, 1074.913074359115399]

    fc_x, fc_y = focal_length(sensor_size, image_size, focal_length_in_pixel)
    #Focal Length（Horizontal）: 20.02824119110166 mm
    #Focal Length（Vertical): 23.88695720798034 mm
