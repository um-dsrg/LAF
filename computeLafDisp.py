import os
import numpy as np
from skimage import io

import LAF.evaluate

# interpolation: left-right consistency check
# every pixel disparity has 3 status
# 0: match
# 1: mismatch
# 2: occlusion
def compute_confidence_disparity(LAF_model_dir,img_path):
    # step 1:  computer confidence
    print(" Compute LAF-Net confidence...")
    confidence_threshold = 0.9
    consistanceFlag = 0.95959595
    unconsistanceFlag = 0.25252525
    left_flag = True
    left_conf_image = LAF.evaluate.compute_LAF_confidence(LAF_model_dir,img_path,left_flag)
    left_flag = False
    right_conf_image = LAF.evaluate.compute_LAF_confidence(LAF_model_dir,img_path,left_flag)

    # step 2: save confidence (option)
    #conf4img = (conf*(256.*256.-1)).astype(np.uint16)
    left_conf_image4Save = (left_conf_image*(256.*256.-1)).astype(np.uint16)
    right_conf_image4Save = (right_conf_image*(256.*256.-1)).astype(np.uint16)
    left_laf_file = os.path.join(img_path,'left_laf.png')
    right_laf_file = os.path.join(img_path,'right_laf.png')
    io.imsave(left_laf_file,left_conf_image4Save)
    io.imsave(right_laf_file,right_conf_image4Save)

    # step 3: select disparity by confidence
    # step 3.1: load disparity
    left_disp_file = os.path.join(img_path,'disp0MCCNN.npy')
    right_disp_file = os.path.join(img_path,'disp1MCCNN.npy')
    left_disparity_map = np.load(left_disp_file)
    right_disparity_map = np.load(right_disp_file)

    # step 3.2 check disparity
    # step 3.2.1 align tight disparity
    height, width = left_disparity_map.shape
    print(" Align the right disparity map by itself if left disparity is unconfidente")
    right_disparity_map_aligned = np.full(left_disparity_map.shape, np.nan)
    right_conf_image_aligned =  np.full(left_disparity_map.shape, 0.0)
    for h in range(height):
        for w in range(width):
            if w + int(right_disparity_map[h,w]) >= 0 and w + int(right_disparity_map[h,w]) < width:
                right_disparity_map_aligned[h,w+ int(right_disparity_map[h,w])] = right_disparity_map[h,w]
                right_conf_image_aligned[h,w+ int(right_disparity_map[h,w])] = right_conf_image[h,w]

    # step 3.2.2 doing left-right consistency check
    # counters for evaluating the result and debuge
    mismatch = 0
    left_mismatch = 0
    right_mismatch = 0
    occlusion = 0
    left_occlusion = 0
    right_occlusion = 0
    left_ser_occlusion = 0
    right_ser_occlusion = 0
    outside = 0
    match = 0

    print(" doing left-right consistency check...")
    consistency_map = np.zeros([height, width], dtype=np.int32)
    for h in range(height):
        for w in range(width):
            left_disparity = int(left_disparity_map[h, w])
            # no corresponding pixel, takes as occlusion
            if ((w - left_disparity)<0)or((w - left_disparity)>=width):
                consistency_map[h, w] = 2
                outside += 1
                continue

            right_disparity = right_disparity_map[h, w-left_disparity]
            if abs(left_disparity - right_disparity) <= 1:
                # match
                match += 1
                continue

            # check if mismatch
            for d in range(width):
                if abs(d - right_disparity_map[h, w-d]) <= 1:
                    # mismatch
                    mismatch += 1
                    consistency_map[h, w] = 1
                    break

            # otherwise take as occlusion
            if consistency_map[h, w] == 0:
                consistency_map[h, w] = 2
                occlusion += 1

    # step 3.3 Revising noise points
    print(" Revising noise points")
    std_threshold = 2
    revised_points = 0
    for h in range(1,height-1):
        for w in range(1,width-1):
            if consistency_map[h, w] > 0:
                # 8 neighbor
                neighborSet = np.ndarray(8, dtype=np.float32)
                neighborSet[0] = left_disparity_map[h-1, w-1]
                neighborSet[1] = left_disparity_map[h-1, w]
                neighborSet[2] = left_disparity_map[h-1, w+1]
                neighborSet[3] = left_disparity_map[h, w-1]
                neighborSet[4] = left_disparity_map[h, w+1]
                neighborSet[5] = left_disparity_map[h+1, w-1]
                neighborSet[6] = left_disparity_map[h+1, w]
                neighborSet[7] = left_disparity_map[h+1, w+1]
                std_neightbor = np.nanstd(neighborSet)

                if std_neightbor < std_threshold:
                    mean_neightbor = np.nanmean(neighborSet)
                    left_disparity = left_disparity_map[h, w]
                    diff_point = np.abs(left_disparity - mean_neightbor)
                    if diff_point > 2*std_neightbor:
                        left_disparity_map[h, w] = np.percentile(neighborSet,50)
                        consistency_map[h, w] = 0
                        revised_points += 1
    print(" Revised points number: {}".format(revised_points))

    # step 3.4 select disparity
    print(" doing interpolation...")
    int_left_disparity_map = np.ndarray([height, width], dtype=np.float32)
    int_left_confidence_map = np.ndarray([height, width], dtype=np.float32)
    for h in range(height):
        for w in range(width):

            left_disparity = left_disparity_map[h, w]
            left_confidence = left_conf_image[h, w]

            if consistency_map[h, w] == 0:
                right_disparity = right_disparity_map[h, w-int(left_disparity)]
                right_confidence = right_conf_image[h, w-int(left_disparity)]
                int_left_confidence_map[h, w] = consistanceFlag
                if left_confidence > confidence_threshold:
                    if right_confidence > confidence_threshold:
                        int_left_disparity_map[h, w] = (left_disparity + right_disparity)/2
                    else:
                        int_left_disparity_map[h, w] = left_disparity
                else:
                    if right_confidence > confidence_threshold:
                        int_left_disparity_map[h, w] = right_disparity
                    else:
                        int_left_disparity_map[h, w] = (left_disparity + right_disparity)/2
            else:
                right_disparity = right_disparity_map_aligned[h, w]
                right_confidence = right_conf_image_aligned[h, w]
                if left_confidence > confidence_threshold:
                    int_left_confidence_map[h, w] = left_confidence if left_confidence > right_confidence else right_confidence
                    if right_confidence > confidence_threshold:
                        int_left_disparity_map[h, w] = (left_disparity + right_disparity)/2
                    else:
                        int_left_disparity_map[h, w] = left_disparity
                else:
                    if right_confidence > confidence_threshold:
                        int_left_disparity_map[h, w] = right_disparity
                    else:
                        neighbours = []
                        count = 0

                        # right
                        for w_ in range(w+1, width):
                            if consistency_map[h, w_] == 0:
                                count += 1
                                if 0<=(w_ - left_disparity)<width:
                                    right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                                    right_confidence = right_conf_image[h, w_-int(left_disparity)]
                                    if left_confidence > right_confidence:
                                        neighbours.append(left_disparity)
                                        left_mismatch += 1
                                    else:
                                        neighbours.append(right_disparity)
                                        right_mismatch += 1
                                else:
                                    neighbours.append(left_disparity)
                                    left_mismatch += 1
                                break

                        # left
                        for w_ in range(w-1, -1, -1):
                            if consistency_map[h, w_] == 0:
                                count += 1
                                if 0<=(w_ - left_disparity)<width:
                                    right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                                    right_confidence = right_conf_image[h, w_-int(left_disparity)]
                                    if left_confidence > right_confidence:
                                        neighbours.append(left_disparity)
                                        left_mismatch += 1
                                    else:
                                        neighbours.append(right_disparity)
                                        right_mismatch += 1
                                else:
                                    neighbours.append(left_disparity)
                                    left_mismatch += 1
                                break

                        # bottom
                        for h_ in range(h+1, height):
                            if consistency_map[h_, w] == 0:
                                count += 1
                                if 0<=(w - left_disparity)<width:
                                    right_disparity = right_disparity_map[h_, w-int(left_disparity)]
                                    right_confidence = right_conf_image[h_, w-int(left_disparity)]
                                    if left_confidence > right_confidence:
                                        neighbours.append(left_disparity)
                                        left_mismatch += 1
                                    else:
                                        neighbours.append(right_disparity)
                                        right_mismatch += 1
                                else:
                                    neighbours.append(left_disparity)
                                    left_mismatch += 1
                                break

                        # up
                        for h_ in range(h-1, -1, -1):
                            if consistency_map[h_, w] == 0:
                                count += 1
                                if 0<=(w - left_disparity)<width:
                                    right_disparity = right_disparity_map[h_, w-int(left_disparity)]
                                    right_confidence = right_conf_image[h_, w-int(left_disparity)]
                                    if left_confidence > right_confidence:
                                        neighbours.append(left_disparity)
                                        left_mismatch += 1
                                    else:
                                        neighbours.append(right_disparity)
                                        right_mismatch += 1
                                else:
                                    neighbours.append(left_disparity)
                                    left_mismatch += 1
                                break

                        neighbours = np.array(neighbours, dtype=np.float32)

                        int_left_confidence_map[h, w] = unconsistanceFlag
                        # no nearest match, use the raw value
                        if count > 0:
                            int_left_disparity_map[h, w] = np.median(neighbours)
                        else:
                            int_left_disparity_map[h, w] = np.nan

    total_point = height*width
    print('\n total points:{} = height:{} * weigth:{}'.format(total_point,height,width))
    print(' mismatch:{} + occlusion:{} + outside:{} + match:{} = total points:{}'.format(mismatch,occlusion,outside,match,total_point))
    print(' mismatch:{}% + occlusion:{}% + outside:{}% + match:{}%'.format(int(mismatch/total_point*100),int(occlusion/total_point*100),int(outside/total_point*100),int(match/total_point*100)))
    print(' left_ser_mismatch:{} + right_ser_mismatch:{}'.format(left_mismatch,right_mismatch))
    print(' outside:{} + occulusion:{} = left_ser_occulusion:{} + right_ser_occulusion:{}'.format(outside,occlusion,left_ser_occlusion,right_ser_occlusion))
    print('                            + left_occulusion:{} + right_occulusion:{}\n'.format(left_occlusion,right_occlusion))
    left_disparity_map = int_left_disparity_map

    return left_disparity_map, int_left_disparity_map

def compute_confidence_disparity_bak_2(LAF_model_dir,img_path):

    print(" Compute LAF-Net confidence...")
    confidence_threshold = 58982.
    left_flag = True
    left_conf_image = liblaf.evaluate.compute_LAF_confidence(LAF_model_dir,img_path,left_flag)
    left_flag = False
    right_conf_image = liblaf.evaluate.compute_LAF_confidence(LAF_model_dir,img_path,left_flag)
    left_laf_file = os.path.join(img_path,'left_laf.png')
    right_laf_file = os.path.join(img_path,'right_laf.png')
    io.imsave(left_laf_file,left_conf_image)
    io.imsave(right_laf_file,right_conf_image)

    left_disp_file = os.path.join(img_path,'disp0MCCNN.npy')
    right_disp_file = os.path.join(img_path,'disp1MCCNN.npy')
    left_disparity_map = np.load(left_disp_file)
    right_disparity_map = np.load(right_disp_file)

    height, width = left_disparity_map.shape

    print(" Align the right disparity map by itself if left disparity is unconfidente")
    right_disparity_map_aligned = np.full(left_disparity_map.shape, np.nan)
    right_conf_image_aligned =  np.full(left_disparity_map.shape, 0.0)
    for h in range(height):
        for w in range(width):
            if w + int(right_disparity_map[h,w]) >= 0 and w + int(right_disparity_map[h,w]) < width:
                right_disparity_map_aligned[h,w+ int(right_disparity_map[h,w])] = right_disparity_map[h,w]
                right_conf_image_aligned[h,w+ int(right_disparity_map[h,w])] = right_conf_image[h,w]

    # counters for evaluating the result and debuge
    mismatch = 0
    left_mismatch = 0
    right_mismatch = 0
    occlusion = 0
    left_occlusion = 0
    right_occlusion = 0
    left_ser_occlusion = 0
    right_ser_occlusion = 0
    outside = 0
    match = 0


    print(" doing left-right consistency check...")
    consistency_map = np.zeros([height, width], dtype=np.int32)
    for h in range(height):
        for w in range(width):
            left_disparity = int(left_disparity_map[h, w])
            # no corresponding pixel, takes as occlusion
            if ((w - left_disparity)<0)or((w - left_disparity)>=width):
                consistency_map[h, w] = 2
                outside += 1
                continue

            right_disparity = right_disparity_map[h, w-left_disparity]
            if abs(left_disparity - right_disparity) <= 1:
                # match
                match += 1
                continue

            # check if mismatch
            for d in range(width):
                if abs(d - right_disparity_map[h, w-d]) <= 1:
                    # mismatch
                    mismatch += 1
                    consistency_map[h, w] = 1
                    break

            # otherwise take as occlusion
            if consistency_map[h, w] == 0:
                consistency_map[h, w] = 2
                occlusion += 1

    print(" Revising noise points")
    std_threshold = 2
    revised_points = 0
    for h in range(1,height-1):
        for w in range(1,width-1):
            if consistency_map[h, w] > 0:
                # 8 neighbor
                neighborSet = np.ndarray(8, dtype=np.float32)
                neighborSet[0] = left_disparity_map[h-1, w-1]
                neighborSet[1] = left_disparity_map[h-1, w]
                neighborSet[2] = left_disparity_map[h-1, w+1]
                neighborSet[3] = left_disparity_map[h, w-1]
                neighborSet[4] = left_disparity_map[h, w+1]
                neighborSet[5] = left_disparity_map[h+1, w-1]
                neighborSet[6] = left_disparity_map[h+1, w]
                neighborSet[7] = left_disparity_map[h+1, w+1]
                std_neightbor = np.nanstd(neighborSet)

                if std_neightbor < std_threshold:
                    mean_neightbor = np.nanmean(neighborSet)
                    left_disparity = left_disparity_map[h, w]
                    diff_point = np.abs(left_disparity - mean_neightbor)
                    if diff_point > 2*std_neightbor:
                        left_disparity_map[h, w] = np.percentile(neighborSet,50)
                        consistency_map[h, w] = 0
                        revised_points += 1
    print(" Revised points number: {}".format(revised_points))

    print(" doing interpolation...")
    int_left_disparity_map = np.ndarray([height, width], dtype=np.float32)
    for h in range(height):
        for w in range(width):

            left_disparity = left_disparity_map[h, w]
            left_confidence = left_conf_image[h, w]
            right_disparity = right_disparity_map_aligned[h, w]
            right_confidence = right_conf_image_aligned[h, w]

            if consistency_map[h, w] == 0:
                right_disparity = right_disparity_map[h, w-int(left_disparity)]
                right_confidence = right_conf_image[h, w-int(left_disparity)]
                if left_confidence > confidence_threshold:
                    if right_confidence > confidence_threshold:
                        int_left_disparity_map[h, w] = (left_disparity + right_disparity)/2
                    else:
                        int_left_disparity_map[h, w] = left_disparity
                else:
                    if right_confidence > confidence_threshold:
                        int_left_disparity_map[h, w] = right_disparity
                    else:
                        int_left_disparity_map[h, w] = (left_disparity + right_disparity)/2

            else:
                right_disparity = right_disparity_map_aligned[h, w]
                right_confidence = right_conf_image_aligned[h, w]
                if left_confidence > confidence_threshold:
                    if right_confidence > confidence_threshold:
                        int_left_disparity_map[h, w] = (left_disparity + right_disparity)/2
                    else:
                        int_left_disparity_map[h, w] = left_disparity
                else:
                    if right_confidence > confidence_threshold:
                        int_left_disparity_map[h, w] = right_disparity
                    else:
                        neighbours = []
                        count = 0

                        # right
                        for w_ in range(w+1, width):
                            if consistency_map[h, w_] == 0:
                                count += 1
                                if 0<=(w_ - left_disparity)<width:
                                    right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                                    right_confidence = right_conf_image[h, w_-int(left_disparity)]
                                    if left_confidence > right_confidence:
                                        neighbours.append(left_disparity)
                                        left_mismatch += 1
                                    else:
                                        neighbours.append(right_disparity)
                                        right_mismatch += 1
                                else:
                                    neighbours.append(left_disparity)
                                    left_mismatch += 1
                                break

                        # left
                        for w_ in range(w-1, -1, -1):
                            if consistency_map[h, w_] == 0:
                                count += 1
                                if 0<=(w_ - left_disparity)<width:
                                    right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                                    right_confidence = right_conf_image[h, w_-int(left_disparity)]
                                    if left_confidence > right_confidence:
                                        neighbours.append(left_disparity)
                                        left_mismatch += 1
                                    else:
                                        neighbours.append(right_disparity)
                                        right_mismatch += 1
                                else:
                                    neighbours.append(left_disparity)
                                    left_mismatch += 1
                                break

                        # bottom
                        for h_ in range(h+1, height):
                            if consistency_map[h_, w] == 0:
                                count += 1
                                if 0<=(w - left_disparity)<width:
                                    right_disparity = right_disparity_map[h_, w-int(left_disparity)]
                                    right_confidence = right_conf_image[h_, w-int(left_disparity)]
                                    if left_confidence > right_confidence:
                                        neighbours.append(left_disparity)
                                        left_mismatch += 1
                                    else:
                                        neighbours.append(right_disparity)
                                        right_mismatch += 1
                                else:
                                    neighbours.append(left_disparity)
                                    left_mismatch += 1
                                break

                        # up
                        for h_ in range(h-1, -1, -1):
                            if consistency_map[h_, w] == 0:
                                count += 1
                                if 0<=(w - left_disparity)<width:
                                    right_disparity = right_disparity_map[h_, w-int(left_disparity)]
                                    right_confidence = right_conf_image[h_, w-int(left_disparity)]
                                    if left_confidence > right_confidence:
                                        neighbours.append(left_disparity)
                                        left_mismatch += 1
                                    else:
                                        neighbours.append(right_disparity)
                                        right_mismatch += 1
                                else:
                                    neighbours.append(left_disparity)
                                    left_mismatch += 1
                                break

                        neighbours = np.array(neighbours, dtype=np.float32)

                        # no nearest match, use the raw value
                        if count > 0:
                            int_left_disparity_map[h, w] = np.median(neighbours)
                        else:
                            int_left_disparity_map[h, w] = np.nan


            '''
            else:
                # occlusion
                # just use the nearest match neighbour value on the right
                # NOTE: in the origin paper, they use left rather than left

                # right
                count = 0
                for w_ in range(w+1, width):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        left_disparity = left_disparity_map[h, w_]
                        left_confidence = left_conf_image[h, w_]
                        if 0<=(w_ - left_disparity)<width:
                            right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                            right_confidence = right_conf_image[h, w_-int(left_disparity)]
                            if left_confidence > right_confidence:
                                int_left_disparity_map[h, w] = left_disparity
                                left_ser_occlusion += 1
                            else:
                                int_left_disparity_map[h, w] = right_disparity
                                right_ser_occlusion += 1
                        else:
                            int_left_disparity_map[h, w] = left_disparity
                            left_ser_occlusion += 1
                        break

                # no match neighbour found, use the raw value
                if count == 0:
                    if 0<(w - left_disparity)<width:
                        left_confidence = left_conf_image[h, w]
                        right_confidence = right_conf_image[h, w-int(left_disparity)]
                        if left_confidence > right_confidence:
                            int_left_disparity_map[h, w] = left_disparity
                            left_occlusion += 1
                        else:
                            int_left_disparity_map[h, w] = right_disparity
                            right_occlusion += 1
                    else:
                        int_left_disparity_map[h, w] = left_disparity
                        left_occlusion += 1
            '''

    total_point = height*width
    print('\n total points:{} = height:{} * weigth:{}'.format(total_point,height,width))
    print(' mismatch:{} + occlusion:{} + outside:{} + match:{} = total points:{}'.format(mismatch,occlusion,outside,match,total_point))
    print(' mismatch:{}% + occlusion:{}% + outside:{}% + match:{}%'.format(int(mismatch/total_point*100),int(occlusion/total_point*100),int(outside/total_point*100),int(match/total_point*100)))
    print(' left_ser_mismatch:{} + right_ser_mismatch:{}'.format(left_mismatch,right_mismatch))
    print(' outside:{} + occulusion:{} = left_ser_occulusion:{} + right_ser_occulusion:{}'.format(outside,occlusion,left_ser_occlusion,right_ser_occlusion))
    print('                            + left_occulusion:{} + right_occulusion:{}\n'.format(left_occlusion,right_occlusion))
    left_disparity_map = int_left_disparity_map

    return left_disparity_map

def compute_confidence_disparity_bak_1(LAF_model_dir,img_path):

    print(" Compute LAF-Net confidence...")
    confidence_threshold = 255.
    left_flag = True
    left_conf_image = liblaf.evaluate.compute_LAF_confidence(LAF_model_dir,img_path,left_flag)
    left_flag = False
    right_conf_image = liblaf.evaluate.compute_LAF_confidence(LAF_model_dir,img_path,left_flag)
    left_laf_file = os.path.join(img_path,'left_laf.png')
    right_laf_file = os.path.join(img_path,'right_laf.png')
    io.imsave(left_laf_file,left_conf_image)
    io.imsave(right_laf_file,right_conf_image)

    left_disp_file = os.path.join(img_path,'disp0MCCNN.npy')
    right_disp_file = os.path.join(img_path,'disp1MCCNN.npy')
    left_disparity_map = np.load(left_disp_file)
    right_disparity_map = np.load(right_disp_file)

    height, width = left_disparity_map.shape

    print(" Align the right disparity map by itself if left disparity is unconfidente")
    right_disparity_map_aligned = np.full(left_disparity_map.shape, np.nan)
    right_conf_image_aligned =  np.full(left_disparity_map.shape, 0.0)
    for h in range(height):
        for w in range(width):
            if w + int(right_disparity_map[h,w]) >= 0 and w + int(right_disparity_map[h,w]) < width:
                right_disparity_map_aligned[h,w+ int(right_disparity_map[h,w])] = right_disparity_map[h,w]
                right_conf_image_aligned[h,w+ int(right_disparity_map[h,w])] = right_conf_image[h,w]

    # counters for evaluating the result and debuge
    mismatch = 0
    left_mismatch = 0
    right_mismatch = 0
    occlusion = 0
    left_occlusion = 0
    right_occlusion = 0
    left_ser_occlusion = 0
    right_ser_occlusion = 0
    outside = 0
    match = 0


    print(" doing left-right consistency check...")
    consistency_map = np.zeros([height, width], dtype=np.int32)
    for h in range(height):
        for w in range(width):
            left_disparity = int(left_disparity_map[h, w])
            # no corresponding pixel, takes as occlusion
            if ((w - left_disparity)<0)or((w - left_disparity)>=width):
                consistency_map[h, w] = 2
                outside += 1
                continue

            right_disparity = right_disparity_map[h, w-left_disparity]
            if abs(left_disparity - right_disparity) <= 1:
                # match
                match += 1
                continue

            # check if mismatch
            for d in range(width):
                if abs(d - right_disparity_map[h, w-d]) <= 1:
                    # mismatch
                    mismatch += 1
                    consistency_map[h, w] = 1
                    break

            # otherwise take as occlusion
            if consistency_map[h, w] == 0:
                consistency_map[h, w] = 2
                occlusion += 1

    print(" Revising noise points")
    std_threshold = 2
    revised_points = 0
    for h in range(1,height-1):
        for w in range(1,width-1):
            if consistency_map[h, w] > 0:
                # 8 neighbor
                neighborSet = np.ndarray(8, dtype=np.float32)
                neighborSet[0] = left_disparity_map[h-1, w-1]
                neighborSet[1] = left_disparity_map[h-1, w]
                neighborSet[2] = left_disparity_map[h-1, w+1]
                neighborSet[3] = left_disparity_map[h, w-1]
                neighborSet[4] = left_disparity_map[h, w+1]
                neighborSet[5] = left_disparity_map[h+1, w-1]
                neighborSet[6] = left_disparity_map[h+1, w]
                neighborSet[7] = left_disparity_map[h+1, w+1]
                std_neightbor = np.nanstd(neighborSet)

                if std_neightbor < std_threshold:
                    mean_neightbor = np.nanmean(neighborSet)
                    left_disparity = left_disparity_map[h, w]
                    diff_point = np.abs(left_disparity - mean_neightbor)
                    if diff_point > 2*std_neightbor:
                        left_disparity_map[h, w] = np.percentile(neighborSet,50)
                        consistency_map[h, w] = 0
                        revised_points += 1
    print(" Revised points number: {}".format(revised_points))

    print(" doing interpolation...")
    int_left_disparity_map = np.ndarray([height, width], dtype=np.float32)
    for h in range(height):
        for w in range(width):

            left_disparity = left_disparity_map[h, w]
            left_confidence = left_conf_image[h, w]
            right_disparity = right_disparity_map_aligned[h, w]
            right_confidence = right_conf_image_aligned[h, w]

            if consistency_map[h, w] == 0:
                right_disparity = right_disparity_map[h, w-int(left_disparity)]
                right_confidence = right_conf_image[h, w-int(left_disparity)]
                if left_confidence > confidence_threshold:
                    if right_confidence > confidence_threshold:
                        int_left_disparity_map[h, w] = (left_disparity + right_disparity)/2
                    else:
                        int_left_disparity_map[h, w] = left_disparity
                else:
                    if right_confidence > confidence_threshold:
                        int_left_disparity_map[h, w] = right_disparity
                    else:
                        int_left_disparity_map[h, w] = (left_disparity + right_disparity)/2

            else:
            #elif consistency_map[h, w] == 1:
                # mismatch, taken median value from nearest match neighbours in 4 directions
                # NOTE: in origin paper, they use 16 directions
                count = 0
                neighbours = []

                # right
                for w_ in range(w+1, width):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        if 0<=(w_ - left_disparity)<width:
                            right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                            right_confidence = right_conf_image[h, w_-int(left_disparity)]
                            if left_confidence > right_confidence:
                                neighbours.append(left_disparity)
                                left_mismatch += 1
                            else:
                                neighbours.append(right_disparity)
                                right_mismatch += 1
                        else:
                            neighbours.append(left_disparity)
                            left_mismatch += 1
                        break

                # left
                for w_ in range(w-1, -1, -1):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        if 0<=(w_ - left_disparity)<width:
                            right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                            right_confidence = right_conf_image[h, w_-int(left_disparity)]
                            if left_confidence > right_confidence:
                                neighbours.append(left_disparity)
                                left_mismatch += 1
                            else:
                                neighbours.append(right_disparity)
                                right_mismatch += 1
                        else:
                            neighbours.append(left_disparity)
                            left_mismatch += 1
                        break

                # bottom
                for h_ in range(h+1, height):
                    if consistency_map[h_, w] == 0:
                        count += 1
                        if 0<=(w - left_disparity)<width:
                            right_disparity = right_disparity_map[h_, w-int(left_disparity)]
                            right_confidence = right_conf_image[h_, w-int(left_disparity)]
                            if left_confidence > right_confidence:
                                neighbours.append(left_disparity)
                                left_mismatch += 1
                            else:
                                neighbours.append(right_disparity)
                                right_mismatch += 1
                        else:
                            neighbours.append(left_disparity)
                            left_mismatch += 1
                        break

                # up
                for h_ in range(h-1, -1, -1):
                    if consistency_map[h_, w] == 0:
                        count += 1
                        if 0<=(w - left_disparity)<width:
                            right_disparity = right_disparity_map[h_, w-int(left_disparity)]
                            right_confidence = right_conf_image[h_, w-int(left_disparity)]
                            if left_confidence > right_confidence:
                                neighbours.append(left_disparity)
                                left_mismatch += 1
                            else:
                                neighbours.append(right_disparity)
                                right_mismatch += 1
                        else:
                            neighbours.append(left_disparity)
                            left_mismatch += 1
                        break

                neighbours = np.array(neighbours, dtype=np.float32)

                # no nearest match, use the raw value
                if count == 0:
                    right_disparity = right_disparity_map_aligned[h, w]
                    right_confidence = right_conf_image_aligned[h, w]
                    if left_confidence > confidence_threshold:
                        if right_confidence > confidence_threshold:
                            int_left_disparity_map[h, w] = (left_disparity + right_disparity)/2
                        else:
                            int_left_disparity_map[h, w] = left_disparity
                    else:
                        if right_confidence > confidence_threshold:
                            int_left_disparity_map[h, w] = right_disparity
                        else:
                            if right_disparity != np.nan:
                                int_left_disparity_map[h, w] = (left_disparity + right_disparity)/2
                            else:
                                int_left_disparity_map[h, w] = left_disparity
                else:
                    int_left_disparity_map[h, w] = np.median(neighbours)
            '''
            else:
                # occlusion
                # just use the nearest match neighbour value on the right
                # NOTE: in the origin paper, they use left rather than left

                # right
                count = 0
                for w_ in range(w+1, width):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        left_disparity = left_disparity_map[h, w_]
                        left_confidence = left_conf_image[h, w_]
                        if 0<=(w_ - left_disparity)<width:
                            right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                            right_confidence = right_conf_image[h, w_-int(left_disparity)]
                            if left_confidence > right_confidence:
                                int_left_disparity_map[h, w] = left_disparity
                                left_ser_occlusion += 1
                            else:
                                int_left_disparity_map[h, w] = right_disparity
                                right_ser_occlusion += 1
                        else:
                            int_left_disparity_map[h, w] = left_disparity
                            left_ser_occlusion += 1
                        break

                # no match neighbour found, use the raw value
                if count == 0:
                    if 0<(w - left_disparity)<width:
                        left_confidence = left_conf_image[h, w]
                        right_confidence = right_conf_image[h, w-int(left_disparity)]
                        if left_confidence > right_confidence:
                            int_left_disparity_map[h, w] = left_disparity
                            left_occlusion += 1
                        else:
                            int_left_disparity_map[h, w] = right_disparity
                            right_occlusion += 1
                    else:
                        int_left_disparity_map[h, w] = left_disparity
                        left_occlusion += 1
            '''

    total_point = height*width
    print('\n total points:{} = height:{} * weigth:{}'.format(total_point,height,width))
    print(' mismatch:{} + occlusion:{} + outside:{} + match:{} = total points:{}'.format(mismatch,occlusion,outside,match,total_point))
    print(' mismatch:{}% + occlusion:{}% + outside:{}% + match:{}%'.format(int(mismatch/total_point*100),int(occlusion/total_point*100),int(outside/total_point*100),int(match/total_point*100)))
    print(' left_ser_mismatch:{} + right_ser_mismatch:{}'.format(left_mismatch,right_mismatch))
    print(' outside:{} + occulusion:{} = left_ser_occulusion:{} + right_ser_occulusion:{}'.format(outside,occlusion,left_ser_occlusion,right_ser_occlusion))
    print('                            + left_occulusion:{} + right_occulusion:{}\n'.format(left_occlusion,right_occlusion))
    left_disparity_map = int_left_disparity_map

    return left_disparity_map

def compute_confidence_disparity_bak(LAF_model_dir,img_path):

    print(" Compute LAF-Net confidence...")
    left_flag = True
    left_conf_image = liblaf.evaluate.compute_LAF_confidence(LAF_model_dir,img_path,left_flag)
    left_flag = False
    right_conf_image = liblaf.evaluate.compute_LAF_confidence(LAF_model_dir,img_path,left_flag)
    left_laf_file = os.path.join(img_path,'left_laf.png')
    right_laf_file = os.path.join(img_path,'right_laf.png')
    io.imsave(left_laf_file,left_conf_image)
    io.imsave(right_laf_file,right_conf_image)

    left_disp_file = os.path.join(img_path,'disp0MCCNN.npy')
    right_disp_file = os.path.join(img_path,'disp1MCCNN.npy')
    left_disparity_map = np.load(left_disp_file)
    right_disparity_map = np.load(right_disp_file)

    height, width = left_disparity_map.shape

    print(" Align the right disparity map by itself if left disparity is unconfidente")
    right_disparity_map_aligned = np.full(left_disparity_map.shape, np.nan)
    right_conf_image_aligned =  np.full(left_disparity_map.shape, 0.0)
    for h in range(height):
        for w in range(width):
            if w + int(right_disparity_map[h,w]) >= 0 and w + int(right_disparity_map[h,w]) < width:
                right_disparity_map_aligned[h,w+ int(right_disparity_map[h,w])] = right_disparity_map[h,w]
                right_conf_image_aligned[h,w+ int(right_disparity_map[h,w])] = right_conf_image[h,w]



    print(" doing left-right consistency check...")
    consistency_map = np.zeros([height, width], dtype=np.int32)

    mismatch = 0
    left_mismatch = 0
    right_mismatch = 0
    occlusion = 0
    left_occlusion = 0
    right_occlusion = 0
    left_ser_occlusion = 0
    right_ser_occlusion = 0
    outside = 0
    match = 0

    for h in range(height):
        for w in range(width):
            left_disparity = int(left_disparity_map[h, w])
            # no corresponding pixel, takes as occlusion
            if ((w - left_disparity)<0)or((w - left_disparity)>=width):
                consistency_map[h, w] = 2
                outside += 1
                continue

            right_disparity = right_disparity_map[h, w-left_disparity]
            if abs(left_disparity - right_disparity) <= 1:
                # match
                match += 1
                continue

            # check if mismatch
            for d in range(w+1):
                if abs(d - right_disparity_map[h, w-d]) <= 1:
                    # mismatch
                    mismatch += 1
                    consistency_map[h, w] = 1
                    break

            # otherwise take as occlusion
            if consistency_map[h, w] == 0:
                consistency_map[h, w] = 2
                occlusion += 1

    print(" Revising noise points")
    std_threshold = 2
    revised_points = 0
    for h in range(1,height-1):
        for w in range(1,width-1):
            if consistency_map[h, w] > 0:
                # 8 neighbor
                neighborSet = np.ndarray(8, dtype=np.float32)
                neighborSet[0] = left_disparity_map[h-1, w-1]
                neighborSet[1] = left_disparity_map[h-1, w]
                neighborSet[2] = left_disparity_map[h-1, w+1]
                neighborSet[3] = left_disparity_map[h, w-1]
                neighborSet[4] = left_disparity_map[h, w+1]
                neighborSet[5] = left_disparity_map[h+1, w-1]
                neighborSet[6] = left_disparity_map[h+1, w]
                neighborSet[7] = left_disparity_map[h+1, w+1]
                std_neightbor = np.nanstd(neighborSet)

                if std_neightbor < std_threshold:
                    mean_neightbor = np.nanmean(neighborSet)
                    left_disparity = left_disparity_map[h, w]
                    diff_point = np.abs(left_disparity - mean_neightbor)
                    if diff_point > 2*std_neightbor:
                        left_disparity_map[h, w] = np.percentile(neighborSet,50)
                        consistency_map[h, w] = 0
                        revised_points += 1
    print(" Revised points number: {}".format(revised_points))

    print(" doing interpolation...")
    int_left_disparity_map = np.ndarray([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):

            left_disparity = left_disparity_map[h, w]
            left_confidence = left_conf_image[h, w]

            if consistency_map[h, w] == 0:
                right_disparity = right_disparity_map[h, w-int(left_disparity)]
                right_confidence = right_conf_image[h, w-int(left_disparity)]
                if left_confidence > right_confidence:
                    int_left_disparity_map[h, w] = left_disparity
                else:
                    int_left_disparity_map[h, w] = right_disparity

            elif consistency_map[h, w] == 1:
                # mismatch, taken median value from nearest match neighbours in 4 directions
                # NOTE: in origin paper, they use 16 directions
                count = 0
                neighbours = []

                # right
                for w_ in range(w+1, width):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        if 0<=(w_ - left_disparity)<width:
                            right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                            right_confidence = right_conf_image[h, w_-int(left_disparity)]
                            if left_confidence > right_confidence:
                                neighbours.append(left_disparity)
                                left_mismatch += 1
                            else:
                                neighbours.append(right_disparity)
                                right_mismatch += 1
                        else:
                            neighbours.append(left_disparity)
                            left_mismatch += 1
                        break

                # left
                for w_ in range(w-1, -1, -1):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        if 0<=(w_ - left_disparity)<width:
                            right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                            right_confidence = right_conf_image[h, w_-int(left_disparity)]
                            if left_confidence > right_confidence:
                                neighbours.append(left_disparity)
                                left_mismatch += 1
                            else:
                                neighbours.append(right_disparity)
                                right_mismatch += 1
                        else:
                            neighbours.append(left_disparity)
                            left_mismatch += 1
                        break

                # bottom
                for h_ in range(h+1, height):
                    if consistency_map[h_, w] == 0:
                        count += 1
                        if 0<=(w - left_disparity)<width:
                            right_disparity = right_disparity_map[h_, w-int(left_disparity)]
                            right_confidence = right_conf_image[h_, w-int(left_disparity)]
                            if left_confidence > right_confidence:
                                neighbours.append(left_disparity)
                                left_mismatch += 1
                            else:
                                neighbours.append(right_disparity)
                                right_mismatch += 1
                        else:
                            neighbours.append(left_disparity)
                            left_mismatch += 1
                        break

                # up
                for h_ in range(h-1, -1, -1):
                    if consistency_map[h_, w] == 0:
                        count += 1
                        if 0<=(w - left_disparity)<width:
                            right_disparity = right_disparity_map[h_, w-int(left_disparity)]
                            right_confidence = right_conf_image[h_, w-int(left_disparity)]
                            if left_confidence > right_confidence:
                                neighbours.append(left_disparity)
                                left_mismatch += 1
                            else:
                                neighbours.append(right_disparity)
                                right_mismatch += 1
                        else:
                            neighbours.append(left_disparity)
                            left_mismatch += 1
                        break

                neighbours = np.array(neighbours, dtype=np.float32)

                # no nearest match, use the raw value
                if count == 0:
                    int_left_disparity_map[h, w] = left_disparity_map[h, w]
                else:
                    int_left_disparity_map[h, w] = np.median(neighbours)

            else:
                # occlusion
                # just use the nearest match neighbour value on the right
                # NOTE: in the origin paper, they use left rather than left

                # right
                count = 0
                for w_ in range(w+1, width):
                    if consistency_map[h, w_] == 0:
                        count += 1
                        left_disparity = left_disparity_map[h, w_]
                        left_confidence = left_conf_image[h, w_]
                        if 0<=(w_ - left_disparity)<width:
                            right_disparity = right_disparity_map[h, w_-int(left_disparity)]
                            right_confidence = right_conf_image[h, w_-int(left_disparity)]
                            if left_confidence > right_confidence:
                                int_left_disparity_map[h, w] = left_disparity
                                left_ser_occlusion += 1
                            else:
                                int_left_disparity_map[h, w] = right_disparity
                                right_ser_occlusion += 1
                        else:
                            int_left_disparity_map[h, w] = left_disparity
                            left_ser_occlusion += 1
                        break

                # no match neighbour found, use the raw value
                if count == 0:
                    if 0<(w - left_disparity)<width:
                        left_confidence = left_conf_image[h, w]
                        right_confidence = right_conf_image[h, w-int(left_disparity)]
                        if left_confidence > right_confidence:
                            int_left_disparity_map[h, w] = left_disparity
                            left_occlusion += 1
                        else:
                            int_left_disparity_map[h, w] = right_disparity
                            right_occlusion += 1
                    else:
                        int_left_disparity_map[h, w] = left_disparity
                        left_occlusion += 1

    total_point = height*width
    print('\n total points:{} = height:{} * weigth:{}'.format(total_point,height,width))
    print(' mismatch:{} + occlusion:{} + outside:{} + match:{} = total points:{}'.format(mismatch,occlusion,outside,match,total_point))
    print(' mismatch:{}% + occlusion:{}% + outside:{}% + match:{}%'.format(int(mismatch/total_point*100),int(occlusion/total_point*100),int(outside/total_point*100),int(match/total_point*100)))
    print(' left_ser_mismatch:{} + right_ser_mismatch:{}'.format(left_mismatch,right_mismatch))
    print(' outside:{} + occulusion:{} = left_ser_occulusion:{} + right_ser_occulusion:{}'.format(outside,occlusion,left_ser_occlusion,right_ser_occlusion))
    print('                            + left_occulusion:{} + right_occulusion:{}\n'.format(left_occlusion,right_occlusion))
    left_disparity_map = int_left_disparity_map

    return left_disparity_map

