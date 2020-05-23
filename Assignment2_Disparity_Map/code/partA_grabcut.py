import numpy as np
import cv2


def grabcut(img, rect, name, manual_mask=None, show_result=False):
    # initialize mask, background model, foreground model
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    if manual_mask is not None:
        # if a manual marked mask are given to the image
        # wherever it is marked white (sure foreground), change mask=1
        # wherever it is marked black (sure background), change mask=0
        mask[manual_mask == 0] = 0
        mask[manual_mask == 255] = 1
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # change mask=0 (sure background), mask=2 (probable background) into value 0
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask[:, :, np.newaxis]

    if show_result:
        cv2.imshow(name, img)
        cv2.waitKey(0)
    # write image to file
    cv2.imwrite("grabcut_result/" + name + ".jpg", img)


basketball_img = cv2.imread("grabcut/a.jpg")
basketball_rect = (725, 185, 275, 275)
basketball_name = 'basketball'
basketball_mask = cv2.imread("grabcut_reslut/mask/a_mask.png", 0)
grabcut(basketball_img, basketball_rect, basketball_name, basketball_mask)

Donald_duck_img = cv2.imread("grabcut/b.jpg")
Donald_duck_rect = (30, 75, 400, 700)
Donald_duck_name = 'Donald_duck'
Donald_duck_mask = cv2.imread("grabcut_result/mask/b_mask.png", 0)
grabcut(Donald_duck_img, Donald_duck_rect, Donald_duck_name, Donald_duck_mask)

statue_img = cv2.imread("grabcut/c.jpg")
statue_rect = (50, 40, 385, 475)
statue_name = 'statue'
statue_mask = cv2.imread("grabcut_result/mask/c_mask.png", 0)
grabcut(statue_img, statue_rect, statue_name, statue_mask)
