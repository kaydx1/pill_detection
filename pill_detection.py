import cv2 as cv
import numpy as np
import os


def apply_brightness_contrast(input_img, brightness=0, contrast=0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


test_dir = './Examples/'
output_dir = './Results/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for file in os.listdir(test_dir):
    img_rgb = cv.imread(os.path.join(test_dir, file), cv.IMREAD_UNCHANGED)
    dst = img_rgb[0:290, 1550:1920]
    img_rgb[0:290, 1550:1920] = apply_brightness_contrast(dst, -65, 40)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    ret, edges = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY_INV)

    template = cv.imread('pill_template.bmp', 0)
    ret2, template_edges = cv.threshold(
        template, 200, 255, cv.THRESH_BINARY_INV)

    contours, hierarchy = cv.findContours(
        edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[1::]

    for c in contours:
        area = cv.contourArea(c)

        if area > 4000 or area < 3000:
            cv.fillPoly(edges, pts=[c], color=255)
            continue

        rect = cv.minAreaRect(c)
        (x, y), (w, h), angle = rect
        aspect_ratio = max(w, h) / min(w, h)

        if (aspect_ratio > 1.2 or aspect_ratio < 0.8):
            cv.fillPoly(edges, pts=[c], color=255)
            continue

    edges = cv.resize(edges, (960, 540))
    res = cv.matchTemplate(edges, template_edges, cv.TM_CCOEFF_NORMED)

    w, h = template.shape[::-1]

    threshold = 0.7
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    h_rgb, w_rgb, _ = img_rgb.shape

    count = 0
    res = []
    v_points = [225 + 163*i for i in range(5)]
    for i in v_points:
        j = 0
        while j < w_rgb:
            if img_rgb[i, j][2] == 255 and img_rgb[i, j][1] == 0:
                count += 1
                res.append((i, j))
                j += 120
            j += 1

    h_points = [105]
    for i in range(1, 10):
        k = h_points[-1]
        if i % 2 != 0:
            k += 163
        else:
            k += 222
        h_points.append(k)

    for j in h_points:
        i = 0
        while i < h_rgb:
            if img_rgb[i, j][2] == 255 and img_rgb[i, j][1] == 0:
                count += 1
                res.append((i, j))
                i += 120
            i += 1

    result = False
    if count == 100:
        result = True
    for i in h_points:
        cv.line(img_rgb, (i, 0), (i, h_rgb), (250, 0, 0), 3)

    for j in v_points:
        cv.line(img_rgb, (0, j), (w_rgb, j), (250, 0, 0), 3)

    if result:
        cv.putText(img_rgb, 'Pass', (800, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 4, (0, 128, 0), 4, cv.LINE_AA)
    else:
        cv.putText(img_rgb, 'Reject', (800, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 4, (128, 0, 128), 4, cv.LINE_AA)

    cv.imwrite(os.path.join(output_dir, file), img_rgb)
