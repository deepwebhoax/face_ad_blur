


def mosaic_blur(image, rect, step=20):
    h, w = image.shape[:2]
    for i in range(rect[0]+step, rect[2], step):
        for j in range(rect[1]+step, rect[3], step):
            image[i-step:i, j-step:j] = image[i,j]
    return image

def mosaic_blur_multiple(image, rects, step=20):
    for rect in rects:
        rect = [int(r) for r in rect]
        for i in range(rect[1]+step, rect[3], step):
            for j in range(rect[0]+step, rect[2], step):
                image[i-step:i, j-step:j] = image[i,j]
    return image