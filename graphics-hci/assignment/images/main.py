from PIL import Image


def negative_partial(image, start_x, start_y, width, height):
    w, h = image.size
    result_img = image.copy()
    for i in range(start_x, min(start_x + width, w)):
        for j in range(start_y, min(start_y + height, h)):
            r, g, b = image.getpixel((i, j))
            result_img.putpixel((i, j), (255 - r, 255 - g, 255 - b))
    return result_img

def negative(image):
    w, h = image.size
    result_img = Image.new('RGB', (w, h))
    for i in range(w):
        for j in range(h):
            r, g, b = image.getpixel((i, j))
            result_img.putpixel((i, j), (255 - r, 255 - g, 255 - b))
    return result_img


def division(image):
    w, h = image.size
    result_img = Image.new('RGB', (w, h))
    for i in range(w):
        for j in range(h):
            r, g, b = image.getpixel((i, j))
            result_img.putpixel((i, j), (min(255, int(255 / (r + 1))),
                                         min(255, int(255 / (g + 1))),
                                         min(255, int(255 / (b + 1)))))
    return result_img


def darken_mode(image1, image2):
    w, h = image1.size
    result_img = Image.new('RGB', (w, h))
    for i in range(w):
        for j in range(h):
            r1, g1, b1 = image1.getpixel((i, j))
            r2, g2, b2 = image2.getpixel((i, j))
            result_img.putpixel((i, j), (min(r1, r2), min(g1, g2), min(b1, b2)))
    return result_img


def main():
    img = Image.open("daisy-8476666_1280.jpg").convert("RGB")
    img2 = Image.open("new_image.jpg").convert("RGB")

    neg_img = negative(img)
    neg_img.save("negative_image.jpg")

    neg_partial_img = negative_partial(img, 50, 50, 400, 400)
    neg_partial_img.save("negative_partial_image.jpg")
    division_img = division(img)
    division_img.save("division_image.jpg")

    darkened_img = darken_mode(img, img2)
    darkened_img.save("darkened_image.jpg")


if __name__ == "__main__":
    main()
