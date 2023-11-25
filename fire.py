import cv2
from Color_Expert import Color_Expert
from Shape_Expert import Shape_Expert
from Movement_Expert import Movement_Expert
from MES import MES

def get_blob(img, bsub):
    blobs = bsub.apply(img)
    blobs = cv2.connectedComponents(blobs)
    return blobs

def get_blobs(imgs):
    blobs = []
    for img in imgs:
        blobs.append(get_blob(img))
    return blobs

def main():
    # Background
    backSub = cv2.createBackgroundSubtractorKNN()

    # Create Experts / MES
    ce = Color_Expert()
    me = Movement_Expert()

    mes = MES(ce, me)

    # Train Experts / MES
    mes.train()

    # Loop / 1 iteration


if __name__ == "__main__":
    main()