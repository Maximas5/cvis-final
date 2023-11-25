import cv2
from Color_Expert import Color_Expert
from Shape_Expert import Shape_Expert
from Movement_Expert import Movement_Expert
from MES import MES

def foreground_mask_extraction(fIm, bmask):
    pass
    return bmask, mask

def background_update(bmask):
    return bmask

def connected_comp_label(mask):
    return blobs

def color_expert(fblobs, bblobs):
    pass

def shape_expert(fblobs, bblobs):
    pass

def movement_expert(fblobs, bblobs):
    pass

def main():
    # Create Experts / MES
    ce = Color_Expert()
    me = Movement_Expert()

    mes = MES(ce, me)

    # Train Experts / MES
    mes.train()

    # Loop / 1 iteration


if __name__ == "__main__":
    main()