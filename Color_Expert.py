from Expert import Expert

class Color_Expert(Expert):
    def __init__(self):
        super().__init__()
        self.thresh = 40

    # Blobs are arrays of length 3 tuples representing pixels converted to YUV
    def predict(self, blob, YMean, UMean, VMean):
        # blob = cv2.cvtColor(blob, cv2.COLOR_BGR2YUV)
        fire = True

        # For each pixel
        for pixel in blob:
            y = pixel[0]
            u = pixel[1]
            v = pixel[2]
            # if all rules satisfied, continue
            if (self.r1(y, u) and self.r2(v, u) and self.r3(y, YMean) 
            and self.r4(u, UMean) and self.r5(v, VMean) and self.r6(v, u)):
                continue
            # else false. Break
            else:
                fire = False
                break

        return fire

    def r1(self, Y, U):
        # Y value is greater than U value
        return Y > U

    def r2(self, V, U):
        # V value is greater than U value
        return V > U

    def r3(self, Y, Ymean):
        # Y is greater than the mean Y value of the entire image
        return Y > Ymean

    def r4(self, U, Umean):
        # U is greater than the mean U value of the entire image
        return U > Umean

    def r5(self, V, VMean):
        # V is less than the mean V value of the entire image
        return V < VMean

    def r6(self, V, U):
        # Difference between V and U values is considerable
        # Threshold defaults to 40 as prescribed by the paper
        return abs(V - U) >= self.thresh