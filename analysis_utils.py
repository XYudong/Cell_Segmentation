import numpy as np
import cv2


def draw_contours(mask):
    AREA_THRESHOLD = 5000
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > AREA_THRESHOLD:
            print(area)
            cv2.drawContours(mask, [cnt], 0, (0, 250, 0), 3)


class FeatureExtractor:
    GRAY_THRESHOLD = 100
    AREA_THRESHOLD = 5000

    def __init__(self):
        self.mask = None
        self.contour = None
        self.area = 0
        self.perimeter = 0
        self.equi_area_diameter = 0
        self.min_circle_radius = 0
        self.solidity = 0
        self.num_colony = 0

    def preprocess_mask(self, mask):
        self.mask = mask
        self.mask[self.mask < self.GRAY_THRESHOLD] = 0

    def cal_contour_and_area(self):
        contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # a list of arrays

        # pick the one with the largest area
        areas = np.array([cv2.contourArea(cnt) for cnt in contours])
        idx = np.argmax(areas)
        self.contour = contours[idx]
        self.area = areas[idx]
        self.num_colony = np.sum(areas > self.AREA_THRESHOLD)

    def cal_perimeter(self):
        self.perimeter = cv2.arcLength(self.contour, True)

    def cal_equi_area_diameter(self):
        assert self.area > 0, 'self.area should > 0: ' + str(self.area)
        self.equi_area_diameter = np.sqrt(4 * self.area / np.pi)

    def cal_min_radius(self):
        _, self.min_circle_radius = cv2.minEnclosingCircle(self.contour)

    def cal_solidity(self):
        assert self.area > 0, 'self.area should > 0: ' + str(self.area)
        hull = cv2.convexHull(self.contour)
        hull_area = cv2.contourArea(hull)
        self.solidity = float(self.area) / hull_area

    def get_feature_vec(self, mask):
        self.preprocess_mask(mask)

        self.cal_contour_and_area()
        self.cal_perimeter()
        self.cal_equi_area_diameter()
        self.cal_min_radius()
        self.cal_solidity()

        fea_vec = np.array([self.area, self.perimeter, self.equi_area_diameter,
                            self.min_circle_radius, self.solidity, self.num_colony], dtype='float32')
        return fea_vec

