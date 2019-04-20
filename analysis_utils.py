import numpy as np
import heapq
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
    GRAY_THRESHOLD = 30
    AREA_THRESHOLD = 2500           # in order to ignore outliers

    def __init__(self):
        self.mask = None
        self.contour = None
        # features:
        self.max_area = 0
        self.ex_area = 0
        self.hole_area = 0          # area of internal contours
        self.perimeter = 0
        self.equi_area_diameter = 0
        self.min_circle_radius = 0
        self.solidity = 0
        self.num_colony = 0         # number of external contours
        self.num_hole = 0           # number of internal contours
        self.centroids_dist = 0     # distance of centroids of the largest two areas

    def preprocess_mask(self, mask):
        self.mask = mask
        self.mask[self.mask < self.GRAY_THRESHOLD] = 0
        self.mask[self.mask > 0] = 255

    def cal_contour_and_area(self):
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # a list of arrays

        # areas of external contours
        ex_areas = [(cv2.contourArea(cnt), cnt) for cnt, h in zip(contours, hierarchy[0]) if h[-1] == -1]
        ex_areas = np.array([tup for tup in ex_areas if tup[0] > self.AREA_THRESHOLD])
        self.num_colony = len(ex_areas)
        assert self.num_colony > 0, 'no valid cells: ' + str(self.num_colony)
        self.ex_area = np.sum(ex_areas[:, 0])                 # assume external area is not empty

        # internal contours
        in_areas = [(cv2.contourArea(cnt), cnt) for cnt, h in zip(contours, hierarchy[0]) if h[-1] != -1]
        in_areas = np.array([tup for tup in in_areas if tup[0] > self.AREA_THRESHOLD])
        self.num_hole = len(in_areas)
        self.hole_area = 0 if not self.num_hole else np.sum(in_areas[:, 0])

        self.ex_area -= self.hole_area

        large_two = heapq.nlargest(2, ex_areas, lambda tup: tup[0])
        self.cal_centroids_dist(large_two)

        # pick the largest one
        self.contour = large_two[0][1]
        self.max_area = large_two[0][0] - self.hole_area      # assume holes are inside of the contour of the max area

    def cal_centroids_dist(self, large_tup):
        if len(large_tup) == 1:         # only one valid external area
            self.centroids_dist = 0
        else:
            momen = [cv2.moments(tup[1]) for tup in large_tup]
            cx = [m['m10'] / m['m00'] for m in momen]
            cy = [m['m01'] / m['m00'] for m in momen]
            self.centroids_dist = np.linalg.norm([cx[0] - cx[1], cy[0] - cy[1]])

    def cal_perimeter(self):
        self.perimeter = cv2.arcLength(self.contour, True)

    def cal_equi_area_diameter(self):
        assert self.max_area > 0, 'self.area should > 0: ' + str(self.max_area)
        self.equi_area_diameter = np.sqrt(4 * self.max_area / np.pi)

    def cal_min_radius(self):
        _, self.min_circle_radius = cv2.minEnclosingCircle(self.contour)

    def cal_solidity(self):
        assert self.max_area > 0, 'self.area should > 0: ' + str(self.max_area)
        hull = cv2.convexHull(self.contour)
        hull_area = cv2.contourArea(hull)
        self.solidity = float(self.max_area) / hull_area

    def get_fea_vec_0(self, mask):
        # combine all the feature values together
        self.preprocess_mask(mask)

        self.cal_contour_and_area()
        self.cal_perimeter()
        self.cal_equi_area_diameter()
        self.cal_min_radius()
        self.cal_solidity()

        fea_vec = np.array([self.ex_area, self.hole_area, self.max_area,
                            self.perimeter, self.equi_area_diameter, self.min_circle_radius,
                            self.solidity, self.num_colony, self.num_hole,
                            self.centroids_dist], dtype='float32')
        return fea_vec

    def get_fea_vec_1(self, mask):
        # just two features
        self.preprocess_mask(mask)

        self.cal_contour_and_area()
        self.cal_solidity()

        fea_vec = np.array([self.ex_area, self.num_colony])

        return fea_vec


if __name__ == '__main__':
    # test a single image
    mask_path = 'results/predict/MM_DMSO_N3/H4_M1/predMask/predMask_40.png'
    mask = cv2.imread(mask_path, 0)

    extractor = FeatureExtractor()
    vec = extractor.get_fea_vec_0(mask)
    print(vec)
