import cv2
import numpy as np
import random

class Match(object):
    def __init__(self, key_1, des_1, key_2, des_2, dis):
        self.key_1 = key_1
        self.key_2 = key_2
        self.des_1 = des_1
        self.des_2 = des_2
        self.dis = dis

    def __str__(self):
        return str(self.dis)

class Matcher():
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()

    def extract_keypoint_and_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, None)

    def match(self, image_1, image_2, thres=300):
        keypoints_1, descriptors_1 = self.extract_keypoint_and_features(
            image_1)
        keypoints_2, descriptors_2 = self.extract_keypoint_and_features(
            image_2)
        des_matches = list()
        for keypoint_1, descriptor_1 in zip(keypoints_1, descriptors_1):
            min_dis = 1.7976931348623157e+308
            vec_1 = np.array(descriptor_1)
            for keypoint_2, descriptor_2 in zip(keypoints_2, descriptors_2):
                vec_2 = np.array(descriptor_2)
                dis = np.linalg.norm(vec_1 - vec_2)
                if dis < min_dis:
                    min_dis = dis
                    des_2 = descriptor_2
                    key_2 = keypoint_2
            des_matches.append(
                Match(keypoint_1, descriptor_1, key_2, des_2, min_dis))
        result = [match for match in des_matches if match.dis < thres]
        result = sorted(result, key=lambda x: x.dis)
        return result

    def concate_image(self, img1, img2):
        shape1 = img1.shape
        shape2 = img2.shape

        new_width = shape1[1] + shape2[1]
        new_height = max(shape1[0], shape2[0])

        res = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        res[0: shape1[0], 0: shape1[1]] = img1
        res[0: shape2[0], shape1[1]: new_width] = img2
        return res

    def draw_match(self, img3, match, color, shift):
        point_1 = match.key_1.pt
        point_2 = match.key_2.pt
        col = shift

        point_1 = (np.float32(point_1[0]), np.float32(point_1[1]))
        point_2 = (np.float32(point_2[0] + col), np.float32(point_2[1]))
        cv2.line(img3, point_1, point_2, color, 1)

    def draw_key_point(self, img3, match, shift):
        col = shift
        keypoint_1 = (np.float32(
            match.key_1.pt[0]), np.float32(match.key_1.pt[1]))
        keypoint_2 = (np.float32(
            match.key_2.pt[0] + col), np.float32(match.key_2.pt[1]))

        cv2.circle(img3, keypoint_1, 5, (255, 255, 255))
        cv2.circle(img3, keypoint_2, 5, (255, 255, 255))

    def draw_result(self, img1, img2, match_result, number_of_points=50,
                    reduce_points=False, inliers=None):
        img3 = self.concate_image(img1, img2)
        if reduce_points:
            if len(match_result) < number_of_points:
                match_result = match_result[:number_of_points]
        for match in match_result:
            self.draw_key_point(img3, match, img1.shape[1])
            if inliers is not None and match in inliers:
                self.draw_match(img3, match, (0, 255, 0), img1.shape[1])
            elif inliers is not None and match not in inliers:
                self.draw_match(img3, match, (0, 0, 255), img1.shape[1])
            else:
                self.draw_match(img3, match, (255, 0, 255), img1.shape[1])
        return img3

    def show_result_image(self, img, name="result"):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cal_distance(self, match, h):
        mat_1 = np.transpose(np.matrix([match[0], match[1], 1]))
        mat_2 = np.transpose(np.matrix([match[2], match[3], 1]))
        estimate_p2 = np.dot(h, mat_1)

        estimate_p2 = estimate_p2 / estimate_p2.item(2)
        error = mat_2 - estimate_p2
        return np.linalg.norm(error)

    def find_homography(self, pairs):
        A = list()
        for pair in pairs:
            p1 = (pair[0], pair[1])
            p2 = (pair[2], pair[3])
            i_1 = [p1[0], p1[1], 1, 0, 0, 0, -
            p2[0] * p1[0], -p2[0] * p1[1], -p2[0]]
            i_2 = [0, 0, 0, p1[0], p1[1], 1, -
            p2[1] * p1[0], -p2[1] * p1[1], -p2[1]]
            A.append(i_1)
            A.append(i_2)

        matrix_A = np.matrix(A)
        u, s, v = np.linalg.svd(matrix_A)
        homo = np.reshape(v[8], (3, 3))
        return homo

    def ransac(self, match_pairs, inlier_thres=0.65):
        correspondence_list = list()
        for match in match_pairs:
            (x1, y1) = match.key_1.pt
            (x2, y2) = match.key_2.pt
            correspondence_list.append([x1, y1, x2, y2])
        max_inliers = []
        homo_final = None
        for i in range(1000):
            temp = list()
            pair_1 = correspondence_list[random.randrange(
                0, len(correspondence_list))]
            pair_2 = correspondence_list[random.randrange(
                0, len(correspondence_list))]
            pair_3 = correspondence_list[random.randrange(
                0, len(correspondence_list))]
            pair_4 = correspondence_list[random.randrange(
                0, len(correspondence_list))]
            temp.append(pair_1)
            temp.append(pair_2)
            temp.append(pair_3)
            temp.append(pair_4)
            homo = self.find_homography(temp)
            inliers = []
            for i in range(len(correspondence_list)):
                dis = self.cal_distance(correspondence_list[i], homo)
                if dis < 5:
                    inliers.append(match_pairs[i])
            if len(inliers) > len(max_inliers):
                max_inliers = inliers
                homo_final = homo
            if len(max_inliers) > len(correspondence_list) * inlier_thres:
                break
        return max_inliers, homo_final

    def feature_matching(self, image_left, image_right, show_result=True,
                         with_homo=True):
        match = self.match(image_left, image_right)
        if len(match) < 4:
            return [], None
        inliers, homo = self.ransac(match)
        if show_result:
            if with_homo:
                result = self.draw_result(
                    image_left, image_right, match, inliers=inliers)
            else:
                result = self.draw_result(image_left, image_right, match)
            self.show_result_image(result)
        return inliers, homo

    def feature_matching_using_opencv(self, image_1,
                                      image_2,
                                      show_result=True):
        kps1, des1 = self.extract_keypoint_and_features(image_1)
        kps2, des2 = self.extract_keypoint_and_features(image_2)
        kps1 = np.float32([kp.pt for kp in kps1])
        kps2 = np.float32([kp.pt for kp in kps2])

        matcher = cv2.BFMatcher()
        raw_matches = matcher.match(des1, des2)
        raw_matches = sorted(raw_matches, key=lambda x: x.distance)
        matches = []
        for m in raw_matches:
            matches.append((m.trainIdx, m.queryIdx))
        if len(matches) > 4:
            ptsA = np.float32([kps1[i] for (_, i) in matches])
            ptsB = np.float32([kps2[i] for (i, _) in matches])
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

            if show_result:
                img3 = self.concate_image(image_1, image_2)
                for ((trainIdx, queryIdx), s) in zip(matches, status):
                    ptA = (np.float32(kps1[queryIdx][0]),
                           np.float32(kps1[queryIdx][1]))
                    ptB = (np.float32(
                        kps2[trainIdx][0] + image_1.shape[1]),
                           np.float32(kps2[trainIdx][1]))
                    if s == 1:
                        cv2.line(img3, ptA, ptB, (0, 255, 0), 1)
                    else:
                        cv2.line(img3, ptA, ptB, (0, 0, 255), 1)
                self.show_result_image(img3)
        return H
