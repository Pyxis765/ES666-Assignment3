import pdb
import glob
import cv2
import os
import numpy as np
import random

class PanaromaStitcher:
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, directory):
        image_files = sorted(glob.glob(os.path.join(directory, '*')))
        print(f'Found {len(image_files)} images for stitching.')

        if len(image_files) < 2:
            print("Not enough images to stitch.")
            return None, []

        homography_matrices = []
        base_image = cv2.imread(image_files[0])

        for i in range(1, len(image_files)):
            next_image = cv2.imread(image_files[i])

            kp1, des1, kp2, des2 = self.extract_keypoints(base_image, next_image)
            matches = self.find_good_matches(kp1, kp2, des1, des2)

            matched_points = np.array([[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1],
                                         kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]]
                                        for m in matches])

            homography = self.ransac(matched_points)

            if homography is None:
                print(f"Could not compute homography for image pair {i}.")
                continue

            homography_matrices.append(homography)
            base_image = self.stitch_images(base_image, next_image, homography)

        print("Stitching completed.")
        return base_image, homography_matrices

    def extract_keypoints(self, img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        print(f"Keypoints in first image: {len(kp1)}")
        print(f"Keypoints in second image: {len(kp2)}")

        return kp1, des1, kp2, des2

    def find_good_matches(self, kp1, kp2, des1, des2):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        knn_matches = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in knn_matches if m.distance < 0.7 * n.distance]

        print(f"Good matches found: {len(good_matches)}")
        return good_matches

    def ransac(self, matched_points):
        best_inliers = []
        best_homography = None
        threshold = 5  # Distance threshold for inliers
        for _ in range(10):
            sample = random.sample(matched_points.tolist(), 4)  # Sample 4 points
            H = self.calculate_homography(sample)
            inliers = []

            for point in matched_points:
                p1 = np.array([point[0], point[1], 1]).reshape(3, 1)
                p2 = np.array([point[2], point[3], 1]).reshape(3, 1)
                transformed_point = np.dot(H, p1)
                transformed_point /= transformed_point[2]  # Normalize

                distance = np.linalg.norm(p2 - transformed_point)

                if distance < threshold:
                    inliers.append(point)

            if len(inliers) > len(best_inliers):
                best_inliers, best_homography = inliers, H

        return best_homography

    def calculate_homography(self, points):
        A = []
        for pt in points:
            x, y = pt[0], pt[1]
            X, Y = pt[2], pt[3]
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

        A = np.array(A)
        _, _, vh = np.linalg.svd(A)
        H = (vh[-1, :].reshape(3, 3))
        H /= H[2, 2]  # Normalize
        return H

    def stitch_images(self, base_img, new_img, homography):
        rows1, cols1 = new_img.shape[:2]
        rows2, cols2 = base_img.shape[:2]

        points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        points2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

        transformed_points = cv2.perspectiveTransform(points2, homography)
        all_points = np.concatenate((points1, transformed_points), axis=0)

        [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

        translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(homography)

        stitched_image = cv2.warpPerspective(base_img, translation_matrix, (x_max - x_min, y_max - y_min))
        stitched_image[(-y_min):rows1 + (-y_min), (-x_min):cols1 + (-x_min)] = new_img

        return stitched_image

    def greet(self):
        print('Hello from Naman!')

    def perform_action(self):
        return None

    def perform_additional_action(self):
        return None
