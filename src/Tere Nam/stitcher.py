import pdb
import glob
import cv2
import os
import numpy as np
import random

class PanaromaStitcher():
    def __init__(self):
        pass

    def create_panaroma_from_directory(self, directory):
        image_path = directory
        image_files = sorted(glob.glob(image_path + os.sep + '*'))
        print('Found {} images for stitching.'.format(len(image_files)))

        if len(image_files) < 2:
            print("Not enough images to stitch.")
            return None, []

        homography_matrices = []

        base_image = cv2.imread(image_files[0])
        for i in range(1, len(image_files)):
            next_image = cv2.imread(image_files[i])

            keypoints1, descriptors1, keypoints2, descriptors2 = self.extract_keypoints(base_image, next_image)
            matches = self.find_keypoint_matches(keypoints1, keypoints2, descriptors1, descriptors2)

            matched_points = np.array([[keypoints1[m.queryIdx].pt[0], keypoints1[m.queryIdx].pt[1], 
                                         keypoints2[m.trainIdx].pt[0], keypoints2[m.trainIdx].pt[1]]
                                        for m in matches])

            homography_matrix = self.compute_ransac(matched_points)

            if homography_matrix is None:
                print("Could not compute homography for image pair {}.".format(i))
                del next_image
                continue
            
            homography_matrices.append(homography_matrix)
            base_image = self.combine_images(base_image, next_image, homography_matrix)

            del next_image

        print("Stitching finished.")
        return base_image, homography_matrices

    def extract_keypoints(self, img1, img2):
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        print(f"Keypoints in the first image: {len(keypoints1)}")
        print(f"Keypoints in the second image: {len(keypoints2)}")

        return keypoints1, descriptors1, keypoints2, descriptors2

    def find_keypoint_matches(self, keypoints1, keypoints2, descriptors1, descriptors2):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        knn_matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []

        for m, n in knn_matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        print(f"Good matches found: {len(good_matches)}")
        return good_matches

    def compute_ransac(self, matched_points):
        best_inliers = []
        final_homography = None
        threshold = 5  # Inlier distance threshold
        for _ in range(10):
            random_sample = random.sample(matched_points.tolist(), k=4)  # Randomly select 4 points
            H = self.calculate_homography(random_sample)
            inliers = []
            for point in matched_points:
                point1 = np.array([point[0], point[1], 1]).reshape(3, 1)
                point2 = np.array([point[2], point[3], 1]).reshape(3, 1)
                transformed_point = np.dot(H, point1)
                transformed_point /= transformed_point[2]  # Normalize

                distance = np.linalg.norm(point2 - transformed_point)

                if distance < threshold:  # Consider it an inlier
                    inliers.append(point)

            if len(inliers) > len(best_inliers):
                best_inliers, final_homography = inliers, H

        return final_homography

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

    def combine_images(self, base_img, new_img, homography):
        rows1, cols1 = new_img.shape[:2]
        rows2, cols2 = base_img.shape[:2]

        points1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        points2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

        transformed_points = cv2.perspectiveTransform(points2, homography)
        combined_points = np.concatenate((points1, transformed_points), axis=0)

        [x_min, y_min] = np.int32(combined_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(combined_points.max(axis=0).ravel() + 0.5)

        translation_matrix = np.array([[1, 0, (-x_min)], [0, 1, (-y_min)], [0, 0, 1]]).dot(homography)

        stitched_img = cv2.warpPerspective(base_img, translation_matrix, (x_max - x_min, y_max - y_min))
        stitched_img[(-y_min):rows1 + (-y_min), (-x_min):cols1 + (-x_min)] = new_img

        del base_img

        return stitched_img

    def greet(self):
        print('Hello from Naman Varshney!')

    def perform_action(self):
        return None

    def perform_additional_action(self):
        return None
