import glob
import cv2
import os
import numpy as np

class PanaromaStitcher:
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} images for stitching'.format(len(all_images)))

        if len(all_images) < 2:
            raise ValueError("At least two images are required to create a panorama.")

        # Prepare to collect homography matrices
        homography_matrix_list = []
        images = [cv2.imread(im) for im in all_images]

        # Check if images are loaded properly
        for img in images:
            if img is None or img.size == 0:
                print("Warning: One of the images is empty or not loaded correctly.")
        
        # Initialize the first image as the base
        current_image = images[0]

        for i in range(1, len(images)):
            # Detect and compute features
            kp1, des1 = self.detect_and_compute_features(current_image)
            kp2, des2 = self.detect_and_compute_features(images[i])

            if des1 is None or des2 is None:
                print(f"Warning: Descriptors could not be computed for images {i-1} and {i}.")
                continue

            # Match features between the two images
            matches = self.match_features(des1, des2)

            print(f"Processing image pair {i-1} and {i}. Found {len(matches)} matches.")

            if len(matches) >= 4:  # Need at least 4 matches to find homography
                H = self.estimate_homography(kp1, kp2, matches)

                if H is not None:
                    homography_matrix_list.append(H)
                    # Warp the next image to the current panorama
                    current_image = self.warp_images(current_image, images[i], H)
                else:
                    print(f"Homography could not be computed for images {i-1} and {i}.")
            else:
                print(f"Not enough matches found between images {i-1} and {i}: {len(matches)} matches.")

        # Return Final panorama and Homography matrices
        return current_image, homography_matrix_list

    def detect_and_compute_features(self, image):
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(image, None)
        if des is None:
            print("Warning: No descriptors found for the image.")
        return kp, des

    def match_features(self, des1, des2):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)

    def estimate_homography(self, kp1, kp2, matches):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        A = []
        for i in range(len(matches)):
            x1, y1 = src_pts[i]
            x2, y2 = dst_pts[i]
            A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

        A = np.array(A)
        _, _, VT = np.linalg.svd(A)
        H = VT[-1].reshape(3, 3)

        return H

    def warp_images(self, base_image, new_image, H):
        h1, w1 = base_image.shape[:2]
        h2, w2 = new_image.shape[:2]

        corners = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)

        all_corners = np.concatenate((corners, warped_corners), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel()) - 5
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel()) + 5

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        result_size = (x_max - x_min, y_max - y_min)
        result = cv2.warpPerspective(base_image, translation, result_size)

        warped_image = cv2.warpPerspective(new_image, translation @ H, result_size)

        mask = (warped_image > 0)
        result[mask] = warped_image[mask]

        return result
