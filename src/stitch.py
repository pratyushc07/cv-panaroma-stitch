import cv2
import numpy as np
import matplotlib.pyplot as plt

class PanoramaStitcher:
    def __init__(self):
        # using SIFT because it's scale invariant and works better for rotation
        self.sift = cv2.SIFT_create()

    def get_keypoints_and_matches(self, img1, img2):
        """
        Takes two images, finds keypoints, and returns the good matches.
        """
        # step 1: get the keypoints and descriptors for both images
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        # step 2: match them up
        # using BFMatcher with k=2 so we can apply the ratio test later
        bf = cv2.BFMatcher()
        raw_matches = bf.knnMatch(des1, des2, k=2)

        # step 3: filter out the bad matches using Lowe's ratio test
        # essentially, if the best match is much better than the second best, keep it
        good_matches = []
        for m, n in raw_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        return kp1, kp2, good_matches

    def find_homography(self, kp1, kp2, matches):
        """
        Calculates the transformation matrix (H) needed to map img1 onto img2.
        """
        # need at least 4 points to solve for the homography matrix
        if len(matches) < 4:
            print("Not enough matches to compute homography!")
            return None

        # grab the coordinates of the matching points from both images
        # queryIdx is from the first image, trainIdx is from the second
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # calculate the matrix using RANSAC to ignore outliers (noise)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    
    def stitch_images(self, img_left, img_center, img_right):
        # PHASE 1: Calculate the transforms 
    
        # first, map the left image to the center
        print("Matching left to center...")
        kp_l, kp_c, matches_lc = self.get_keypoints_and_matches(img_left, img_center)
        H_lc = self.find_homography(kp_l, kp_c, matches_lc)
        
        # next, map the right image to the center
        print("Matching right to center...")
        kp_r, _, matches_rc = self.get_keypoints_and_matches(img_right, img_center)
        H_rc = self.find_homography(kp_r, kp_c, matches_rc)

        if H_lc is None or H_rc is None:
            print("Could not find homography. Check your images.")
            return None

        # PHASE 2: Figure out the canvas size
        # since we are warping images, they might fall outside the normal frame.
        # we need to project the corners to find the new bounds.
        
        h_l, w_l = img_left.shape[:2]
        h_r, w_r = img_right.shape[:2]
        h_c, w_c = img_center.shape[:2]
        

        corners_l = np.float32([[0, 0], [0, h_l], [w_l, h_l], [w_l, 0]]).reshape(-1, 1, 2)
      
        warped_corners_l = cv2.perspectiveTransform(corners_l, H_lc)
       
        corners_r = np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)
        warped_corners_r = cv2.perspectiveTransform(corners_r, H_rc)
        
        corners_c = np.float32([[0, 0], [0, h_c], [w_c, h_c], [w_c, 0]]).reshape(-1, 1, 2)
        
        # stack all corners together to find the min/max x and y
        all_corners = np.concatenate((warped_corners_l, corners_c, warped_corners_r), axis=0)
        
        # find the extent of the final panorama
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # we need a translation matrix because if xmin is negative, the image will be cut off.
        # this shifts everything to the positive side.
        t = [-xmin, -ymin]
        H_translation = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)

        #PHASE 3: Warping everything
        
        output_size = (xmax - xmin, ymax - ymin)

        # Warp left image -> applies translation * homography
        output_img = cv2.warpPerspective(img_left, H_translation.dot(H_lc), output_size)
        
        # Warp right image
        warped_right = cv2.warpPerspective(img_right, H_translation.dot(H_rc), output_size)
        
        # Warp center image 
        warped_center = cv2.warpPerspective(img_center, H_translation, output_size)

        # PHASE 4: Blending
        
        # 1. Naive stitch (Just overwriting pixels)
        # This shows the seams clearly
        naive_result = output_img.copy()
        naive_result[warped_center > 0] = warped_center[warped_center > 0]
        naive_result[warped_right > 0] = warped_right[warped_right > 0]

        # 2. Better blending
        # Using np.maximum to take the brighter pixel in overlap regions.
        # This isn't perfect (pyramid blending is better) but it handles basic overlaps
        # better than just overwriting.
        final_result = np.maximum(output_img, warped_center)
        final_result = np.maximum(final_result, warped_right)

        return naive_result, final_result

#running the pipeline

# Load images
print("Loading images...")

img_l = cv2.imread('images/left.jpeg')
img_c = cv2.imread('images/centre.jpeg')
img_r = cv2.imread('images/right.jpeg')

if img_l is None or img_c is None:
    print("Error loading images!")
    exit()

# resizing to make processing faster during testing
scale_factor = 0.5
img_l = cv2.resize(img_l, None, fx=scale_factor, fy=scale_factor)
img_c = cv2.resize(img_c, None, fx=scale_factor, fy=scale_factor)
img_r = cv2.resize(img_r, None, fx=scale_factor, fy=scale_factor)

stitcher = PanoramaStitcher()
naive_pano, final_pano = stitcher.stitch_images(img_l, img_c, img_r)

# Displaying results
cv2.imshow("Naive Stitch (Seams Visible)", naive_pano)
cv2.imshow("Final Panorama", final_pano)

print("Press any key to close windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saving output images
cv2.imwrite("results/naive_stitch.jpg", naive_pano)
cv2.imwrite("results/final_panorama.jpg", final_pano)
print("Results saved!")