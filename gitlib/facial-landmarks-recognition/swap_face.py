import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe face mesh.
mp_face_mesh = mp.solutions.face_mesh

def get_face_landmarks(image):
    """
    Process the image with MediaPipe and return a list of landmark (x,y) tuples.
    If no face is found, returns None.
    """
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            h, w = image.shape[:2]
            # Convert normalized coordinates to pixel coordinates.
            pts = [(int(l.x * w), int(l.y * h)) 
                   for l in results.multi_face_landmarks[0].landmark]
            return pts
        return None

def clamp_rect(x, y, w, h, img_w, img_h):
    """
    Clamp (x,y,w,h) so the rectangle lies within the image.
    Returns (x,y,w,h) if valid, or None otherwise.
    """
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)

def clamp_points_to_image(pts, img_w, img_h):
    """
    Clamps the (x,y) coordinates in an array of points
    so that they lie within [0, img_w-1] and [0, img_h-1].
    """
    c = pts.copy()
    c[:, 0] = np.clip(c[:, 0], 0, img_w - 1)
    c[:, 1] = np.clip(c[:, 1], 0, img_h - 1)
    return c

def adjust_center(center, src_w, src_h, dst_w, dst_h):
    """
    Shifts the center so that the ROI of size (src_w x src_h),
    when centered at 'center', lies completely within a destination
    of size (dst_w x dst_h). Returns the adjusted center or None if
    it is impossible to fit the ROI.
    """
    cx, cy = center
    roi_x = cx - src_w // 2
    roi_y = cy - src_h // 2
    roi_x2 = roi_x + src_w
    roi_y2 = roi_y + src_h

    # Adjust if the ROI extends off the left or top
    if roi_x < 0:
        cx += -roi_x
    if roi_y < 0:
        cy += -roi_y

    roi_x = cx - src_w // 2
    roi_y = cy - src_h // 2
    roi_x2 = roi_x + src_w
    roi_y2 = roi_y + src_h

    # Adjust if the ROI extends off the right or bottom
    if roi_x2 > dst_w:
        cx -= roi_x2 - dst_w
    if roi_y2 > dst_h:
        cy -= roi_y2 - dst_h

    # Final check to be safe
    roi_x = cx - src_w // 2
    roi_y = cy - src_h // 2
    roi_x2 = roi_x + src_w
    roi_y2 = roi_y + src_h
    if roi_x < 0 or roi_y < 0 or roi_x2 > dst_w or roi_y2 > dst_h:
        return None
    return (cx, cy)

def warp_triangle(img_src, img_dst, t_src, t_dst):
    """
    Warps and blends a triangular region from img_src to img_dst using an affine transformation.
    """
    # Find bounding rectangles for each triangle
    r_src = cv2.boundingRect(np.float32([t_src]))
    r_dst = cv2.boundingRect(np.float32([t_dst]))

    # Offset points by the top left corner of the respective rectangles
    t_src_offset = []
    t_dst_offset = []
    t_dst_offset_int = []

    for i in range(3):
        t_src_offset.append(((t_src[i][0] - r_src[0]), (t_src[i][1] - r_src[1])))
        t_dst_offset.append(((t_dst[i][0] - r_dst[0]), (t_dst[i][1] - r_dst[1])))
        t_dst_offset_int.append((int(t_dst[i][0] - r_dst[0]), int(t_dst[i][1] - r_dst[1])))

    # Get the affine transformation matrix for the triangle
    warp_mat = cv2.getAffineTransform(np.float32(t_src_offset), np.float32(t_dst_offset))
    
    # Warp the source patch to fit the destination patch size
    img_src_patch = img_src[r_src[1]: r_src[1] + r_src[3], r_src[0]: r_src[0] + r_src[2]]
    warped_patch = cv2.warpAffine(img_src_patch, warp_mat, (r_dst[2], r_dst[3]), 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    # Create a mask for the destination triangle
    mask = np.zeros((r_dst[3], r_dst[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(t_dst_offset_int), 255)
    
    # Extract the destination patch, and blend the warped patch using the mask
    img_dst_patch = img_dst[r_dst[1]: r_dst[1] + r_dst[3], r_dst[0]: r_dst[0] + r_dst[2]]
    img_dst_patch = cv2.bitwise_and(img_dst_patch, img_dst_patch, mask=cv2.bitwise_not(mask))
    img_dst_patch = img_dst_patch + cv2.bitwise_and(warped_patch, warped_patch, mask=mask)
    
    img_dst[r_dst[1]: r_dst[1] + r_dst[3], r_dst[0]: r_dst[0] + r_dst[2]] = img_dst_patch

def apply_delaunay_triangulation(src_img, dst_img, src_points, dst_points, hull_index):
    """
    Computes Delaunay triangulation based on the convex hull of the destination points,
    and warps each triangle from the source image to the destination image.
    """
    # Create convex hull lists for source and destination landmarks.
    src_hull = [src_points[i] for i in hull_index]
    dst_hull = [dst_points[i] for i in hull_index]
    
    # Compute bounding rectangle for Delaunay triangulation (cover whole destination image)
    size = dst_img.shape
    rect = (0, 0, size[1], size[0])
    
    subdiv = cv2.Subdiv2D(rect)
    for p in dst_hull:
        subdiv.insert((p[0], p[1]))
    
    triangle_list = subdiv.getTriangleList()
    triangle_indices = []

    # For each triangle in the output, lookup the index of each vertex
    for t in triangle_list:
        pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        indices = []
        for p in pt:
            for k, dp in enumerate(dst_hull):
                if np.linalg.norm(np.array(p) - np.array(dp)) < 1.0:
                    indices.append(k)
                    break
        if len(indices) == 3:
            triangle_indices.append(indices)
            
    # Warp each triangle from src to dst
    for tri in triangle_indices:
        t_src = [src_hull[tri[0]], src_hull[tri[1]], src_hull[tri[2]]]
        t_dst = [dst_hull[tri[0]], dst_hull[tri[1]], dst_hull[tri[2]]]
        warp_triangle(src_img, dst_img, t_src, t_dst)
    
    return dst_img

def face_swap(src_img, dst_img, src_points, dst_points):
    """
    Given two images and their corresponding facial landmark points,
    performs face swapping by warping the source face to the destination face
    and then blending it using seamless cloning.
    """
    # Compute convex hull for destination face based on landmarks.
    hull_index = cv2.convexHull(np.array(dst_points), returnPoints=False).flatten()
    
    # Make a copy of the destination image to store the warped face.
    warped_face = np.copy(dst_img)
    warped_face = apply_delaunay_triangulation(src_img, warped_face, src_points, dst_points, hull_index)
    
    # Create a mask from the destination face convex hull.
    dst_hull = np.array([dst_points[int(i)] for i in hull_index], dtype=np.int32)
    mask = np.zeros(dst_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_hull, 255)
    
    # Find the center for blending (average of convex hull points or bounding rect center)
    rect = cv2.boundingRect(dst_hull)
    center = (rect[0] + rect[2]//2, rect[1] + rect[3]//2)
    
    # Use seamless cloning for natural blending.
    output = cv2.seamlessClone(warped_face, dst_img, mask, center, cv2.NORMAL_CLONE)
    # Alternatively, you can try cv2.MIXED_CLONE for different blending results:
    # output = cv2.seamlessClone(warped_face, dst_img, mask, center, cv2.MIXED_CLONE)
    
    return output

# -------------------- Main Script --------------------
# 1) Load source (face to swap) and destination images.
image1 = cv2.imread("image3.jpg")   # Destination image.
image2 = cv2.imread("image8.jpeg")  # Source image.

if image1 is None or image2 is None:
    print("Không thể đọc một hoặc hai ảnh. Kiểm tra đường dẫn!")
    exit(1)

# 2) Detect face landmarks.
landmarks1 = get_face_landmarks(image1)
landmarks2 = get_face_landmarks(image2)

if landmarks1 is None or landmarks2 is None:
    print("Không nhận diện được khuôn mặt trong một hoặc cả hai ảnh.")
    exit(1)

# Convert landmark lists into numpy arrays.
pts1 = np.array(landmarks1, dtype=np.int32)
pts2 = np.array(landmarks2, dtype=np.int32)

# 3) Compute convex hulls.
hull1 = cv2.convexHull(pts1)
hull2 = cv2.convexHull(pts2)

# 4) Compute bounding boxes around the convex hulls.
x1, y1, w1, h1 = cv2.boundingRect(hull1)  # For destination face.
x2, y2, w2, h2 = cv2.boundingRect(hull2)  # For source face.

# Clamp these rectangles to lie within each image.
img1_h, img1_w = image1.shape[:2]
img2_h, img2_w = image2.shape[:2]

rect1 = clamp_rect(x1, y1, w1, h1, img1_w, img1_h)
rect2 = clamp_rect(x2, y2, w2, h2, img2_w, img2_h)

if rect1 is None or rect2 is None:
    print("Bounding box sau khi clamp không hợp lệ (0x0). Dừng lại.")
    exit(1)

x1, y1, w1, h1 = rect1
x2, y2, w2, h2 = rect2

# 5) Extract the source face region and resize it to match the destination face size.
face_region2 = image2[y2:y2+h2, x2:x2+w2]
face_region2 = cv2.resize(face_region2, (w1, h1), interpolation=cv2.INTER_LINEAR)

# 6) Create a mask for the cloning process.
# The mask must be the same size as the source patch.
# We create a mask of size (h1, w1) and fill the convex polygon (from destination face)
# with white (255). To do so, adjust the destination convex hull (hull1) relative to (x1, y1).
face_mask = np.zeros((h1, w1), dtype=np.uint8)
# Squeeze to remove the extra dimension and convert to ROI coordinates.
hull1_crop = hull1.squeeze(1) - np.array([x1, y1])
# Clip points so that they lie between 0 and w1-1 (or h1-1).
hull1_crop[:, 0] = np.clip(hull1_crop[:, 0], 0, w1 - 1)
hull1_crop[:, 1] = np.clip(hull1_crop[:, 1], 0, h1 - 1)
hull1_crop = hull1_crop.astype(np.int32)
cv2.fillConvexPoly(face_mask, hull1_crop, 255)

# 7) Determine the center for seamlessClone.
# Here we use the center of the destination bounding rectangle.
center = (x1 + w1 // 2, y1 + h1 // 2)

# 8) (Optional) Debug: Check that the ROI when placing face_region2 lies entirely inside image1.
roi_x = center[0] - w1 // 2
roi_y = center[1] - h1 // 2
roi_x2 = roi_x + w1
roi_y2 = roi_y + h1
print(f"Destination ROI: ({roi_x}, {roi_y}) to ({roi_x2}, {roi_y2})")
print(f"Destination image size: (width={img1_w}, height={img1_h})")
if roi_x < 0 or roi_y < 0 or roi_x2 > img1_w or roi_y2 > img1_h:
    print("Computed ROI nằm ngoài biên của image1. Điều chỉnh center hoặc kích thước khuôn mặt.")
    exit(1)

print(f"Using center: {center} and face_region2 size: {face_region2.shape}")

# 9) Perform seamless cloning.
try:
    output = cv2.seamlessClone(
        face_region2,   # Source patch.
        image1,         # Destination image.
        face_mask,      # Mask (same size as face_region2).
        center,         # Center in destination image.
        cv2.MIXED_CLONE
    )
except cv2.error as e:
    print("OpenCV error khi gọi seamlessClone:", e)
    exit(1)

cv2.imwrite("face_swapped_result1.jpg", output)
print("Ghép mặt thành công, xem ảnh 'face_swapped_result.jpg'")
