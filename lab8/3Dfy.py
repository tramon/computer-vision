import numpy as np
import cv2
import open3d as o3d

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def reconstruct_and_display(imgL, imgR, min_disp, num_disp, window_size, ply_output_path="result.ply"):
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=120,
        speckleRange=132
    )

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    h, w = imgL.shape[:2]
    f = 1.2 * w
    Q = np.float32([
        [1, 0, 0, -0.5 * w],
        [0, -1, 0, 0.5 * h],
        [0, 0, 0, -f],
        [0, 0, 1, 0]
    ])

    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]

    write_ply(ply_output_path, out_points, out_colors)
    print(f'[INFO] Point cloud saved as: {ply_output_path}')

    cv2.imshow('Left Image', imgL)
    cv2.imshow('Disparity', (disp - min_disp) / num_disp)

    print("[INFO] Rendering point cloud...")
    pcd = o3d.io.read_point_cloud(ply_output_path)
    o3d.visualization.draw_geometries([pcd], width=500, height=500, left=20, top=20)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    left_image = cv2.pyrDown(cv2.imread('left.jpg'))
    right_image = cv2.pyrDown(cv2.imread('right.jpg'))

    window_size = 9
    min_disp = 32
    num_disp = 64 - min_disp

    reconstruct_and_display(
        imgL=left_image,
        imgR=right_image,
        min_disp=min_disp,
        num_disp=num_disp,
        window_size=window_size,
        ply_output_path="result.ply"
    )
