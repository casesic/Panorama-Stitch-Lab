import sys
import os
import numpy as np
import cv2 as cv

from PyQt6 import QtCore

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
)

from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QMouseEvent,
    QWheelEvent
)

class Image_Viewer_Base(QMainWindow):
    def __init__(self, H_show: int = 960, W_show: int = 1920) -> None:
        super().__init__()

        # 原图与显示图分离：保留原始图用于保存 mask
        self.orig_image = None
        self.image_cv = self.init_image_base()

        self.matrix_global = np.identity(3)
        self.lb_img = QLabel()
        self.H_show = self.image_cv.shape[0]
        self.W_show = self.image_cv.shape[1]
        ratio = self.H_show / self.W_show
        MAX_DIM = 1920  
       # 保证 ratio 定义为原始高除以原始宽： ratio = orig_H / orig_W
        if self.H_show > self.W_show:
            self.H_show = min(self.H_show, MAX_DIM)
            self.W_show = max(1, round(self.H_show / ratio))  # 用 round 并防止为 0
        else:
            self.W_show = min(self.W_show, MAX_DIM)
            self.H_show = max(1, round(self.W_show * ratio))
        self.image_cv = cv.resize(self.image_cv, (self.W_show, self.H_show))
        # 多边形 Mask 模式状态
        self.mask_mode = False
        self.polygon_pts_window = []  # 存储窗口坐标的顶点，用于绘制与保存
        self.polygons_list = []  # 存储已闭合的多边形（每个为窗口坐标列表）
        self.mask_preview_color = (0, 0, 255)  # 红色半透明预览
        self.init_ui_base()
        self.refresh_image()
        

    def init_ui_base(self):
        self.setWindowTitle('Image Viewer')
        self.lb_img.setFixedHeight(self.H_show)
        self.lb_img.setFixedWidth(self.W_show)
        self.setCentralWidget(self.lb_img)

    def init_image_base(self):
        img_path = self.get_image()
        if not img_path:
            raise FileNotFoundError("No image selected")
        img = cv.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {img_path}")
        # 保留原图与原始路径
        self.orig_image = img.copy()
        self.orig_image_path = img_path
        return img

    def refresh_image(self):
        '''
        @param
            * self.matrix_global
            * self.image_cv
        '''
        if self.W_show > 200:
            SCALE_FOR_HIGH_DPI = 1
        else:
            SCALE_FOR_HIGH_DPI = 2

        W_show_scaled = self.W_show * SCALE_FOR_HIGH_DPI
        H_show_scaled = self.H_show * SCALE_FOR_HIGH_DPI

        image_show = cv.warpPerspective(
            self.image_cv,
            self.matrix_global,
            (W_show_scaled, H_show_scaled),
            flags=cv.INTER_NEAREST
        )

        # 如果处于 mask 模式且有顶点，绘制多边形预览（窗口坐标系）
        if getattr(self, 'mask_mode', False):
            overlay = image_show.copy()
            # 绘制已闭合多边形列表
            for poly in getattr(self, 'polygons_list', []):
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32)
                    pts = (pts * SCALE_FOR_HIGH_DPI).reshape((-1, 1, 2))
                    cv.fillPoly(overlay, [pts], self.mask_preview_color)
            # 绘制当前正在绘制的多边形（半透明覆盖）
            if len(self.polygon_pts_window) > 0:
                pts_cur = np.array(self.polygon_pts_window, dtype=np.int32)
                pts_cur = (pts_cur * SCALE_FOR_HIGH_DPI).reshape((-1, 1, 2))
                cv.fillPoly(overlay, [pts_cur], self.mask_preview_color)

            image_show = cv.addWeighted(image_show, 0.7, overlay, 0.3, 0)
        
        # 确保连续内存并传入正确的 bytesPerLine
        arr = np.ascontiguousarray(image_show)
        bytes_per_line = int(arr.strides[0])
        qimg_show = QImage(
            arr.data,
            W_show_scaled,
            H_show_scaled,
            bytes_per_line,
            QImage.Format.Format_BGR888
        )
        qpixm_show = QPixmap.fromImage(qimg_show)
        self.lb_img.setPixmap(qpixm_show)

    def mousePressEvent(self, e:QMouseEvent):
        # 若处于 mask 模式，则用左键添加顶点；其他模式为拖拽平移起点
        btn = e.button()
        pos = np.array([e.pos().x(), e.pos().y()])
        if getattr(self, 'mask_mode', False):
            if btn == QtCore.Qt.MouseButton.LeftButton:
                # 存储窗口坐标（浮点转整型在绘制时处理）
                self.polygon_pts_window.append((pos[0], pos[1]))
                self.refresh_image()
                return
            elif btn == QtCore.Qt.MouseButton.RightButton:
                # 右键结束/闭合多边形（不自动保存）
                # 若顶点>=3，则自动在显示中闭合（refresh 已处理）
                return

        # 非 mask 模式继续作为平移起点
        self.location_start = pos
        return

    def mouseReleaseEvent(self, e:QMouseEvent):
        print('mouseReleaseEvent')
        # 输出当前鼠标的位置和rgb值，通过matrix_global反算回原图位置
        location_present = np.array([e.pos().x(), e.pos().y()])
        matrix_global_inv = np.linalg.inv(self.matrix_global)
        location_in_image = np.dot(matrix_global_inv, np.array([location_present[0], location_present[1], 1])) 
        location_in_image = location_in_image.astype(int)
        # 检查是否出界
        if (location_in_image[0] < 0 or location_in_image[0] >= self.image_cv.shape[1] or
            location_in_image[1] < 0 or location_in_image[1] >= self.image_cv.shape[0]):
            print("Clicked outside the image area.")
        else:
            bgr_value = self.image_cv[location_in_image[1], location_in_image[0]]
            print(f"RGB at window({location_present[0]}, {location_present[1]}) -> image({location_in_image[0]}, {location_in_image[1]}) = {bgr_value}")
        return

    def mouseMoveEvent(self, e:QMouseEvent):
        # 如果处于 mask 模式，不进行平移
        if getattr(self, 'mask_mode', False):
            return
        # 移动鼠标时，图片跟着移动
        location_present = np.array([e.pos().x(), e.pos().y()])
        dx, dy = location_present - self.location_start
        # 直接更新平移分量
        self.matrix_global[0, 2] += dx
        self.matrix_global[1, 2] += dy
        self.location_start = location_present
        self.refresh_image()
        
        
    def wheelEvent(self, e:QWheelEvent):
        location_present = np.array([e.position().x(), e.position().y()])
        delta = e.angleDelta().y()
        # 每 120 单位约 10% 缩放
        factor = 1 + delta / 120 * 0.1

        # 提取当前缩放和平移（假定矩阵为 [s,0,tx; 0,s,ty; 0,0,1]）
        s = float(self.matrix_global[0, 0])
        tx = float(self.matrix_global[0, 2])
        ty = float(self.matrix_global[1, 2])

        # 按指针位置做锚点缩放：tx' = factor*tx + (1-factor)*px
        s_new = s * factor
        tx_new = factor * tx + (1 - factor) * location_present[0]
        ty_new = factor * ty + (1 - factor) * location_present[1]

        self.matrix_global[0, 0] = s_new
        self.matrix_global[1, 1] = s_new
        self.matrix_global[0, 2] = tx_new
        self.matrix_global[1, 2] = ty_new

        self.refresh_image()

    def keyPressEvent(self, e):
        """键盘快捷键：
        M - 切换 Mask 模式
        C - 闭合并预览多边形
        R - 重置多边形
        S - 保存 Mask 到原始图片目录
        """
        key = e.key()
        if key == QtCore.Qt.Key.Key_M:
            self.mask_mode = not getattr(self, 'mask_mode', False)
            print(f"Mask mode: {self.mask_mode}")
            # 进入 mask 模式时清空起点避免误平移
            if self.mask_mode:
                self.polygon_pts_window = []
            self.refresh_image()

        elif key == QtCore.Qt.Key.Key_C:
            # 闭合当前多边形并加入已闭合列表，便于继续画下一个
            if len(self.polygon_pts_window) >= 3:
                self.polygons_list.append(list(self.polygon_pts_window))
                self.polygon_pts_window = []
                print(f"Polygon closed and saved. Total polygons: {len(self.polygons_list)}")
            else:
                print("Not enough points to close polygon (need >=3)")
            self.refresh_image()

        elif key == QtCore.Qt.Key.Key_R:
            # 重置所有多边形
            self.polygon_pts_window = []
            self.polygons_list = []
            print("All polygons reset")
            self.refresh_image()

        elif key == QtCore.Qt.Key.Key_U:
            # 撤销上一个已闭合多边形
            if len(self.polygons_list) > 0:
                self.polygons_list.pop()
                print(f"Undo last polygon. Remaining: {len(self.polygons_list)}")
            else:
                print("No polygon to undo")
            self.refresh_image()

        elif key == QtCore.Qt.Key.Key_S:
            # 保存 Mask 至原图同目录，文件名: <origname>_mask.png
            if not hasattr(self, 'orig_image') or self.orig_image is None:
                print("No original image to save mask for")
                return

            # 保存所有已闭合多边形及当前未闭合多边形（如果有）到一个 mask
            all_polys = []
            # 先加入已闭合的
            for poly in getattr(self, 'polygons_list', []):
                if len(poly) >= 3:
                    all_polys.append(poly)
            # 再加入当前绘制的
            if len(self.polygon_pts_window) >= 3:
                all_polys.append(list(self.polygon_pts_window))

            if len(all_polys) == 0:
                print("No polygons to save")
                return

            # 将窗口坐标列表转换到原图坐标并生成 mask
            mat_inv = np.linalg.inv(self.matrix_global)
            img_cv_h, img_cv_w = self.image_cv.shape[:2]
            orig_h, orig_w = self.orig_image.shape[:2]

            contours_orig = []
            for poly in all_polys:
                pts_img = []
                for (wx, wy) in poly:
                    p = np.dot(mat_inv, np.array([wx, wy, 1.0]))
                    px = p[0] / p[2]
                    py = p[1] / p[2]
                    pts_img.append((px, py))
                sx = orig_w / float(img_cv_w)
                sy = orig_h / float(img_cv_h)
                pts_orig = np.array([[int(x * sx), int(y * sy)] for (x, y) in pts_img], dtype=np.int32)
                contours_orig.append(pts_orig.reshape((-1,1,2)))

            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv.fillPoly(mask, contours_orig, 255)

            # 保存到与原图同目录，自动命名为 <orig_basename>_mask.png
            if hasattr(self, 'orig_image_path') and self.orig_image_path:
                orig_dir = os.path.dirname(self.orig_image_path)
                orig_base = os.path.splitext(os.path.basename(self.orig_image_path))[0]
                save_path = os.path.join(orig_dir, f"{orig_base}_ground_mask.png")
                cv.imwrite(save_path, mask)
                print(f"Mask saved: {save_path}")
            else:
                # 兜底：弹出保存对话
                save_path, _ = QFileDialog.getSaveFileName(self, 'Save Mask', 'mask.png', '*.png')
                if save_path:
                    cv.imwrite(save_path, mask)
                    print(f"Mask saved: {save_path}")
                else:
                    print("Save cancelled")
            return

        else:
            super().keyPressEvent(e)
        
    def get_image(self):
        img_name, ____ = QFileDialog.getOpenFileName(
            self, 
            'Open Image File', 
            '*.jpg;;*.png;;*.jpeg'
        )
        return img_name
    

if __name__=='__main__':
    app = QApplication(sys.argv)
    im = Image_Viewer_Base()
    im.show()
    sys.exit(app.exec())
    
    
'''
RGB at window(320, 501)  cam2 (0,0,1) 对应60度 0-120   (0,0)
 
RGB at window(1588, 442)  cam4对应296度 240-360  236(-124)  (-122,11)

RGB at window(949, 512)   cam6对应183度 120-240  123  (118,-2)
'''