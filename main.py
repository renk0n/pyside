import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                               QHBoxLayout, QWidget, QLabel, QFileDialog)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Image Processor (PySide6)")
        self.resize(800, 600)

        # UI要素の作成
        self.label_image = QLabel("画像を読み込んでください")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setStyleSheet("border: 1px solid black;")

        btn_open = QPushButton("画像を開く")
        btn_gray = QPushButton("グレースケール変換")
        btn_edge = QPushButton("エッジ抽出 (Canny)")
        btn_reset = QPushButton("リセット")

        # レイアウト
        layout_buttons = QVBoxLayout()
        layout_buttons.addWidget(btn_open)
        layout_buttons.addWidget(btn_gray)
        layout_buttons.addWidget(btn_edge)
        layout_buttons.addWidget(btn_reset)
        layout_buttons.addStretch()

        layout_main = QHBoxLayout()
        layout_main.addLayout(layout_buttons, 1)
        layout_main.addWidget(self.label_image, 4)

        container = QWidget()
        container.setLayout(layout_main)
        self.setCentralWidget(container)

        # イベント接続
        btn_open.clicked.connect(self.load_image)
        btn_gray.clicked.connect(self.convert_gray)
        btn_edge.clicked.connect(self.detect_edges)
        btn_reset.clicked.connect(self.reset_image)

        self.original_cv_img = None
        self.current_cv_img = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "画像選択", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            # OpenCVは日本語パスに弱いのでnp.fromfileを使用
            self.original_cv_img = cv2.imdecode(np.fromfile(
                file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.current_cv_img = self.original_cv_img.copy()
            self.display_image(self.current_cv_img)

    def display_image(self, cv_img):
        if cv_img is None:
            return

        # RGB変換
        if len(cv_img.shape) == 3:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h,
                            bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = cv_img.shape
            bytes_per_line = w
            qt_img = QImage(cv_img.data, w, h, bytes_per_line,
                            QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qt_img)
        self.label_image.setPixmap(pixmap.scaled(
            self.label_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def convert_gray(self):
        if self.current_cv_img is not None:
            if len(self.current_cv_img.shape) == 3:
                self.current_cv_img = cv2.cvtColor(
                    self.current_cv_img, cv2.COLOR_BGR2GRAY)
                self.display_image(self.current_cv_img)

    def detect_edges(self):
        if self.current_cv_img is not None:
            if len(self.current_cv_img.shape) == 3:
                gray = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_cv_img
            self.current_cv_img = cv2.Canny(gray, 100, 200)
            self.display_image(self.current_cv_img)

    def reset_image(self):
        if self.original_cv_img is not None:
            self.current_cv_img = self.original_cv_img.copy()
            self.display_image(self.current_cv_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())
