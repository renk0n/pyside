import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFileDialog
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


class AdvancedImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Image Processor - v1")
        self.resize(800, 600)
        self.label_image = QLabel("No Image")
        self.label_image.setAlignment(Qt.AlignCenter)

        btn_open = QPushButton("画像を開く")
        btn_open.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(btn_open)
        layout.addWidget(self.label_image)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "画像を開く", "", "Images (*.png *.jpg)")
        if file_path:
            img_array = np.fromfile(file_path, dtype=np.uint8)
            cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.label_image.setPixmap(QPixmap.fromImage(qt_img).scaled(
                self.label_image.size(), Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedImageProcessor()
    window.show()
    sys.exit(app.exec())
