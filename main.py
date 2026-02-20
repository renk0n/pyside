import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                               QHBoxLayout, QWidget, QLabel, QFileDialog, QScrollArea,
                               QSlider, QGroupBox, QMessageBox, QStatusBar)
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt


class AdvancedImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced CV Image Processor - 知能情報EX課題")
        self.resize(1000, 700)

        # 画像データ保持用
        self.original_cv_img = None
        self.current_cv_img = None
        self.history_cv_img = None  # 1つ前の状態を保持(簡易Undo)

        self.init_ui()

    def init_ui(self):
        # --- メニューバー ---
        menubar = self.menuBar()
        file_menu = menubar.addMenu('ファイル')

        open_action = QAction('画像を開く', self)
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)

        save_action = QAction('画像を保存', self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        # --- ステータスバー ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("画像を読み込んでください")

        # --- メインレイアウト ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- 左側：コントロールパネル ---
        control_panel = QVBoxLayout()

        # 1. 基本フィルタグループ
        group_basic = QGroupBox("基本フィルタ")
        vbox_basic = QVBoxLayout()

        btn_gray = QPushButton("グレースケール変換")
        btn_gray.clicked.connect(self.convert_gray)
        btn_blur = QPushButton("ガウシアンぼかし")
        btn_blur.clicked.connect(self.apply_blur)
        btn_thresh = QPushButton("大津の二値化")
        btn_thresh.clicked.connect(self.apply_otsu_threshold)

        vbox_basic.addWidget(btn_gray)
        vbox_basic.addWidget(btn_blur)
        vbox_basic.addWidget(btn_thresh)
        group_basic.setLayout(vbox_basic)

        # 2. 高度な解析グループ (エッジ抽出 + スライダー)
        group_advanced = QGroupBox("エッジ抽出 (Canny)")
        vbox_adv = QVBoxLayout()

        self.label_slider = QLabel("閾値: 100")
        self.slider_canny = QSlider(Qt.Horizontal)
        self.slider_canny.setRange(50, 200)
        self.slider_canny.setValue(100)
        self.slider_canny.valueChanged.connect(self.update_slider_label)

        btn_edge = QPushButton("エッジ抽出を適用")
        btn_edge.clicked.connect(self.apply_canny)

        vbox_adv.addWidget(self.label_slider)
        vbox_adv.addWidget(self.slider_canny)
        vbox_adv.addWidget(btn_edge)
        group_advanced.setLayout(vbox_adv)

        # 3. AI / CV グループ
        group_cv = QGroupBox("画像解析 (CV)")
        vbox_cv = QVBoxLayout()
        btn_face = QPushButton("顔認識 (Haar Cascade)")
        btn_face.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;")
        btn_face.clicked.connect(self.detect_faces)
        vbox_cv.addWidget(btn_face)
        group_cv.setLayout(vbox_cv)

        # 4. 操作グループ
        group_ops = QGroupBox("操作")
        vbox_ops = QVBoxLayout()
        btn_undo = QPushButton("1つ戻る (Undo)")
        btn_undo.clicked.connect(self.undo_action)
        btn_reset = QPushButton("初期状態にリセット")
        btn_reset.clicked.connect(self.reset_image)
        vbox_ops.addWidget(btn_undo)
        vbox_ops.addWidget(btn_reset)
        group_ops.setLayout(vbox_ops)

        # コントロールパネルにグループを追加
        control_panel.addWidget(group_basic)
        control_panel.addWidget(group_advanced)
        control_panel.addWidget(group_cv)
        control_panel.addWidget(group_ops)
        control_panel.addStretch()

        # --- 右側：画像表示エリア (スクロール対応) ---
        self.label_image = QLabel("No Image")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setStyleSheet("background-color: #e0e0e0;")

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.label_image)
        scroll_area.setWidgetResizable(True)

        # レイアウト統合 (左:右 = 1:4 の比率)
        main_layout.addLayout(control_panel, 1)
        main_layout.addWidget(scroll_area, 4)

    # --- 以下、機能実装 ---
    def save_history(self):
        """処理前に現在の画像を履歴に保存"""
        if self.current_cv_img is not None:
            self.history_cv_img = self.current_cv_img.copy()

    def update_slider_label(self, value):
        self.label_slider.setText(f"閾値: {value}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "画像を開く", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # OpenCVで日本語パス対応読み込み
            img_array = np.fromfile(file_path, dtype=np.uint8)
            self.original_cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            self.current_cv_img = self.original_cv_img.copy()
            self.history_cv_img = None
            self.display_image(self.current_cv_img)
            h, w = self.current_cv_img.shape[:2]
            self.statusBar.showMessage(f"画像を読み込みました: {w} x {h} pixels")

    def save_image(self):
        if self.current_cv_img is None:
            QMessageBox.warning(self, "警告", "保存する画像がありません。")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "画像を保存", "processed_image.jpg", "JPEG (*.jpg);;PNG (*.png)")
        if file_path:
            # 日本語パス対応保存
            ext = file_path.split('.')[-1]
            result, encimg = cv2.imencode(f'.{ext}', self.current_cv_img)
            if result:
                with open(file_path, mode='w+b') as f:
                    encimg.tofile(f)
                self.statusBar.showMessage(f"保存しました: {file_path}")

    def display_image(self, cv_img):
        if cv_img is None:
            return

        if len(cv_img.shape) == 3:  # カラー画像
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h,
                            bytes_per_line, QImage.Format_RGB888)
        else:  # グレースケール画像
            h, w = cv_img.shape
            bytes_per_line = w
            qt_img = QImage(cv_img.data, w, h, bytes_per_line,
                            QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qt_img)
        # 画面サイズに合わせて縮小（拡大はしない）
        self.label_image.setPixmap(pixmap.scaled(
            self.label_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def convert_gray(self):
        if self.current_cv_img is not None and len(self.current_cv_img.shape) == 3:
            self.save_history()
            self.current_cv_img = cv2.cvtColor(
                self.current_cv_img, cv2.COLOR_BGR2GRAY)
            self.display_image(self.current_cv_img)
            self.statusBar.showMessage("グレースケールに変換しました")

    def apply_blur(self):
        if self.current_cv_img is not None:
            self.save_history()
            self.current_cv_img = cv2.GaussianBlur(
                self.current_cv_img, (15, 15), 0)
            self.display_image(self.current_cv_img)
            self.statusBar.showMessage("ガウシアンぼかしを適用しました")

    def apply_otsu_threshold(self):
        if self.current_cv_img is not None:
            self.save_history()
            gray = self.current_cv_img
            if len(self.current_cv_img.shape) == 3:
                gray = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2GRAY)

            # 大津の二値化
            _, self.current_cv_img = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.display_image(self.current_cv_img)
            self.statusBar.showMessage("大津の二値化を適用しました")

    def apply_canny(self):
        if self.current_cv_img is not None:
            self.save_history()
            gray = self.current_cv_img
            if len(self.current_cv_img.shape) == 3:
                gray = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2GRAY)

            threshold2 = self.slider_canny.value()
            threshold1 = threshold2 // 2
            self.current_cv_img = cv2.Canny(gray, threshold1, threshold2)
            self.display_image(self.current_cv_img)
            self.statusBar.showMessage(
                f"エッジ抽出を適用しました (閾値: {threshold1}, {threshold2})")

    def detect_faces(self):
        """OpenCVのカスケード分類器を用いた顔認識"""
        if self.current_cv_img is None:
            return
        self.save_history()

        # 学習済みモデルの読み込み (OpenCVに同梱されているものを使用)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        gray = self.current_cv_img
        if len(self.current_cv_img.shape) == 3:
            gray = cv2.cvtColor(self.current_cv_img, cv2.COLOR_BGR2GRAY)

        # 顔の検出
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 検出結果を描画（カラー画像に戻して赤い四角を描く）
        if len(self.current_cv_img.shape) == 2:
            self.current_cv_img = cv2.cvtColor(
                self.current_cv_img, cv2.COLOR_GRAY2BGR)

        for (x, y, w, h) in faces:
            cv2.rectangle(self.current_cv_img, (x, y),
                          (x+w, y+h), (0, 0, 255), 3)

        self.display_image(self.current_cv_img)
        self.statusBar.showMessage(f"顔認識完了: {len(faces)} 人の顔を検出しました")

    def undo_action(self):
        if self.history_cv_img is not None:
            self.current_cv_img = self.history_cv_img.copy()
            self.history_cv_img = None  # 1回だけ戻れる仕様
            self.display_image(self.current_cv_img)
            self.statusBar.showMessage("1つ前の状態に戻しました")

    def reset_image(self):
        if self.original_cv_img is not None:
            self.save_history()
            self.current_cv_img = self.original_cv_img.copy()
            self.display_image(self.current_cv_img)
            self.statusBar.showMessage("画像を初期状態にリセットしました")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdvancedImageProcessor()
    window.show()
    sys.exit(app.exec())
