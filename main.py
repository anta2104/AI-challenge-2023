import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QTextEdit, QPushButton, QLabel
from PyQt5.QtGui import QPixmap

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Ứng dụng tạo giao diện')
        self.setGeometry(300, 300, 800, 600)

        # Tạo ô tìm kiếm
        self.search_label = QLabel("Tìm kiếm:")
        self.search_input = QLineEdit()
        self.search_button = QPushButton("Tìm")

        # Tạo khu vực hiển thị hình ảnh
        self.image_label = QLabel()
        self.image_label.setFixedSize(400, 400)

        # Tạo layout
        layout = QVBoxLayout()
        layout.addWidget(self.search_label)
        layout.addWidget(self.search_input)
        layout.addWidget(self.search_button)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        # Xử lý sự kiện nút Tìm
        self.search_button.clicked.connect(self.display_image)

    def display_image(self):
        search_text = self.search_input.text()
        if search_text:
            # TODO: Thêm mã để tạo và hiển thị hình ảnh dựa trên từ khóa tìm kiếm
            # Ví dụ: tải hình ảnh từ API hoặc tạo hình ảnh từ dữ liệu

            # Hiển thị hình ảnh ví dụ
            pixmap = QPixmap('0312.jpg')
            pixmap = pixmap.scaled(400, 400)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MyWidget()
    widget.show()
    sys.exit(app.exec_())