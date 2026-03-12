from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QLinearGradient,
    QFont,
    QPainterPath
)


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedSize(600, 350)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.progress = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)  # ~60 FPS

    def animate(self):
        self.progress += 1
        if self.progress > self.width():
            self.progress = self.width()

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        # ---------- Rounded background ----------
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 28, 28)

        painter.setClipPath(path)

        # ---------- Background gradient ----------
        bg = QLinearGradient(0, 0, 0, rect.height())
        bg.setColorAt(0, QColor("#021B44"))
        bg.setColorAt(1, QColor("#001A3A"))

        painter.fillRect(rect, bg)

        # ---------- Curved overlay shape ----------
        shape = QPainterPath()
        shape.moveTo(-100, rect.height() * 0.55)
        shape.cubicTo(
            rect.width() * 0.3,
            rect.height() * 0.35,
            rect.width() * 0.7,
            rect.height() * 0.85,
            rect.width() + 100,
            rect.height() * 0.55
        )
        shape.lineTo(rect.width() + 100, rect.height())
        shape.lineTo(-100, rect.height())
        shape.closeSubpath()

        painter.fillPath(shape, QColor("#0A2A66"))

        # ---------- Title ----------
        painter.setPen(QColor("white"))

        title_font = QFont("Segoe UI", 40, QFont.Weight.Bold)
        painter.setFont(title_font)

        title_text = "◆  VoxelForge"

        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter,
            title_text
        )

        # ---------- Footer ----------
        footer_font = QFont("Segoe UI", 9)
        painter.setFont(footer_font)
        painter.setPen(QColor("#B0B8C5"))

        footer_rect = rect.adjusted(0, 0, 0, -40)

        painter.drawText(
            footer_rect,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            "© 2026 VoxelForge — All rights reserved."
        )

        # ---------- Gradient loading bar ----------
        bar_height = 14

        bar_rect = QRectF(
            0,
            rect.height() - bar_height,
            self.progress,
            bar_height
        )

        gradient = QLinearGradient(
            0,
            rect.height(),
            rect.width(),
            rect.height()
        )

        gradient.setColorAt(0, QColor("#6A00FF"))
        gradient.setColorAt(0.5, QColor("#3A7BFF"))
        gradient.setColorAt(1, QColor("#00CFFF"))

        painter.fillRect(bar_rect, gradient)