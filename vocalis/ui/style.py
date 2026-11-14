"""Couleurs, feuille de style et icône de l'app."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

FOND = "#14161b"
SURFACE = "#1d2027"
BORD = "#2c313b"
TEXTE = "#eef0f4"
TEXTE_DOUX = "#9aa3b2"
ACCENT = "#6c8cff"
ROUGE = "#ff5d52"

FEUILLE = f"""
QWidget {{ background: {FOND}; color: {TEXTE}; font-family: 'Segoe UI'; font-size: 10pt; }}
QTabWidget::pane {{ border: 0; }}
QTabBar::tab {{ background: transparent; color: {TEXTE_DOUX}; padding: 10px 18px;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXTE}; border-bottom: 2px solid {ACCENT}; }}
QLineEdit, QComboBox, QPlainTextEdit, QListWidget {{
    background: {SURFACE}; border: 1px solid {BORD}; border-radius: 6px; padding: 6px 8px;
    selection-background-color: {ACCENT}; }}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; border: 1px solid {BORD};
                              selection-background-color: {ACCENT}; }}
QComboBox::drop-down {{ border: 0; width: 26px; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {BORD};
                       border-radius: 4px; background: {SURFACE}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QListWidget::item {{ padding: 8px 6px; border-bottom: 1px solid {BORD}; }}
QListWidget::item:selected {{ background: #2a3350; color: {TEXTE}; }}
QPushButton {{ background: {SURFACE}; border: 1px solid {BORD}; border-radius: 6px;
              padding: 7px 14px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {TEXTE_DOUX}; }}
QPushButton#principal {{ background: {ACCENT}; border-color: {ACCENT}; color: white;
                        font-weight: 600; }}
QLabel#doux {{ color: {TEXTE_DOUX}; }}
QLabel#erreur {{ color: {ROUGE}; }}
QLabel#titre {{ font-size: 15pt; font-weight: 600; }}
"""


def icone(couleur: str = ACCENT) -> QIcon:
    """Icône de l'app dessinée à la volée : un micro stylisé dans un disque."""
    image = QPixmap(64, 64)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(couleur))
    p.drawEllipse(QRectF(2, 2, 60, 60))
    p.setBrush(QColor("white"))
    p.drawRoundedRect(QRectF(25, 12, 14, 26), 7, 7)          # capsule du micro
    stylo = p.pen()
    stylo.setStyle(Qt.PenStyle.SolidLine)
    stylo.setColor(QColor("white"))
    stylo.setWidthF(3.5)
    stylo.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(stylo)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(19, 20, 26, 26), 180 * 16, 180 * 16)    # arceau
    p.drawLine(32, 46, 32, 52)                               # pied
    p.end()
    return QIcon(image)
