"""Pastille flottante affichée pendant la dictée (ne prend jamais le focus)."""

from collections import deque

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

from ..controleur import ECOUTE, REPOS, TRANSCRIPTION
from .style import ACCENT, ROUGE, TEXTE

LARGEUR, HAUTEUR, NB_BARRES = 250, 46, 18


class Pastille(QWidget):
    def __init__(self):
        super().__init__(None,
                         Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool
                         | Qt.WindowType.WindowDoesNotAcceptFocus
                         | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(LARGEUR, HAUTEUR)
        self.etat = REPOS
        self.texte = ""
        self._niveaux = deque([0.0] * NB_BARRES, maxlen=NB_BARRES)
        self._phase = 0
        self._animation = QTimer(self, interval=120)
        self._animation.timeout.connect(self._animer)
        self._masquage = QTimer(self, singleShot=True)
        self._masquage.timeout.connect(self._fin_message)

    # --- API -----------------------------------------------------------------

    def afficher_etat(self, etat: str) -> None:
        self.etat = etat
        self._masquage.stop()
        if etat == REPOS:
            self._animation.stop()
            self.hide()
            return
        self.texte = "Écoute…" if etat == ECOUTE else "Transcription…"
        if etat == ECOUTE:
            self._niveaux.extend([0.0] * NB_BARRES)
        self._animation.start()
        self._montrer()

    def afficher_niveau(self, rms: float) -> None:
        # Échelle perceptive : une voix normale (RMS ≈ 0.05) remplit ~70 % de la barre.
        self._niveaux.append(min(1.0, (max(rms, 0.0) * 12) ** 0.6))
        if self.etat == ECOUTE:
            self.update()

    def afficher_message(self, texte: str, duree_ms: int = 2500) -> None:
        if self.etat != REPOS:
            return                      # ne pas masquer l'écoute en cours
        self.texte = texte
        self._montrer()
        self._masquage.start(duree_ms)

    # --- Interne -------------------------------------------------------------

    def _fin_message(self) -> None:
        if self.etat == REPOS:
            self.hide()

    def _animer(self) -> None:
        self._phase = (self._phase + 1) % 3
        self.update()

    def _montrer(self) -> None:
        ecran = QGuiApplication.primaryScreen()
        if ecran is not None:
            zone = ecran.availableGeometry()
            self.move(zone.center().x() - LARGEUR // 2, zone.bottom() - HAUTEUR - 48)
        self.show()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 22, 27, 238))
        p.drawRoundedRect(QRectF(0, 0, LARGEUR, HAUTEUR), HAUTEUR / 2, HAUTEUR / 2)

        x = 18.0
        if self.etat == ECOUTE:
            p.setBrush(QColor(ROUGE))
            p.drawEllipse(QRectF(x, HAUTEUR / 2 - 5, 10, 10))
            x += 20
            p.setBrush(QColor(TEXTE))
            for niveau in self._niveaux:
                h = 3 + niveau * 22
                p.drawRoundedRect(QRectF(x, (HAUTEUR - h) / 2, 3, h), 1.5, 1.5)
                x += 5
            x += 8
        elif self.etat == TRANSCRIPTION:
            for i in range(3):
                couleur = QColor(ACCENT)
                couleur.setAlpha(255 if i == self._phase else 90)
                p.setBrush(couleur)
                p.drawEllipse(QRectF(x + i * 12, HAUTEUR / 2 - 4, 8, 8))
            x += 46

        p.setPen(QColor(TEXTE))
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(QRectF(x, 0, LARGEUR - x - 14, HAUTEUR),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   p.fontMetrics().elidedText(self.texte, Qt.TextElideMode.ElideRight,
                                              int(LARGEUR - x - 14)))
        p.end()
