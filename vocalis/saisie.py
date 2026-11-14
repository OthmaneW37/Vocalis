"""Saisie du texte dicté dans la fenêtre active, via le presse-papiers."""

from typing import Callable

from PySide6.QtCore import QMimeData, QTimer


def _copie(mime: QMimeData | None) -> QMimeData | None:
    """Copie profonde du contenu du presse-papiers (texte, image, fichiers…)."""
    if mime is None or not mime.formats():
        return None
    copie = QMimeData()
    for fmt in mime.formats():
        copie.setData(fmt, mime.data(fmt))
    if mime.hasImage():
        copie.setImageData(mime.imageData())
    return copie


def envoyer_ctrl_v() -> None:
    from pynput.keyboard import Controller, Key, KeyCode

    clavier = Controller()
    touche_v = KeyCode.from_vk(0x56)
    with clavier.pressed(Key.ctrl):
        clavier.press(touche_v)
        clavier.release(touche_v)


def coller_texte(texte: str, presse_papiers,
                 envoyer_coller: Callable[[], None] = envoyer_ctrl_v,
                 delai_ms: int = 150) -> None:
    """Pose `texte` dans le presse-papiers, simule Ctrl+V, puis remet l'ancien contenu.

    La restauration attend `delai_ms` : l'application cible lit le presse-papiers
    un court instant après avoir reçu Ctrl+V.
    """
    if not texte:
        return
    ancien = _copie(presse_papiers.mimeData())
    presse_papiers.setText(texte)
    envoyer_coller()

    def restaurer():
        if presse_papiers.text() != texte:
            return   # l'utilisateur a copié autre chose entre-temps : on n'y touche pas
        if ancien is None:
            presse_papiers.clear()
        else:
            presse_papiers.setMimeData(ancien)

    QTimer.singleShot(delai_ms, restaurer)
