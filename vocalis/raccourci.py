"""Raccourci clavier global : logique pure + écoute du clavier Windows."""

from typing import Callable

MODIFICATEURS = frozenset({"ctrl", "alt", "shift", "win"})
_ALIAS = {"control": "ctrl", "ctl": "ctrl", "maj": "shift", "espace": "space",
          "windows": "win", "cmd": "win", "super": "win", "entree": "enter",
          "return": "enter", "echap": "esc", "escape": "esc"}


def analyser(texte: str) -> frozenset[str]:
    """« Ctrl+Space » → {"ctrl", "space"}. Exige exactement une touche non modificatrice."""
    noms = [_ALIAS.get(p.strip().lower(), p.strip().lower())
            for p in (texte or "").split("+")]
    noms = [n for n in noms if n]
    combinaison = frozenset(noms)
    principales = combinaison - MODIFICATEURS
    if len(principales) != 1:
        raise ValueError(
            "Le raccourci doit contenir exactement une touche en plus des "
            "modificateurs (ex. ctrl+space, f9).")
    return combinaison


class DetecteurRaccourci:
    """Suit les touches enfoncées et déclenche `sur_debut` / `sur_fin`.

    - mode « maintenir » : début à l'appui complet, fin au relâchement ;
    - mode « basculer » : un appui démarre, l'appui suivant arrête.
    """

    def __init__(self, combinaison: frozenset[str], mode: str,
                 sur_debut: Callable[[], None], sur_fin: Callable[[], None]):
        self.combinaison = frozenset(combinaison)
        self.principale = next(iter(self.combinaison - MODIFICATEURS))
        self.mode = mode
        self.sur_debut = sur_debut
        self.sur_fin = sur_fin
        self._enfoncees: set[str] = set()
        self._arme = False      # combinaison actuellement tenue (anti-répétition)
        self._en_cours = False  # dictée démarrée

    def modificateurs_tenus(self) -> bool:
        """Vrai si les modificateurs enfoncés sont exactement ceux du raccourci."""
        return (self._enfoncees & MODIFICATEURS) == (self.combinaison & MODIFICATEURS)

    def touche_enfoncee(self, nom: str) -> None:
        self._enfoncees.add(nom)
        if self._arme or self._enfoncees != self.combinaison:
            return
        self._arme = True
        if self.mode == "basculer" and self._en_cours:
            self._en_cours = False
            self.sur_fin()
        else:
            self._en_cours = True
            self.sur_debut()

    def touche_relachee(self, nom: str) -> None:
        self._enfoncees.discard(nom)
        if nom not in self.combinaison or not self._arme:
            return
        self._arme = False
        if self.mode == "maintenir" and self._en_cours:
            self._en_cours = False
            self.sur_fin()

    def synchroniser(self, modificateurs_reels: set[str]) -> None:
        """Recale les modificateurs sur l'état réel du clavier.

        Un relâchement peut nous échapper (écran de verrouillage, fenêtre
        d'administration) ; sans ce recalage le raccourci resterait bloqué.
        """
        self._enfoncees = (self._enfoncees - MODIFICATEURS) | set(modificateurs_reels)

    def reinitialiser(self) -> None:
        self._enfoncees.clear()
        self._arme = False
        self._en_cours = False


# --- Écoute du clavier (Windows, via pynput) -------------------------------

_VK_FIXES = {
    0x11: "ctrl", 0xA2: "ctrl", 0xA3: "ctrl",
    0x12: "alt", 0xA4: "alt", 0xA5: "alt",
    0x10: "shift", 0xA0: "shift", 0xA1: "shift",
    0x5B: "win", 0x5C: "win",
    0x20: "space", 0x0D: "enter", 0x1B: "esc", 0x09: "tab", 0x08: "backspace",
    0x14: "capslock", 0x2D: "insert", 0x2E: "delete", 0x24: "home", 0x23: "end",
    0x21: "pageup", 0x22: "pagedown", 0x13: "pause", 0x91: "scrolllock",
    0x25: "left", 0x26: "up", 0x27: "right", 0x28: "down",
}
_APPUIS = (0x0100, 0x0104)       # WM_KEYDOWN, WM_SYSKEYDOWN
_RELACHES = (0x0101, 0x0105)     # WM_KEYUP, WM_SYSKEYUP
_INJECTE = 0x10                  # LLKHF_INJECTED


def nom_touche(vk: int) -> str:
    """Code de touche virtuelle Windows → nom utilisé dans les raccourcis."""
    if vk in _VK_FIXES:
        return _VK_FIXES[vk]
    if 0x70 <= vk <= 0x87:
        return f"f{vk - 0x6F}"
    if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
        return chr(vk).lower()
    if 0x60 <= vk <= 0x69:
        return f"num{vk - 0x60}"
    return f"vk{vk}"


def _modificateurs_reels() -> set[str]:
    import ctypes

    etat = ctypes.windll.user32.GetAsyncKeyState
    codes = {"ctrl": (0x11,), "alt": (0x12,), "shift": (0x10,), "win": (0x5B, 0x5C)}
    return {nom for nom, vks in codes.items() if any(etat(vk) & 0x8000 for vk in vks)}


class EcouteurClavier:
    """Branche un `DetecteurRaccourci` sur le clavier système.

    La touche principale du raccourci est « avalée » quand les bons
    modificateurs sont tenus, pour que Ctrl+Espace ne déclenche pas aussi
    l'autocomplétion de l'éditeur dans lequel on dicte.
    """

    def __init__(self, detecteur: DetecteurRaccourci):
        self.detecteur = detecteur
        self._ecouteur = None
        self._avalee = False

    def _filtre(self, msg, data) -> bool:
        if data.flags & _INJECTE:
            return True          # nos propres Ctrl+V simulés ne comptent pas
        nom = nom_touche(data.vkCode)
        d = self.detecteur
        if msg in _APPUIS:
            if nom not in MODIFICATEURS:
                d.synchroniser(_modificateurs_reels())
            avaler = nom == d.principale and d.modificateurs_tenus()
            d.touche_enfoncee(nom)
            if avaler:
                self._avalee = True
                self._ecouteur.suppress_event()
        elif msg in _RELACHES:
            d.touche_relachee(nom)
            if nom == d.principale and self._avalee:
                self._avalee = False
                self._ecouteur.suppress_event()
        return True

    def demarrer(self) -> None:
        from pynput import keyboard

        self.arreter()
        self._ecouteur = keyboard.Listener(win32_event_filter=self._filtre)
        self._ecouteur.start()

    def arreter(self) -> None:
        if self._ecouteur is not None:
            self._ecouteur.stop()
            self._ecouteur = None
        self.detecteur.reinitialiser()
