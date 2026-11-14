"""Lancement de Vocalis à l'ouverture de session Windows (clé Run de l'utilisateur)."""

import sys
from pathlib import Path

CLE = r"Software\Microsoft\Windows\CurrentVersion\Run"
NOM = "Vocalis"


def commande() -> str:
    """Ligne de commande qui relance l'app sans console."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    racine = Path(__file__).resolve().parent.parent
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    return f'"{pythonw}" "{racine / "lancer.pyw"}"'


def est_actif(registre=None) -> bool:
    registre = registre or _winreg()
    try:
        with registre.OpenKey(registre.HKEY_CURRENT_USER, CLE) as cle:
            registre.QueryValueEx(cle, NOM)
        return True
    except OSError:
        return False


def activer(actif: bool, registre=None) -> None:
    registre = registre or _winreg()
    with registre.OpenKey(registre.HKEY_CURRENT_USER, CLE, 0,
                          registre.KEY_SET_VALUE) as cle:
        if actif:
            registre.SetValueEx(cle, NOM, 0, registre.REG_SZ, commande())
        else:
            try:
                registre.DeleteValue(cle, NOM)
            except OSError:
                pass        # déjà absent


def _winreg():
    import winreg
    return winreg
