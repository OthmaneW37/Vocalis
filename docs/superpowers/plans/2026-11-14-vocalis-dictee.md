# Vocalis — dictée à deux moteurs : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformer la démo Gradio en app Windows de dictée (raccourci global → texte collé dans la fenêtre active) avec un moteur Whisper et un moteur entraîné de zéro.

**Architecture:** Un paquet `vocalis/` (app PySide6 dans la zone de notification) pilote des moteurs interchangeables derrière une interface unique `Moteur.transcrire`. Un paquet `modele/` contient le modèle maison (Conformer + CTC), son tokenizer BPE, son entraînement et son évaluation. `stt/` reste la base commune (audio, log-Mel, WER, SRT).

**Tech Stack:** Python 3.12, PySide6, faster-whisper (CTranslate2, CPU int8), sounddevice, pynput, SQLite, PyTorch (partie B uniquement), pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-vocalis-dictee-design.md`

## Global Constraints

- Windows 11 uniquement ; tout fonctionne hors ligne une fois les modèles présents.
- Audio interne : mono, 16 kHz, `float32` dans [-1, 1] (`stt.audio.SR = 16000`).
- Noms de fonctions et de modules en français, comme le code existant.
- Le modèle maison n'utilise aucun poids pré-entraîné.
- Le WER du modèle maison est affiché tel que mesuré, jamais arrondi à son avantage.
- Le presse-papiers de l'utilisateur est restauré après chaque collage.
- Données de l'app dans `%APPDATA%\Vocalis` ; la variable `VOCALIS_DOSSIER` le remplace (tests).
- L'app (partie A) ne doit pas importer PyTorch : `torch` n'est chargé que si le moteur Maison est choisi.
- Travail sur la branche `feat/app-dictee`, commits locaux, aucun push.
- Tests : `.venv\Scripts\python -m pytest`.

## Review Focus

1. Enregistrement vide ou silencieux (appui bref sur le raccourci) → aucun texte collé, pas de texte inventé. *(Tâches 3, 6)*
2. Aucun micro, ou micro débranché en cours → message clair dans la pastille, l'app reste vivante. *(Tâches 3, 6)*
3. Raccourci pressé pendant qu'une transcription est en cours → ignoré, pas de double enregistrement. *(Tâche 6)*
4. Presse-papiers contenant autre chose que du texte (image, fichiers) → restauré à l'identique. *(Tâche 5)*
5. Moteur Maison choisi sans checkpoint entraîné → repli sur le moteur Pro avec un message, pas de plantage. *(Tâche 2)*

---

## Structure des fichiers

```
stt/resultat.py          Resultat + vers_srt (sans dépendance lourde)
vocalis/chemins.py       dossier de données
vocalis/config.py        Reglages (JSON)
vocalis/moteurs/base.py  interface Moteur
vocalis/moteurs/pro.py   faster-whisper
vocalis/moteurs/maison.py  checkpoint maison (partie B)
vocalis/moteurs/__init__.py  creer_moteur
vocalis/vad.py           détection de voix par énergie
vocalis/micro.py         capture micro
vocalis/raccourci.py     raccourci global
vocalis/saisie.py        collage dans la fenêtre active
vocalis/historique.py    SQLite
vocalis/controleur.py    machine à états de la dictée
vocalis/demarrage.py     lancement avec Windows
vocalis/ui/              pastille, zone de notification, réglages, historique, fichiers
vocalis/__main__.py      python -m vocalis
modele/tokenizer.py      BPE écrit à la main
modele/features.py       log-Mel en PyTorch (parité avec stt.features)
modele/conformer.py      architecture
modele/decodage.py       décodage CTC glouton
modele/donnees.py        jeu de données + SpecAugment
modele/preparer.py       téléchargement / manifeste Common Voice
modele/entrainer.py      boucle d'entraînement
modele/evaluer.py        WER / CER, comparaison avec Whisper
tests/                   un fichier de test par module
```

---

# Partie A — l'application et le moteur Pro

### Task 1 : socle (environnement, `Resultat`, chemins, réglages)

**Files:** Create `requirements-app.txt`, `stt/resultat.py`, `vocalis/__init__.py`, `vocalis/chemins.py`, `vocalis/config.py`, `tests/test_resultat.py`, `tests/test_config.py` · Modify `stt/transcribe.py` (importe `Resultat`, `vers_srt` depuis `stt.resultat`), `.gitignore`

**Interfaces — Produces:**
- `stt.resultat.Resultat(texte, segments=[(debut, fin, texte)], langue="", probas_langues=[], duree_audio=0.0, temps_calcul=0.0)` avec la propriété `facteur_temps_reel`
- `stt.resultat.vers_srt(segments, duree_audio) -> str`
- `vocalis.chemins.dossier_donnees() -> Path` (créé si absent)
- `vocalis.config.Reglages` : `raccourci="ctrl+space"`, `mode="maintenir"` (`"maintenir"`|`"basculer"`), `moteur="pro"` (`"pro"`|`"maison"`), `modele_pro="small"`, `langue="fr"` (`None` = auto), `micro=None`, `coller=True`, `demarrage_auto=False`
- `Reglages.charger() -> Reglages`, `Reglages.enregistrer() -> None`

- [ ] Créer `.venv` (Python 3.12), `requirements-app.txt` : `PySide6`, `faster-whisper`, `sounddevice`, `pynput`, `numpy`, `soundfile`, `librosa`, `jiwer`, `pytest` ; installer.
- [ ] Tests : `vers_srt([(0.0, 1.5, "a")], 2.0)` contient `00:00:00,000 --> 00:00:01,500` ; `Reglages.charger()` sans fichier rend les valeurs par défaut ; enregistrer puis charger rend les mêmes valeurs ; un JSON corrompu ou avec une clé inconnue rend les défauts sans lever.
- [ ] Vérifier l'échec, implémenter, vérifier le succès.
- [ ] Commit `feat: socle de l'app (resultat, chemins, réglages)`.

### Task 2 : moteurs

**Files:** Create `vocalis/moteurs/{__init__,base,pro,maison}.py`, `tests/test_moteurs.py`

**Interfaces — Produces:**
- `base.Moteur` (Protocol) : `nom: str`, `transcrire(audio: np.ndarray, langue: str | None) -> Resultat`
- `pro.MoteurPro(taille: str = "small")` — `WhisperModel(taille, device="cpu", compute_type="int8")`, `vad_filter=True`, `beam_size=5`
- `maison.MoteurMaison(checkpoint: Path)` — lève `ModeleMaisonAbsent` si le fichier manque (corps complété en tâche 15)
- `creer_moteur(reglages) -> tuple[Moteur, str | None]` : le second élément est un avertissement quand il y a eu repli sur Pro
- `maison.CHECKPOINT_DEFAUT = dossier_donnees() / "modele_maison.pt"`

- [ ] Tests : `MoteurPro("base").transcrire(charger_audio("samples/exemple_fr.wav"), "fr")` a un WER < 15 % contre `references.json` ; 2 s de zéros → `texte == ""` ; `creer_moteur(Reglages(moteur="maison"))` sans checkpoint rend un `MoteurPro` et un avertissement non vide.
- [ ] Vérifier l'échec, implémenter, vérifier le succès.
- [ ] Commit `feat(moteurs): interface commune et moteur faster-whisper`.

### Task 3 : capture micro et détection de voix

**Files:** Create `vocalis/vad.py`, `vocalis/micro.py`, `tests/test_vad.py`, `tests/test_micro.py`

**Interfaces — Produces:**
- `vad.rogner_silences(audio, sr=16000, marge_s=0.2) -> np.ndarray` — trames de 30 ms, seuil = max(4 × plancher de bruit (10ᵉ centile des RMS), 0.005) ; rend un tableau vide s'il n'y a pas de voix
- `micro.lister_micros() -> list[tuple[int, str]]`
- `micro.Enregistreur(peripherique=None, sur_niveau: Callable[[float], None] | None = None)` : `demarrer()`, `arreter() -> np.ndarray`, lève `MicroIndisponible(message)` ; le flux `sounddevice` est injecté par un paramètre `fabrique_flux` pour les tests

- [ ] Tests VAD : silence + 1 s de sinus 220 Hz à 0.3 + silence → durée rendue entre 1.0 et 1.6 s ; zéros → vide ; bruit blanc à 0.001 → vide ; tableau vide en entrée → vide.
- [ ] Tests micro (faux flux) : les blocs poussés sont concaténés dans l'ordre ; `arreter()` sans `demarrer()` rend un tableau vide ; une fabrique qui lève `sounddevice.PortAudioError` donne `MicroIndisponible`.
- [ ] Vérifier l'échec, implémenter, vérifier le succès. Commit `feat: capture micro et détection de voix`.

### Task 4 : raccourci global

**Files:** Create `vocalis/raccourci.py`, `tests/test_raccourci.py`

**Interfaces — Produces:**
- `analyser(texte: str) -> frozenset[str]` — `"Ctrl+Space"` → `{"ctrl", "space"}` ; lève `ValueError` si vide ou composé uniquement de modificateurs
- `DetecteurRaccourci(combinaison, mode, sur_debut, sur_fin)` : `touche_enfoncee(nom)`, `touche_relachee(nom)` — logique pure
- `EcouteurClavier(detecteur)` : `demarrer()`, `arreter()` — adaptateur `pynput`, qui avale la touche principale quand la combinaison est active

- [ ] Tests : mode `maintenir` → `sur_debut` à l'appui complet, `sur_fin` au relâchement de n'importe quelle touche de la combinaison ; la répétition automatique du clavier (plusieurs `touche_enfoncee` de suite) ne déclenche qu'un seul `sur_debut` ; mode `basculer` → 1ᵉʳ appui début, 2ᵉ appui fin ; une touche en trop (`ctrl+shift+space` pour `ctrl+space`) ne déclenche rien.
- [ ] Vérifier l'échec, implémenter, vérifier le succès. Commit `feat: raccourci global`.

### Task 5 : collage et historique

**Files:** Create `vocalis/saisie.py`, `vocalis/historique.py`, `tests/test_saisie.py`, `tests/test_historique.py`

**Interfaces — Produces:**
- `saisie.coller_texte(texte: str, presse_papiers, envoyer_coller: Callable[[], None], delai_ms: int = 150) -> None` — sauvegarde tout le `QMimeData` courant (tous formats), pose le texte, envoie Ctrl+V, restaure après `delai_ms`
- `saisie.envoyer_ctrl_v() -> None` (pynput)
- `historique.Historique(chemin=None)` : `ajouter(texte, moteur, duree_audio, temps_calcul) -> int`, `lister(limite=200) -> list[Entree]`, `supprimer(id)`, `vider()` ; `Entree(id, date, texte, moteur, duree_audio, temps_calcul)`

- [ ] Tests saisie (vrai `QClipboard`, `QT_QPA_PLATFORM=offscreen`) : après collage et délai, le texte d'origine est revenu ; une image dans le presse-papiers est restaurée comme image ; `envoyer_coller` est appelé une fois, pendant que le presse-papiers contient le texte dicté ; un texte vide ne touche à rien.
- [ ] Tests historique : ajout puis `lister()` rend l'entrée la plus récente en premier ; accents et emojis conservés ; `supprimer` et `vider`.
- [ ] Vérifier l'échec, implémenter, vérifier le succès. Commit `feat: collage automatique et historique`.

### Task 6 : contrôleur de dictée

**Files:** Create `vocalis/controleur.py`, `tests/test_controleur.py`

**Interfaces — Consumes:** `Moteur`, `Enregistreur`, `rogner_silences`, `Historique`, `Reglages`.
**Produces:** `Controleur(QObject)` — états `REPOS`, `ECOUTE`, `TRANSCRIPTION` ; méthodes `debut_dictee()`, `fin_dictee()`, `changer_moteur(moteur)` ; signaux `etat_change(str)`, `niveau(float)`, `texte_pret(str)`, `erreur(str)`. La transcription tourne dans un `QThread`. Durée minimale d'enregistrement : 0,3 s.

- [ ] Tests (faux moteur, faux enregistreur, `qtbot`-style avec `QEventLoop`) : cycle complet REPOS → ECOUTE → TRANSCRIPTION → REPOS avec `texte_pret("bonjour")` et une entrée d'historique ; audio silencieux → retour à REPOS sans `texte_pret` et sans appel au moteur ; `debut_dictee()` pendant TRANSCRIPTION ignoré ; `MicroIndisponible` → `erreur` émis, état REPOS ; exception du moteur → `erreur` émis, état REPOS.
- [ ] Vérifier l'échec, implémenter, vérifier le succès. Commit `feat: contrôleur de dictée`.

### Task 7 : interface (pastille, zone de notification, réglages, historique, fichiers)

**Files:** Create `vocalis/ui/{__init__,pastille,tray,reglages,historique,fichiers,style}.py`, `vocalis/demarrage.py`, `vocalis/__main__.py`, `vocalis/ressources/icone.svg`, `tests/test_ui.py`, `tests/test_demarrage.py`

**Interfaces — Produces:**
- `pastille.Pastille` : fenêtre sans bordure, toujours au-dessus, qui ne prend pas le focus (`Qt.WindowDoesNotAcceptFocus | Qt.Tool`), centrée en bas de l'écran ; `afficher_etat(etat)`, `afficher_niveau(float)`, `afficher_message(str, duree_ms=2500)`
- `reglages.FenetreReglages(reglages)` : signal `enregistres(Reglages)`
- `historique.FenetreHistorique(historique)`
- `fichiers.FenetreFichiers(moteur)` : choisir un fichier audio → texte + export `.srt`
- `demarrage.activer(bool)`, `demarrage.est_actif() -> bool` — clé `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`, valeur `Vocalis`
- `python -m vocalis` : instance unique (`QLockFile`), charge le moteur en arrière-plan, câble tout

- [ ] Tests (offscreen) : la pastille passe par les trois états sans prendre le focus ; `FenetreReglages` rend un `Reglages` reflétant les champs modifiés et refuse un raccourci invalide ; `FenetreHistorique` liste les entrées ; `demarrage` testé avec `winreg` remplacé par un faux.
- [ ] Implémenter ; lancer `python -m vocalis` et vérifier à la main : dictée dans le Bloc-notes, changement de moteur, historique, transcription de `samples/exemple_fr.wav`.
- [ ] Commit `feat(ui): app de dictée dans la zone de notification`.

### Task 8 : finitions de la partie A

**Files:** Modify `README.md`, `app.py` (choix du moteur Pro/Maison via `creer_moteur`), Create `lancer.bat`, `vocalis.spec` (PyInstaller)

- [ ] README : installation, lancement, raccourci, les deux moteurs, structure.
- [ ] `lancer.bat` démarre `pythonw -m vocalis`.
- [ ] Construire l'exécutable avec PyInstaller et vérifier qu'il démarre et dicte.
- [ ] Suite complète de tests verte. Commit `docs: README de l'app et empaquetage`.

---

# Partie B — le modèle maison

Prérequis qui dépendent d'Othmane : ~15 Go de disque libérés, conditions de Common Voice acceptées.

### Task 9 : test du GPU (fixe la taille du modèle)

- [ ] Essayer dans l'ordre : PyTorch ROCm sous WSL2 (Ubuntu 24.04), PyTorch ROCm sous Windows, `torch-directml`. Critère : une passe avant + arrière d'un Conformer de 30 M de paramètres sur un lot de 16 × 10 s, avec `ctc_loss`, plus rapide que sur CPU.
- [ ] Consigner le résultat dans `modele/README.md` : périphérique retenu, vitesse mesurée, taille choisie (`moyen` si GPU, `petit` sinon).

### Task 10 : tokenizer BPE

**Files:** Create `modele/tokenizer.py`, `tests/test_tokenizer.py`

**Produces:** `normaliser_texte(str) -> str` (minuscules, NFC, ne garde que lettres françaises, apostrophe, tiret, espace) ; `TokenizerBPE.entrainer(textes, taille_vocab=500) -> TokenizerBPE` ; `encoder(str) -> list[int]` ; `decoder(list[int]) -> str` ; `enregistrer(chemin)` / `charger(chemin)` ; id 0 réservé au blanc CTC ; l'espace est marqué par `▁` en début de mot.

- [ ] Tests : `decoder(encoder(t)) == normaliser_texte(t)` pour des phrases avec accents, apostrophes et un caractère jamais vu ; taille du vocabulaire ≤ 500 ; l'id 0 n'est jamais produit par `encoder` ; sauvegarde puis chargement donnent le même encodage.

### Task 11 : log-Mel en PyTorch

**Files:** Create `modele/features.py`, `tests/test_features_torch.py`

**Produces:** `LogMel(nn.Module)` : `forward(audio: Tensor[B, T]) -> Tensor[B, 80, T // 160]`, mêmes constantes que `stt.features` (`N_FFT=400`, `HOP=160`, `N_MELS=80`), normalisation par énoncé (moyenne 0, variance 1 par bande).

- [ ] Test : avant normalisation, l'écart absolu maximal avec `stt.features.log_mel` sur `samples/exemple_fr.wav` est < 1e-3.

### Task 12 : Conformer + CTC

**Files:** Create `modele/conformer.py`, `modele/decodage.py`, `tests/test_conformer.py`, `tests/test_decodage.py`

**Produces:**
- `ConfigModele(d=256, couches=12, tetes=4, noyau_conv=15, dropout=0.1, taille_vocab=500)` ; `ConfigModele.petit()` = `d=144, couches=8`
- `Conformer(config)` : `forward(mel: Tensor[B, 80, T], longueurs: Tensor[B]) -> tuple[Tensor[B, T // 4, vocab], Tensor[B]]` (log-probabilités)
- Blocs écrits à la main : sous-échantillonnage convolutif ×4, attention multi-têtes à positions relatives, module de convolution (pointwise → GLU → depthwise → BatchNorm → Swish → pointwise), feed-forward en demi-résidus
- `decodage.decoder_glouton(log_probas, longueurs, tokenizer) -> list[str]` — fusion des répétitions puis retrait du blanc

- [ ] Tests : formes de sortie et longueurs pour deux durées différentes ; nombre de paramètres entre 25 M et 35 M (`moyen`) et entre 8 M et 12 M (`petit`) ; le masque de padding rend la sortie d'un énoncé indépendante de ce avec quoi il est mis en lot (écart < 1e-4 en mode `eval`) ; **sur-apprentissage** : 200 pas sur 4 énoncés synthétiques font tomber la perte CTC sous 0.1 et le décodage rend les 4 textes exacts ; `decoder_glouton` sur `[0, 5, 5, 0, 5, 7]` rend les tokens `5 5 7`.

### Task 13 : données

**Files:** Create `modele/preparer.py`, `modele/donnees.py`, `tests/test_donnees.py`

**Produces:**
- `preparer.py --heures 100 --sortie donnees/` : écrit `train.tsv`, `dev.tsv`, `test.tsv` (colonnes `chemin`, `duree`, `texte`) à partir de Common Voice français ; ne garde que les clips de 1 à 15 s ; `test.tsv` = jeu de test officiel, sans locuteur commun avec `train`
- `donnees.JeuAudio(manifeste, tokenizer, augmenter=False)` ; `donnees.assembler(lot)` → `(audio, longueurs_audio, cibles, longueurs_cibles)` ; `donnees.EchantillonneurParDuree(durees, secondes_par_lot)` ; `donnees.SpecAugment(masques_freq=2, largeur_freq=27, masques_temps=5, part_temps=0.05)`

- [ ] Tests (WAV synthétiques dans `tmp_path`) : `assembler` complète avec des zéros et rend les bonnes longueurs ; chaque lot de l'échantillonneur fait au plus `secondes_par_lot` ; `SpecAugment` ne modifie rien en mode `eval` et met des zéros en mode `train` ; une ligne de manifeste dont le fichier manque est ignorée avec un avertissement.

### Task 14 : entraînement et évaluation

**Files:** Create `modele/entrainer.py`, `modele/evaluer.py`, `tests/test_entrainer.py`

**Produces:**
- `entrainer.py --donnees donnees/ --taille moyen --epoques 60 --reprendre` : AdamW (`lr` crête 1e-3, 10 000 pas de montée puis décroissance en 1/√pas), précision mixte si disponible, écrêtage du gradient à 5.0, moyenne des 5 meilleurs checkpoints, journal `journal.csv` (`pas, perte, wer_dev`), checkpoint `{config, vocab, poids, pas, wer_dev}`
- `evaluer.py --checkpoint … --manifeste … [--comparer-whisper small]` : affiche WER et CER, écrit `resultats.json`

- [ ] Tests : un entraînement de 30 pas sur 8 énoncés synthétiques fait baisser la perte et écrit un checkpoint ; `--reprendre` repart du bon pas ; `evaluer` sur ce checkpoint rend un WER entre 0 et 1.
- [ ] Lancer l'entraînement réel ; reporter les courbes et le WER mesuré dans `modele/README.md`.

### Task 15 : branchement du moteur Maison

**Files:** Modify `vocalis/moteurs/maison.py`, `README.md` · Test `tests/test_moteur_maison.py`

- [ ] `MoteurMaison.transcrire` : `LogMel` → `Conformer` → `decoder_glouton`, par tranches de 20 s coupées sur les silences ; un seul segment horodaté par tranche ; `langue` ignorée (français).
- [ ] Tests : avec un mini-checkpoint produit par la tâche 14, `transcrire` rend un `Resultat` ; audio vide → `texte == ""`.
- [ ] Copier le checkpoint entraîné vers `CHECKPOINT_DEFAUT`, vérifier la dictée avec le moteur Maison dans l'app.
- [ ] README : tableau WER / CER / vitesse, Maison contre Whisper, sur le jeu de test et sur `samples/`.
