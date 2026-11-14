# Vocalis — app de dictée vocale à deux moteurs

Date : 2026-11-14 · Statut : en attente de validation

## Intention

Faire de Vocalis une application Windows de dictée utilisable au quotidien, dans
l'esprit de Handy : un raccourci clavier global, on parle, le texte s'écrit dans la
fenêtre active. L'app embarque deux moteurs :

- **Moteur Pro** : Whisper (via faster-whisper), pour l'usage réel.
- **Moteur Maison** : un modèle de reconnaissance vocale écrit et entraîné de zéro
  dans ce repo, sélectionnable dans l'app, avec son code d'entraînement et son
  taux d'erreur mesuré.

Décidé par Othmane : les deux moteurs cohabitent (choix « Les deux moteurs »).
Tout le reste ci-dessous est ma proposition, à corriger.

## Objectifs

1. Dicter dans n'importe quelle application Windows avec un raccourci global.
2. Fonctionner hors ligne, sur CPU, avec un délai court après la fin de la phrase.
3. Un modèle maison réellement entraîné de zéro, qui transcrit du français clair,
   avec un WER mesuré sur un jeu de test jamais vu à l'entraînement.
4. Garder la valeur pédagogique du repo actuel (notebook des 7 étapes, WER, SRT).

## Hors périmètre

- macOS et Linux (Windows 11 uniquement pour l'instant).
- Transcription mot à mot en direct pendant qu'on parle (le texte arrive à la fin
  de la phrase, comme dans Handy).
- Modèle maison multilingue : il ne fait que le français.
- Installeur signé / mise à jour automatique.

## Ce qui serait un échec

- Présenter le modèle maison comme équivalent à Whisper. Son WER réel est affiché
  tel quel dans l'app et le README.
- Un « modèle maison » qui réutilise des poids pré-entraînés (fine-tuning déguisé).
- Une app qui perd de l'audio, colle le texte dans la mauvaise fenêtre ou écrase
  le presse-papiers sans le restaurer.

## Contraintes constatées sur la machine

- Ryzen 5 7500F (6 cœurs), 16 Go de RAM, Radeon RX 9060 XT : pas de CUDA.
- 10,5 Go libres sur C:. Le cache Hugging Face en occupe 38 Go.
- Python 3.12 et 3.11 installés, PyTorch absent du Python système.
- WSL2 avec Ubuntu 24.04 présent.
- `faster-whisper-small` et `-base` sont déjà dans le cache local.

## Approches considérées

### L'application

| Approche | Pour | Contre |
|---|---|---|
| **A. App Python native (PySide6)** — retenue | Réutilise le code `stt/` existant ; les deux moteurs tournent dans le même processus ; un seul langage | Exécutable plus lourd qu'une app Rust |
| B. Réécriture Tauri/Rust comme Handy | Binaire léger, démarrage rapide | Réécriture complète ; le modèle maison doit passer par un export ONNX ; beaucoup plus de risque pour le même résultat visible |
| C. Garder Gradio + script de raccourci | Très peu de travail | Reste une page web, pas une app ; pas de saisie fiable dans les autres fenêtres |

### Le modèle maison

| Approche | Pour | Contre |
|---|---|---|
| **1. Encodeur Conformer + CTC** — retenue | Converge de façon fiable avec ~100 h de données ; décodage rapide sur CPU ; pas d'hallucinations | Pas de ponctuation ni de majuscules « apprises » finement |
| 2. Encodeur-décodeur façon Whisper | Même famille que le moteur Pro, joli parallèle pédagogique | Converge mal avec peu de données ; invente du texte sur les silences |
| 3. Fine-tuning de Whisper | Bien meilleure qualité | Ce n'est pas « de zéro » : exclu |

## Conception

### Application (dossier `vocalis/`)

- **Icône dans la zone de notification** avec menu : moteur actif, langue,
  historique, réglages, quitter.
- **Raccourci global** (défaut `Ctrl+Espace`, modifiable), deux modes :
  maintenir pour parler, ou appuyer une fois pour démarrer / une fois pour arrêter.
- **Pastille flottante** en bas de l'écran pendant l'écoute (niveau du micro, puis
  « transcription… »).
- **Capture micro** 16 kHz mono (`sounddevice`), avec détection de voix pour couper
  les silences de début et de fin.
- **Saisie du texte** : copie dans le presse-papiers, `Ctrl+V` simulé dans la
  fenêtre active, puis restauration de l'ancien presse-papiers.
- **Fenêtre de réglages** : micro, raccourci, moteur, taille du modèle Whisper,
  langue, démarrage avec Windows.
- **Historique** des dictées (texte, date, moteur, durée), stocké localement, avec
  copie en un clic.
- **Transcription de fichiers** audio depuis l'app, avec export SRT (reprend
  `vers_srt` existant).

```
  ┌────────────────────────────────┐
  │  ● ▁▃▆▃▁  Écoute…   Ctrl+Espace │   ← pastille pendant la dictée
  └────────────────────────────────┘
```

### Moteurs (interface commune)

Chaque moteur expose `transcrire(audio, langue) -> Resultat` (le `Resultat` actuel
de `stt/transcribe.py`). L'app ne connaît que cette interface.

- **Pro** : faster-whisper, CPU en int8, tailles `base` / `small` / `large-v3-turbo`.
- **Maison** : le checkpoint entraîné, chargé par PyTorch, décodage CTC.

### Modèle maison (dossier `modele/`)

- **Entrée** : spectrogramme log-Mel 80 bandes, en réutilisant l'implémentation
  manuelle de `stt/features.py`.
- **Architecture** : sous-échantillonnage convolutif ×4, puis 12 blocs Conformer
  (attention + convolution), ~30 M de paramètres, tête CTC. Tout est écrit à partir
  des briques de base de PyTorch, sans modèle importé.
- **Vocabulaire** : sous-mots BPE (~500 unités) appris sur les transcriptions
  d'entraînement.
- **Données** : Common Voice français, sous-ensemble validé d'environ 100 h, gardé
  en MP3 (~3 Go) ; les spectrogrammes sont calculés à la volée.
- **Entraînement** : SpecAugment, AdamW avec montée puis décroissance du taux
  d'apprentissage, précision mixte, reprise sur checkpoint, courbes de perte et de
  WER enregistrées.
- **Évaluation** : WER et CER sur le jeu de test officiel de Common Voice et sur
  `samples/`, comparés à Whisper dans un tableau du README.
- **Ordre de grandeur attendu** (estimation, pas une mesure) : 20 à 35 % de WER,
  contre environ 10 % pour Whisper small sur les mêmes phrases.

### Ce qui reste du repo actuel

- `stt/` devient la base commune (audio, features, WER, SRT).
- `demo_etapes.ipynb` et `RECHERCHE_Speech_Recognition.md` sont conservés.
- `app.py` (Gradio) est conservé comme démo pédagogique et gagne le choix du moteur.

## Risques ouverts

1. **Entraînement sur la Radeon.** Trois pistes à tester dans cet ordre : PyTorch
   ROCm sous WSL2, PyTorch ROCm sous Windows, DirectML. Si aucune ne marche, deux
   replis : entraîner sur un GPU gratuit en ligne (Kaggle/Colab) avec le même
   script, ou réduire le modèle à ~10 M de paramètres pour un entraînement CPU de
   plusieurs jours, avec un WER nettement plus mauvais. Ce test est la première
   étape du plan, car il fixe la taille du modèle.
2. **Disque.** 10,5 Go libres ne suffisent pas pour PyTorch + données +
   checkpoints. Il faut libérer environ 15 Go ; le modèle Qwen 27B du cache Hugging
   Face est le candidat évident, mais c'est à Othmane de décider. Rien n'est
   supprimé sans son accord.
3. **Common Voice** demande d'accepter ses conditions d'utilisation sur Hugging
   Face ou Mozilla Data Collective : action d'Othmane.

## Laissé au constructeur

Bibliothèque de raccourci global, forme exacte de la détection de voix, format de
stockage de l'historique, hyperparamètres fins, empaquetage en `.exe`.
