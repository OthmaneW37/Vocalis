# Projet 25 – Audio : Speech Recognition (Speech-to-Text)

Dossier de recherche et base de travail — rédigé le 11/11/2026.

> **Consigne** : *Explain speech-to-text and demonstrate transcription of a short recording.*
> Trois livrables attendus : (1) expliquer le fonctionnement, (2) détailler les étapes signal → texte, (3) une démo : un fichier audio en entrée, sa transcription en sortie.

---

## 0. En bref : ce que je te recommande

| Décision | Choix recommandé | Pourquoi |
|---|---|---|
| Modèle | **Whisper** (OpenAI, open source) | Cité dans la consigne, multilingue (FR/EN/AR), robuste, très documenté |
| Bibliothèque d'inférence | **Hugging Face Transformers** | Déjà installée chez toi, citée dans la consigne, et elle expose chaque étape (spectrogramme, tokens) → idéal pour *expliquer* |
| Audio | **Librosa** | Citée dans la consigne ; charge + rééchantillonne à 16 kHz, et trace forme d'onde / spectrogramme Mel |
| Taille du modèle | **`small`** par défaut, `base` en secours pour la démo live | Meilleur compromis qualité/temps sur ton CPU |
| Matériel | **CPU** (pas de GPU) | Suffisant pour un enregistrement court ; évite 2–3 Go d'installation CUDA |
| Interface | **Gradio** | Import de fichier + micro dans un seul composant, ~30 lignes |
| Langue de démo | **Français + anglais**, détection automatique | Montre le côté multilingue |
| Évaluation | **WER** avec `jiwer` sur 3–5 enregistrements | Donne des chiffres à présenter, pas juste « ça marche » |

Le reste du document justifie ces choix et donne les alternatives.

---

## 1. Comment fonctionne la reconnaissance vocale

### 1.1 Le problème

Trouver la suite de mots **W** la plus probable sachant un signal audio **X** :
`W* = argmax P(W | X)`.

Ce qui rend la tâche difficile : variabilité des locuteurs (accent, débit, timbre), bruit et réverbération, pas de frontière nette entre les mots à l'oral, homophones (« vert / verre / vers »), longueurs différentes entre l'audio (des milliers de trames) et le texte (quelques dizaines de mots).

### 1.2 Trois générations de systèmes

| Époque | Approche | Principe |
|---|---|---|
| 1980 – 2012 | **HMM-GMM** | Chaîne de 3 modules séparés : modèle acoustique (MFCC → phonèmes), dictionnaire de prononciation (phonèmes → mots), modèle de langue n-gram (mots → phrases) |
| 2012 – 2016 | **Hybride DNN-HMM** | Un réseau de neurones remplace les GMM, le reste de la chaîne est conservé |
| 2016 → aujourd'hui | **End-to-end** | Un seul réseau audio → texte, entraîné d'un bloc |

Les trois familles end-to-end à connaître :

- **CTC** (Connectionist Temporal Classification, Graves 2006) : le réseau émet un caractère (ou un « blanc ») par trame, puis on fusionne les répétitions. Simple et rapide, mais chaque trame est prédite indépendamment. Exemple : wav2vec 2.0.
- **Encodeur-décodeur avec attention** (seq2seq) : un encodeur résume l'audio, un décodeur écrit le texte token par token en « regardant » l'audio par attention. **C'est l'architecture de Whisper.**
- **Transducer** (RNN-T / TDT) : compromis entre les deux, adapté au streaming. Exemple : NVIDIA Parakeet.

Deux façons d'entraîner :

- **Auto-supervisé** (wav2vec 2.0, HuBERT) : pré-entraînement sur de l'audio brut sans transcription, puis fine-tuning sur peu de données annotées.
- **Faiblement supervisé à grande échelle** (Whisper) : 680 000 heures d'audio du web avec leurs sous-titres. Moins « propre », mais énorme et varié → modèle robuste sans fine-tuning.

### 1.3 Le lien avec le module Vision Artificielle

C'est l'angle à mettre en avant devant le prof : **Whisper transforme le son en image, puis la traite avec des outils de vision.**

- Le spectrogramme log-Mel est une image 2D (temps × fréquence), ici 80 × 3000 « pixels ».
- Les deux premières couches de Whisper sont des **convolutions**, comme dans un CNN.
- La suite est un **Transformer**, le même principe que les Vision Transformers (ViT) : l'image est découpée en positions sur lesquelles on applique de l'auto-attention.

---

## 2. Les étapes : du signal audio au texte

```
 Voix (onde de pression)
   │  microphone + convertisseur analogique-numérique
   ▼
[1] Signal numérique          échantillonnage + quantification (ex. 44,1 kHz, 16 bits)
   ▼
[2] Prétraitement             mono, rééchantillonnage 16 kHz, normalisation, découpe en 30 s
   ▼
[3] Extraction de features    fenêtrage → STFT → banc de filtres Mel → log  = spectrogramme log-Mel
   ▼
[4] Encodeur                  2 convolutions + blocs Transformer → représentation acoustique
   ▼
[5] Décodeur                  génère les tokens un par un (attention sur la sortie de l'encodeur)
   ▼
[6] Décodage / recherche      greedy ou beam search → meilleure séquence de tokens
   ▼
[7] Post-traitement           tokens → texte, ponctuation, horodatage
   ▼
 Transcription
```

### [1] Numérisation
- **Échantillonnage** : on mesure l'amplitude N fois par seconde. Théorème de Nyquist-Shannon : pour capter une fréquence *f*, il faut échantillonner à au moins *2f*. La parole utile est sous 8 kHz → **16 kHz suffit** (c'est ce qu'attend Whisper).
- **Quantification** : chaque mesure est codée sur 16 bits (65 536 niveaux).
- Résultat : un simple tableau 1D de nombres. 1 s d'audio = 16 000 valeurs.

### [2] Prétraitement
- Conversion en **mono**, rééchantillonnage à **16 kHz**, amplitudes en flottants dans [-1, 1].
- Whisper travaille par **fenêtres de 30 s** : un audio plus court est complété par du silence, un plus long est découpé.
- Optionnel : **VAD** (Voice Activity Detection) pour retirer les silences, réduction de bruit.

### [3] Extraction de caractéristiques — le spectrogramme log-Mel
L'onde brute est peu exploitable ; on passe dans le domaine fréquentiel.

1. **Fenêtrage** : on découpe en trames de **25 ms** (400 échantillons) décalées de **10 ms** (160 échantillons), pondérées par une fenêtre de Hann. Sur 25 ms, la parole est quasi stationnaire.
2. **STFT** (transformée de Fourier à court terme) : une FFT par trame → énergie par fréquence.
3. **Échelle Mel** : on regroupe les fréquences avec un banc de **80 filtres** triangulaires, serrés dans les graves et larges dans les aigus, comme l'oreille humaine (`mel = 2595·log10(1 + f/700)`).
4. **Logarithme** : la perception du volume est logarithmique ; cela compresse aussi la dynamique.

Résultat pour 30 s : une matrice **80 × 3000** (80 bandes Mel, 3000 trames). `large-v3` et `turbo` utilisent 128 bandes.

> **MFCC** : les anciens systèmes ajoutaient une DCT pour obtenir ~13 coefficients décorrélés. Les réseaux profonds s'en passent et prennent directement le log-Mel.

### [4] Encodeur
- 2 couches de **convolution 1D** (la seconde avec un pas de 2) → 3000 trames deviennent **1500 positions** (une toutes les 20 ms).
- Ajout d'un encodage de position, puis *N* blocs **Transformer** (auto-attention + MLP).
- Sortie : 1500 vecteurs qui décrivent « ce qui est dit » à chaque instant, dans leur contexte.

### [5] Décodeur
Un Transformer **autorégressif** : il prédit le token suivant à partir des tokens déjà écrits et de la sortie de l'encodeur (attention croisée). C'est lui qui joue le rôle de modèle de langue.

Whisper est **multitâche**, piloté par des tokens spéciaux en début de séquence :

```
<|startoftranscript|> <|fr|> <|transcribe|> <|notimestamps|>  Bonjour tout le monde <|endoftext|>
                       langue   tâche           option
```

`<|transcribe|>` peut être remplacé par `<|translate|>` (traduction vers l'anglais). Sans langue imposée, le modèle la détecte lui-même. Les tokens sont des sous-mots (BPE, ~51 865 entrées).

### [6] Stratégie de décodage
- **Greedy** : on prend le token le plus probable à chaque pas. Rapide.
- **Beam search** : on garde les *k* meilleures hypothèses (souvent 5). Meilleur, plus lent.
- Whisper ajoute un repli sur la **température** quand la sortie semble dégénérée (répétitions).

### [7] Post-traitement
Détokenisation, horodatage des segments (pour un export sous-titres `.srt`). Whisper produit directement ponctuation et majuscules, ce que les systèmes CTC ne font pas.

### Évaluation : le WER
**Word Error Rate** = (Substitutions + Suppressions + Insertions) / nombre de mots de la référence.

Exemple : référence « le chat dort sur le canapé », sortie « le chat dors sur canapé » → 1 substitution + 1 suppression sur 6 mots = **WER 33 %**. Avant de calculer, on normalise (minuscules, ponctuation retirée). Bibliothèque : `jiwer`.

---

## 3. Whisper en détail

- Publié par OpenAI en septembre 2022 (Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*). Code et poids sous licence MIT.
- Entraîné sur **680 000 h** d'audio multilingue. `large-v3` : 1 M d'heures faiblement annotées + 4 M pseudo-annotées.
- **99 langues**, transcription + traduction vers l'anglais + détection de langue + horodatage.

| Modèle | Paramètres | Couches enc./déc. | VRAM indicative | Vitesse relative* |
|---|---|---|---|---|
| tiny | 39 M | 4 / 4 | ~1 Go | ~10× |
| base | 74 M | 6 / 6 | ~1 Go | ~7× |
| small | 244 M | 12 / 12 | ~2 Go | ~4× |
| medium | 769 M | 24 / 24 | ~5 Go | ~2× |
| large-v3 | 1 550 M | 32 / 32 | ~10 Go | 1× |
| turbo (large-v3-turbo) | 809 M | 32 / 4 | ~6 Go | ~8× |

\* Mesurée par OpenAI sur GPU A100, en anglais. Sur ton PC les rapports seront différents : à mesurer.

`turbo` = `large-v3` dont le décodeur passe de 32 à 4 couches : presque la même qualité, beaucoup plus rapide, mais pas de traduction.

**Limites connues (à citer dans la présentation)** :
- **Hallucinations** : sur du silence ou du bruit, le modèle peut inventer du texte ou boucler sur une phrase.
- Qualité inégale selon la langue : très bon en anglais/français, plus faible sur les langues peu représentées et les dialectes (la **darija** est un bon exemple à montrer).
- Pas temps réel par construction (fenêtres de 30 s).
- Noms propres et vocabulaire technique souvent mal orthographiés.
- Ne dit pas *qui* parle (pas de diarisation).

**Où en est l'état de l'art fin 2026** : Whisper n'est plus en tête de l'Open ASR Leaderboard de Hugging Face. D'après des comparatifs tiers d'octobre 2026 (chiffres indicatifs, classement anglais), `large-v3` tourne autour de 7,2 % de WER moyen contre 5,3–5,4 % pour les meilleurs modèles ouverts (NVIDIA, Qwen3-ASR, Cohere Transcribe). Il reste la référence pédagogique : multilingue, simple à installer, omniprésent. Une phrase là-dessus en conclusion montre que tu connais le contexte.

---

## 4. Les choix à faire

### Choix 1 — Quel modèle ?

| Option | Pour | Contre |
|---|---|---|
| **Whisper** ✅ | Dans la consigne, multilingue, ponctuation, zéro réglage | Plus n°1 en précision pure |
| wav2vec 2.0 (Meta) | Illustre bien le CTC ; rapide | Un modèle par langue, pas de ponctuation, moins bon |
| Parakeet / Canary (NVIDIA) | Meilleur WER, très rapide | Installation NeMo lourde sous Windows, hors consigne |
| API cloud (OpenAI, Google, Deepgram) | Aucune installation | Payant, besoin d'internet le jour J, « boîte noire » peu valorisante |

→ **Whisper**. Tu peux citer wav2vec 2.0 dans la partie théorique comme contre-exemple CTC.

### Choix 2 — Quelle bibliothèque pour faire tourner Whisper ?

| Option | Pour | Contre |
|---|---|---|
| **`transformers`** ✅ | Déjà installée (5.22.1) avec PyTorch ; donne accès au spectrogramme et aux tokens ; pas besoin de ffmpeg si l'audio est chargé avec Librosa | Un peu plus lente que faster-whisper |
| `openai-whisper` | La plus simple (`model.transcribe(...)`), fonctions pédagogiques | **Exige ffmpeg, absent de ton PC** |
| `faster-whisper` | Plus rapide sur CPU (quantification int8), VAD intégré, sans ffmpeg | Moins transparente ; GPU pénible sous Windows (CUDA 12 + cuDNN 9) |

→ **`transformers` + `librosa`** : tu utilises exactement les technologies listées par le prof. `faster-whisper` peut servir de bonus « optimisation » (comparer les temps).

### Choix 3 — Taille du modèle

Ta machine : i5-10300H, 16 Go de RAM, GTX 1650 Ti 4 Go.

- `tiny` / `base` : quasi instantané, mais fautes visibles en français.
- **`small`** ✅ : bon français, tient largement en RAM, quelques secondes pour un extrait court.
- `medium` / `turbo` : meilleure qualité, mais lent sur CPU.

→ **`small` par défaut**, avec un menu déroulant dans l'application pour changer de modèle : la comparaison `tiny` vs `small` devient une partie de la démo.

### Choix 4 — CPU ou GPU ?

Ton PyTorch actuel est la version **CPU** (`2.14.1+cpu`), le GPU n'est donc pas utilisé.

- **Rester sur CPU** ✅ : rien à installer, suffisant pour « a short recording ».
- Passer au GPU : réinstaller PyTorch en version CUDA (plusieurs Go). 4 Go de VRAM limitent à `small`/`medium`, et les GTX 16xx ont un problème connu en demi-précision (fp16) → il faut rester en float32.

→ **CPU**. Le gain ne vaut pas le risque d'une installation cassée avant la soutenance.

### Choix 5 — Interface

| Option | Pour | Contre |
|---|---|---|
| **Gradio** ✅ | `gr.Audio` gère import **et** micro ; standard des démos ML | À installer (compatible Python 3.13, vérifié) |
| Streamlit | Déjà installé chez toi (1.52.2) | Un peu plus de code pour le même résultat |
| Tkinter | Vraie fenêtre de bureau, aucune dépendance | Rendu daté, pas de micro simple |
| Notebook / script | Le plus rapide à écrire | Moins « application », moins démonstratif |

→ **Gradio**. Si tu es plus à l'aise avec Streamlit, c'est un choix tout aussi valable.

### Choix 6 — Périmètre fonctionnel

**Socle (obligatoire pour répondre à la consigne)**
1. Importer un fichier audio (wav, mp3).
2. Afficher la transcription.

**Ce qui fait la différence, par ordre de rentabilité**
3. Afficher la **forme d'onde et le spectrogramme log-Mel** → relie la démo à la théorie.
4. Afficher la **langue détectée** et le **temps de calcul**.
5. Choix du modèle (`tiny` / `base` / `small`).
6. Enregistrement au **micro** → démo en direct.
7. **Horodatage** des segments + export `.txt` / `.srt`.
8. Calcul du **WER** si l'utilisateur fournit le texte de référence.
9. Mode **traduction** vers l'anglais.

→ Vise 1 à 6. Les points 7 à 9 sont à ajouter s'il reste du temps.

### Choix 7 — Langue et enregistrements de démo

Prépare **toi-même** 4 ou 5 enregistrements de 10 à 30 s, avec leur transcription exacte :

| Fichier | But |
|---|---|
| Français, voix claire | Cas nominal |
| Anglais | Montre le multilingue et la détection de langue |
| Français avec bruit de fond | Montre la robustesse et ses limites |
| Darija ou mélange FR/AR | Montre une limite honnête du modèle |
| Phrase avec noms propres / termes techniques | Montre les erreurs typiques |

Enregistrer ta propre voix évite toute question de droits et prouve que ce n'est pas un exemple préparé par quelqu'un d'autre.

---

## 5. Architecture proposée

```
Projet/
├── app.py               # interface Gradio
├── stt/
│   ├── audio.py         # chargement (librosa), rééchantillonnage, durée
│   ├── features.py      # forme d'onde + spectrogramme log-Mel (figures)
│   ├── transcribe.py    # chargement du modèle, transcription, langue, temps
│   └── evaluate.py      # WER avec jiwer
├── demo_etapes.ipynb    # notebook qui déroule les 7 étapes une par une
├── samples/             # tes enregistrements + references.json
├── requirements.txt
└── RECHERCHE_Speech_Recognition.md
```

Le **notebook** sert à la partie « expliquer » (on voit chaque transformation), l'**application** à la partie « démontrer ».

### Installation

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install torch transformers librosa gradio jiwer matplotlib
```

Versions résolues aujourd'hui sur ton Python 3.13.5 : gradio 6.31.0, librosa 1.0.0, jiwer 4.0.0 (faster-whisper 1.2.1 s'installe aussi). Au premier lancement, le modèle `small` (~1 Go) est téléchargé depuis Hugging Face : **fais-le avant la soutenance**, pas pendant.

### Cœur du code (esquisse, non exécutée — à valider à l'implémentation)

```python
import time, librosa, gradio as gr
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcrire(chemin):
    audio, sr = librosa.load(chemin, sr=16000, mono=True)   # étapes 1-2
    t0 = time.time()
    sortie = asr({"raw": audio, "sampling_rate": sr},        # étapes 3 à 7
                 return_timestamps=True)
    return sortie["text"], f"{time.time() - t0:.1f} s"

gr.Interface(
    fn=transcrire,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath"),
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Temps")],
).launch()
```

Pour montrer les étapes dans le notebook, on remplace le `pipeline` par ses deux moitiés : `WhisperProcessor` (audio → matrice 80 × 3000, qu'on affiche comme une image) puis `WhisperForConditionalGeneration.generate` (matrice → identifiants de tokens, qu'on décode un par un).

---

## 6. Expérience à mener (pour avoir des résultats à montrer)

Sur tes 4–5 enregistrements, pour `tiny`, `base` et `small` : mesurer le **WER** et le **temps de transcription**. Cela donne un tableau et un graphique « précision vs vitesse », bien plus convaincants qu'une seule transcription réussie.

| Modèle | WER moyen | Temps moyen | Facteur temps réel (temps / durée audio) |
|---|---|---|---|
| tiny | … | … | … |
| base | … | … | … |
| small | … | … | … |

---

## 7. Plan de présentation suggéré

1. **Introduction** — où la reconnaissance vocale est utilisée (assistants, sous-titres, dictée), pourquoi c'est difficile.
2. **Du son au nombre** — échantillonnage, 16 kHz, forme d'onde.
3. **Du nombre à l'image** — STFT, échelle Mel, spectrogramme (figure issue de *ton* enregistrement).
4. **De l'image au texte** — évolution HMM → end-to-end, architecture de Whisper (schéma encodeur-décodeur), tokens spéciaux.
5. **Démonstration** — import d'un fichier, puis micro en direct, puis comparaison `tiny` / `small`.
6. **Résultats** — tableau WER / temps.
7. **Limites et perspectives** — hallucinations, dialectes, temps réel, diarisation, modèles plus récents.

**Questions probables du prof**
- Pourquoi 16 kHz ? → Nyquist, la parole utile est sous 8 kHz.
- Pourquoi l'échelle Mel et le logarithme ? → pour imiter la perception humaine.
- Quelle différence entre CTC et encodeur-décodeur ? → trame par trame sans contexte de sortie vs génération autorégressive avec attention.
- Comment Whisper connaît-il la langue ? → il prédit le token de langue après `<|startoftranscript|>`.
- Comment mesure-t-on la qualité ? → WER.
- Quel rapport avec la vision ? → spectrogramme = image, convolutions, Transformer comme ViT.
- As-tu entraîné le modèle ? → non, modèle pré-entraîné utilisé en inférence ; l'entraînement a demandé 680 000 h d'audio.

---

## 8. Pièges à éviter

- **ffmpeg n'est pas installé** sur ton PC : `openai-whisper` et le chargement de fichier direct par `transformers` échoueront. Passer par `librosa.load` contourne le problème pour wav/mp3/flac. Pour du `.m4a` (dictaphone iPhone), convertis en wav ou installe ffmpeg.
- **Fréquence d'échantillonnage** : donner du 44,1 kHz à Whisper sans rééchantillonner produit du charabia.
- **Premier lancement lent** (téléchargement + chargement du modèle) : charger le modèle une seule fois au démarrage, pas à chaque clic.
- **Wi-Fi de la salle** : tout doit tourner hors ligne ; teste en mode avion.
- **Micro dans le navigateur** : Gradio demande une autorisation ; teste-le sur le PC de la soutenance.
- **Audio de plus de 30 s** : il faut l'horodatage ou le découpage activé, sinon seul le début est transcrit.
- **Silences longs** : risque d'hallucination ; coupe-les ou mentionne-le comme limite.
- Garde une **capture vidéo de la démo** en secours.

---

## 9. Étapes de réalisation

1. Créer l'environnement, installer, télécharger `small`.
2. Enregistrer les échantillons et écrire leurs transcriptions de référence.
3. Notebook : dérouler les 7 étapes sur un enregistrement, sauvegarder les figures.
4. Application Gradio : socle, puis spectrogramme, langue, temps, choix du modèle.
5. Lancer l'expérience WER / temps, remplir le tableau.
6. Rapport et diapositives à partir du plan de la section 7.
7. Répétition hors ligne.

---

## 10. Points que je n'ai pas pu trancher à ta place

- **Format du rendu** : rapport écrit, diapositives, ou les deux ? Durée de la soutenance ?
- **Travail seul ou en groupe ?**
- **Date limite ?** — elle détermine jusqu'où aller dans la liste du choix 6.

---

## Sources

Articles fondateurs
- Radford et al., 2022 — *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper) — https://arxiv.org/abs/2212.04356
- Baevski et al., 2020 — *wav2vec 2.0* — https://arxiv.org/abs/2006.11477
- Vaswani et al., 2017 — *Attention Is All You Need* — https://arxiv.org/abs/1706.03762
- Graves et al., 2006 — *Connectionist Temporal Classification* — https://www.cs.toronto.edu/~graves/icml_2006.pdf
- *Open ASR Leaderboard* (article) — https://arxiv.org/abs/2510.06961

Documentation et outils
- Whisper (dépôt officiel) — https://github.com/openai/whisper
- Tailles de modèles Whisper — https://www.mintlify.com/openai/whisper/concepts/models
- faster-whisper — https://github.com/SYSTRAN/faster-whisper
- Cours audio Hugging Face (démo Gradio) — https://huggingface.co/learn/audio-course/chapter5/demo
- Guide Gradio reconnaissance vocale — https://gradio.app/guides/real-time-speech-recognition
- Open ASR Leaderboard (dépôt) — https://github.com/huggingface/open_asr_leaderboard

Comparatifs 2026 (sources tierces, chiffres indicatifs)
- https://pinggy.io/blog/best_open_source_self_hosted_speech_to_text_models/
- https://apidog.com/blog/best-open-source-speech-to-text-models/
- https://the-decoder.com/open-asr-leaderboard-tests-more-than-60-speech-recognition-models-for-accuracy-and-speed/
