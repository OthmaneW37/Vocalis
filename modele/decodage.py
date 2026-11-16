"""Décodage CTC glouton et taux d'erreur (mots / caractères)."""

from .tokenizer import normaliser_texte


def reduire_ctc(ids) -> list[int]:
    """Règle de la CTC : fusionner les répétitions, puis retirer le blanc (id 0)."""
    sortie, precedent = [], None
    for i in ids:
        i = int(i)
        if i != precedent and i != 0:
            sortie.append(i)
        precedent = i
    return sortie


def decoder_glouton(log_probas, longueurs, tokenizer) -> list[str]:
    """log_probas [B, T, vocab] → un texte par énoncé (meilleur symbole à chaque trame)."""
    meilleurs = log_probas.argmax(dim=-1)
    return [tokenizer.decoder(reduire_ctc(meilleurs[b, : int(longueurs[b])].tolist()))
            for b in range(meilleurs.shape[0])]


def distance_edition(reference: list, hypothese: list) -> int:
    """Distance de Levenshtein (substitutions + suppressions + insertions)."""
    precedente = list(range(len(hypothese) + 1))
    for i, r in enumerate(reference, 1):
        courante = [i]
        for j, h in enumerate(hypothese, 1):
            courante.append(min(precedente[j] + 1, courante[j - 1] + 1,
                                precedente[j - 1] + (r != h)))
        precedente = courante
    return precedente[-1]


def taux_erreur(references: list[str], hypotheses: list[str], niveau: str = "mot") -> float:
    """WER (niveau="mot") ou CER (niveau="caractere") sur un corpus, après normalisation."""
    erreurs = total = 0
    for reference, hypothese in zip(references, hypotheses):
        r, h = normaliser_texte(reference), normaliser_texte(hypothese)
        r, h = (r.split(), h.split()) if niveau == "mot" else (list(r), list(h))
        erreurs += distance_edition(r, h)
        total += len(r)
    return erreurs / total if total else 0.0
