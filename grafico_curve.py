"""Curve di apprendimento delle varianti a partire dalle valutazioni periodiche.

Produce, nella cartella di uscita:
- curve_<varianti>.png        figura riassuntiva (esiti in alto, successo per seed in basso);
- esiti_<variante>.png        composizione degli esiti nel corso dell'addestramento
                              (media sui seed), una figura per variante;
- successo_<variante>.png     tasso di successo di ogni seed (linee sottili) e media
                              (linea spessa), una figura per variante;
- successo_medio.png          tasso di successo medio delle varianti sovrapposte,
                              un colore per variante.
Le figure singole sono senza titolo (da mettere in didascalia), con la legenda
a destra, e sono salvate ad alta risoluzione (300 dpi).

Uso:  python grafico_curve.py --varianti A B C
      python grafico_curve.py --varianti A B C --cartella results/curve --uscita results/figures --dpi 300
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# Esiti in ordine fisso, colori categoriali fissi (stesso colore = stesso esito ovunque)
ESITI = [
    ("successo", "Successo", "#2a78d6"),
    ("schianto", "Schianto", "#eb6834"),
    ("fuori_piattaforma", "Fuori piattaforma", "#1baf7a"),
    ("fuori_area", "Fuori area", "#eda100"),
    ("timeout", "Timeout", "#e87ba4"),
]
# Colori delle varianti nel grafico sovrapposto (ordine fisso: A, B, C)
COLORI_VARIANTI = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COLORE_SUCCESSO = "#2a78d6"
TESTO, TESTO_2, GRIGLIA, SFONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ETICHETTA_X = "passi di addestramento (migliaia)"


def leggi(cartella, variante):
    curve = []
    for f in sorted(Path(cartella).glob(f"{variante}_seed*.csv")):
        righe = list(csv.DictReader(open(f)))
        curve.append({k: np.array([float(r[k]) for r in righe]) for k in righe[0]})
    if not curve:
        raise SystemExit(f"Nessuna curva trovata per la variante {variante} in {cartella}")
    return curve


def stile(ax):
    ax.set_facecolor(SFONDO)
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    for lato in ("left", "bottom"):
        ax.spines[lato].set_color(GRIGLIA)
    ax.tick_params(colors=TESTO_2, labelsize=9)
    ax.grid(axis="y", color=GRIGLIA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def legenda_esiti(fig_o_ax, **kw):
    maniglie = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ESITI]
    kw.setdefault("ncol", 5)
    fig_o_ax.legend(maniglie, [nome for _, nome, _ in ESITI], frameon=False,
                    fontsize=9, labelcolor=TESTO, **kw)


# --- singoli pannelli ------------------------------------------------------------
def disegna_esiti(ax, curve, variante, titolo=True):
    passi = curve[0]["passi"] / 1e3
    medie = [np.mean([c["tasso_" + k] for c in curve], axis=0) * 100 for k, _, _ in ESITI]
    ax.stackplot(passi, *medie, colors=[c for _, _, c in ESITI], edgecolor=SFONDO, linewidth=1.5)
    stile(ax)
    if titolo:
        ax.set_title(f"Variante {variante}: esiti (media su {len(curve)} seed)",
                     loc="left", fontsize=11, color=TESTO)


def disegna_successo(ax, curve, variante, titolo=True):
    passi = curve[0]["passi"] / 1e3
    for c in curve:
        ax.plot(passi, c["tasso_successo"] * 100, color=COLORE_SUCCESSO, alpha=0.35, linewidth=1)
    media = np.mean([c["tasso_successo"] for c in curve], axis=0) * 100
    ax.plot(passi, media, color=COLORE_SUCCESSO, linewidth=2.2)
    stile(ax)
    if titolo:
        ax.set_title(f"Variante {variante}: successo per seed", loc="left", fontsize=11,
                     color=TESTO)
    # etichetta a destra dell'ultimo punto, fuori dalle curve
    ax.text(passi[-1] + 15, max(media[-1], 7), f"media\n{media[-1]:.0f}%", va="center",
            fontsize=9, color=TESTO)
    ax.set_xlim(right=passi[-1] + 110)


def salva(fig, percorso, dpi):
    fig.savefig(percorso, dpi=dpi, facecolor=SFONDO, bbox_inches="tight")
    plt.close(fig)
    print("salvato", percorso)


# --- figure -----------------------------------------------------------------------
def figura_riassuntiva(dati, uscita, dpi):
    n = len(dati)
    fig, assi = plt.subplots(2, n, figsize=(max(5.2 * n, 8.5), 7.2), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SFONDO)
    for j, (variante, curve) in enumerate(dati.items()):
        disegna_esiti(assi[0, j], curve, variante)
        disegna_successo(assi[1, j], curve, variante)
        assi[1, j].set_xlabel(ETICHETTA_X, color=TESTO_2, fontsize=9)
    assi[0, 0].set_ylabel("quota degli episodi", color=TESTO_2, fontsize=9)
    assi[1, 0].set_ylabel("tasso di successo", color=TESTO_2, fontsize=9)
    legenda_esiti(fig, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    salva(fig, uscita / f"curve_{''.join(dati)}.png", dpi)


def figure_singole(dati, uscita, dpi):
    for variante, curve in dati.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        fig.patch.set_facecolor(SFONDO)
        disegna_esiti(ax, curve, variante, titolo=False)
        ax.set_xlabel(ETICHETTA_X, color=TESTO_2, fontsize=9)
        ax.set_ylabel("quota degli episodi", color=TESTO_2, fontsize=9)
        legenda_esiti(ax, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
        salva(fig, uscita / f"esiti_{variante}.png", dpi)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        fig.patch.set_facecolor(SFONDO)
        disegna_successo(ax, curve, variante, titolo=False)
        ax.set_xlabel(ETICHETTA_X, color=TESTO_2, fontsize=9)
        ax.set_ylabel("tasso di successo", color=TESTO_2, fontsize=9)
        maniglie = [plt.Line2D([], [], color=COLORE_SUCCESSO, alpha=0.35, linewidth=1),
                    plt.Line2D([], [], color=COLORE_SUCCESSO, linewidth=2.2)]
        ax.legend(maniglie, ["singolo seed", "media sui seed"], loc="center left",
                  bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, labelcolor=TESTO)
        salva(fig, uscita / f"successo_{variante}.png", dpi)


def figura_medie(dati, uscita, dpi):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    fig.patch.set_facecolor(SFONDO)
    finali = []
    n = len(dati)
    for i, (colore, (variante, curve)) in enumerate(zip(COLORI_VARIANTI, dati.items())):
        passi = curve[0]["passi"] / 1e3
        media = np.mean([c["tasso_successo"] for c in curve], axis=0) * 100
        # la prima variante sta sopra (zorder più alto) e ogni linea ha un bordo
        # del colore dello sfondo, così le curve sovrapposte restano distinguibili
        ax.plot(passi, media, color=colore, linewidth=2.2, label=f"Variante {variante}",
                zorder=3 + n - i,
                path_effects=[pe.Stroke(linewidth=4.2, foreground=SFONDO), pe.Normal()])
        finali.append([passi[-1], media[-1], variante])
    stile(ax)
    # etichette dirette a fine curva, spostate se si sovrappongono
    finali.sort(key=lambda f: f[1])
    for i in range(1, len(finali)):
        finali[i][1] = max(finali[i][1], finali[i - 1][1] + 6)
    for x, y, variante in finali:
        ax.text(x + 12, y, variante, va="center", fontsize=10, color=TESTO)
    ax.set_xlim(right=ax.get_xlim()[1] + 30)
    ax.set_xlabel(ETICHETTA_X, color=TESTO_2, fontsize=9)
    ax.set_ylabel("tasso di successo", color=TESTO_2, fontsize=9)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
              labelcolor=TESTO)
    salva(fig, uscita / "successo_medio.png", dpi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--varianti", nargs="+", required=True)
    parser.add_argument("--cartella", default="results/curve")
    parser.add_argument("--uscita", default="results/figures", help="cartella delle figure")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    uscita = Path(args.uscita)
    uscita.mkdir(parents=True, exist_ok=True)
    dati = {v: leggi(args.cartella, v) for v in args.varianti}
    figura_riassuntiva(dati, uscita, args.dpi)
    figure_singole(dati, uscita, args.dpi)
    figura_medie(dati, uscita, args.dpi)


if __name__ == "__main__":
    main()
