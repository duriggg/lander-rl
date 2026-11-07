"""Riepilogo rapido delle curve di addestramento di una variante.

Per ogni seed stampa l'ultima valutazione periodica (20 discese fisse) e il
miglior tasso di successo raggiunto durante l'addestramento.

Uso:  python riepilogo.py --variante C
"""

import argparse
import csv
from pathlib import Path

ESITI = ["successo", "fuori_piattaforma", "schianto", "fuori_area", "timeout"]

parser = argparse.ArgumentParser()
parser.add_argument("--variante", required=True)
args = parser.parse_args()

file_curve = sorted(Path("results/curve").glob(f"{args.variante}_seed*.csv"),
                    key=lambda p: int(p.stem.split("seed")[1]))
if not file_curve:
    raise SystemExit(f"Nessuna curva trovata per la variante {args.variante}")

print(f"Variante {args.variante}: ultima valutazione (20 discese) e miglior successo")
print(f"{'seed':>4}  " + "  ".join(f"{e[:10]:>10}" for e in ESITI)
      + f"  {'passi':>6}  {'princ.':>6}  {'max succ.':>9}")
finali = []
for f in file_curve:
    righe = list(csv.DictReader(open(f)))
    ultima = righe[-1]
    massimo = max(float(r["tasso_successo"]) for r in righe)
    finali.append(ultima)
    print(f"{f.stem.split('seed')[1]:>4}  "
          + "  ".join(f"{float(ultima['tasso_' + e]):>10.0%}" for e in ESITI)
          + f"  {float(ultima['passi_medi']):>6.0f}  {float(ultima['accensioni_principale_medie']):>6.0f}"
          + f"  {massimo:>9.0%}")

print("\nMedia sui seed dell'ultima valutazione:")
for e in ESITI:
    media = sum(float(r["tasso_" + e]) for r in finali) / len(finali)
    print(f"  {e:18s} {media:.0%}")
