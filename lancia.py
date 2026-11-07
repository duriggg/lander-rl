"""Lancia gli addestramenti di una variante su più seed, alcuni in parallelo.

Uso:
    python lancia.py --variante C                    (10 seed, 1.000.000 di passi)
    python lancia.py --variante C --passi 200000     (prova breve)
    python lancia.py --variante C --seed 3 4         (solo alcuni seed)

Al massimo MAX_PARALLELI addestramenti girano insieme; appena uno finisce
parte il successivo. L'output di ognuno va in logs/<variante>_seed<seed>.txt;
a schermo compare un riepilogo quando ciascuno termina.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from rewards import FUNZIONI_REWARD

SEEDS = list(range(10))   # seed 0, 1, ..., 9
MAX_PARALLELI = 5         # lascia liberi alcuni core del PC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variante", required=True, choices=sorted(FUNZIONI_REWARD))
    parser.add_argument("--passi", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    da_avviare = list(args.seed)
    in_corso = {}
    inizio = time.time()

    while da_avviare or in_corso:
        # avvia nuovi addestramenti finché c'è posto
        while da_avviare and len(in_corso) < MAX_PARALLELI:
            seed = da_avviare.pop(0)
            log = open(f"logs/{args.variante}_seed{seed}.txt", "w")
            comando = [sys.executable, "-u", "train.py", "--variante", args.variante,
                       "--seed", str(seed), "--passi", str(args.passi)]
            in_corso[seed] = (subprocess.Popen(comando, stdout=log,
                                               stderr=subprocess.STDOUT), log)
            print(f"[{(time.time() - inizio) / 60:5.1f} min] avviato seed {seed}")

        time.sleep(5)
        for seed in list(in_corso):
            processo, log = in_corso[seed]
            codice = processo.poll()
            if codice is not None:
                log.close()
                del in_corso[seed]
                stato = "OK" if codice == 0 else f"ERRORE (codice {codice}, vedi il log)"
                print(f"[{(time.time() - inizio) / 60:5.1f} min] seed {seed} terminato: {stato}")

    print(f"Variante {args.variante} completata in {(time.time() - inizio) / 60:.1f} minuti.")


if __name__ == "__main__":
    main()
