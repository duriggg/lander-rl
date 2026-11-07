"""Valutazione finale dei modelli addestrati (v. tesi 4.4).

Ogni modello guida il lander, con policy deterministica, su N_EPISODI discese
con seed fissi (gli stessi per tutti i modelli, diversi da quelli usati in
addestramento e nelle valutazioni periodiche). Per ogni discesa si registrano
esito, durata, accensioni dei motori e velocità d'impatto.

Uso:
    python evaluate.py --varianti A B C --seed 22 23 24 25 26 27 28 29 30 31
    python evaluate.py --varianti C --seed 28 --episodi 100        (prova breve)

Produce:
    results/valutazione/<variante>_seed<seed>.csv   una riga per discesa
    results/valutazione/riepilogo.csv               una riga per modello
e stampa la media (± deviazione standard sui seed) per ogni variante.
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from env import FUORI_AREA, FUORI_PIATTAFORMA, SCHIANTO, SUCCESSO, TIMEOUT, crea_ambiente
from rewards import FUNZIONI_REWARD

N_EPISODI = 1000
PRIMO_SEED_EPISODI = 10_000      # discese 10000, 10001, ..., 10999
ESITI = [SUCCESSO, FUORI_PIATTAFORMA, SCHIANTO, FUORI_AREA, TIMEOUT]
CARTELLA = Path("results/valutazione")


def valuta_modello(compito):
    variante, seed, n_episodi = compito
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    model = PPO.load(f"models/{variante}/seed_{seed}.zip", device="cpu")
    env = crea_ambiente(FUNZIONI_REWARD[variante])
    righe = []
    for i in range(n_episodi):
        seed_episodio = PRIMO_SEED_EPISODI + i
        obs, _ = env.reset(seed=seed_episodio)
        ritorno = 0.0
        while True:
            azione, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(azione))
            ritorno += reward
            if terminated or truncated:
                break
        righe.append({
            "episodio": seed_episodio,
            "esito": info["esito"],
            "passi": info["passi"],
            "accensioni_principale": info["accensioni_principale"],
            "accensioni_laterali": info["accensioni_laterali"],
            "vel_impatto": "" if info["vel_impatto"] is None else round(info["vel_impatto"], 4),
            "x_finale": round(info["x_finale"], 4),
            "ritorno": round(ritorno, 2),
        })
    env.close()

    with open(CARTELLA / f"{variante}_seed{seed}.csv", "w", newline="") as f:
        scrittore = csv.DictWriter(f, fieldnames=list(righe[0]))
        scrittore.writeheader()
        scrittore.writerows(righe)

    esiti = [r["esito"] for r in righe]
    successi = [r for r in righe if r["esito"] == SUCCESSO]
    riepilogo = {"variante": variante, "seed": seed, "episodi": n_episodi}
    for e in ESITI:
        riepilogo[f"tasso_{e}"] = esiti.count(e) / n_episodi
    riepilogo["passi_medi"] = np.mean([r["passi"] for r in righe])
    riepilogo["principale_medio"] = np.mean([r["accensioni_principale"] for r in righe])
    riepilogo["laterali_medi"] = np.mean([r["accensioni_laterali"] for r in righe])
    riepilogo["principale_medio_successi"] = (
        np.mean([r["accensioni_principale"] for r in successi]) if successi else float("nan"))
    vel = [r["vel_impatto"] for r in righe if r["vel_impatto"] != ""]
    riepilogo["vel_impatto_media"] = np.mean(vel) if vel else float("nan")
    print(f"{variante} seed {seed}: successo {riepilogo['tasso_successo']:.1%}", flush=True)
    return riepilogo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--varianti", nargs="+", required=True)
    parser.add_argument("--seed", type=int, nargs="+", required=True)
    parser.add_argument("--episodi", type=int, default=N_EPISODI)
    parser.add_argument("--processi", type=int, default=5)
    args = parser.parse_args()

    CARTELLA.mkdir(parents=True, exist_ok=True)
    compiti = [(v, s, args.episodi) for v in args.varianti for s in args.seed]
    with Pool(args.processi) as pool:
        riepiloghi = pool.map(valuta_modello, compiti)

    with open(CARTELLA / "riepilogo.csv", "w", newline="") as f:
        scrittore = csv.DictWriter(f, fieldnames=list(riepiloghi[0]))
        scrittore.writeheader()
        scrittore.writerows(riepiloghi)

    colonne = [f"tasso_{e}" for e in ESITI] + ["passi_medi", "principale_medio",
                                              "laterali_medi", "vel_impatto_media"]
    print(f"\nMedia sui seed (± dev. std), {args.episodi} discese per modello")
    for v in args.varianti:
        gruppo = [r for r in riepiloghi if r["variante"] == v]
        print(f"\nVariante {v} ({len(gruppo)} modelli)")
        for c in colonne:
            valori = np.array([r[c] for r in gruppo], dtype=float)
            if c.startswith("tasso"):
                print(f"  {c[6:]:24s} {np.mean(valori):6.1%} ± {np.std(valori):5.1%}")
            else:
                print(f"  {c:24s} {np.nanmean(valori):7.2f} ± {np.nanstd(valori):6.2f}")


if __name__ == "__main__":
    main()
