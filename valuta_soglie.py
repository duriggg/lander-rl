"""Atterraggi riusciti che rispettano anche i limiti di contatto di un lander reale.

Per ogni modello si eseguono N_EPISODI discese (policy deterministica, seed
10000, 10001, ...) e si registra lo stato del lander nell'ultimo passo prima del
primo contatto di una gamba con il suolo: velocità verticale, velocità
orizzontale, inclinazione e velocità angolare. Un atterraggio riuscito (criterio
di env.py) è "conforme" se tutte e quattro le grandezze sono entro le soglie.

Soglie: limiti di progetto del carrello del Lunar Module Apollo (Rogers, 1972,
NASA TN D-6850), riportati alla scala del simulatore con la similitudine di
Froude (stesso rapporto tra forze d'inerzia e peso):
    velocità:          v_sim = v_reale · sqrt(g_sim · L_sim / (g_reale · L_reale))
    velocità angolare: ω_sim = ω_reale · sqrt((g_sim / L_sim) / (g_reale / L_reale))
    angoli:            invariati
con L = distanza tra i piedi del carrello. Le soglie sono poi convertite nelle
unità dell'osservazione di LunarLander.

Uso:
    python valuta_soglie.py --varianti A B C --seed 22 23 24 25 26 27 28 29 30 31
"""

import argparse
import csv
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from gymnasium.envs.box2d.lunar_lander import FPS, SCALE, VIEWPORT_H, VIEWPORT_W, heuristic

from env import SUCCESSO, crea_ambiente
from rewards import FUNZIONI_REWARD, reward_A

N_EPISODI = 1000
PRIMO_SEED_EPISODI = 10_000

# --- Limiti reali: Apollo LM (Rogers, 1972, NASA TN D-6850, Appendice A) ------
FT = 0.3048
V_VERT_REALE = 10 * FT                 # 3,05 m/s
V_ORIZ_REALE = 4 * FT                  # 1,22 m/s (limite per v_vert <= 7 ft/s)
ANGOLO_REALE = math.radians(6)         # 6° tra asse del LM e verticale
OMEGA_REALE = math.radians(2)          # 2°/s
G_REALE = 1.62                         # gravità lunare, m/s²
L_REALE = 2 * 167.57 * 0.0254          # piedi opposti a 167,57 in dall'asse: 8,51 m

# --- Simulatore (misurati in LunarLander-v3) ----------------------------------
G_SIM = 10.0                           # gravità del mondo Box2D
L_SIM = 1.716                          # distanza tra i piedi a lander fermo

K_V = math.sqrt(G_SIM * L_SIM / (G_REALE * L_REALE))       # scala delle velocità
K_OMEGA = math.sqrt((G_SIM / L_SIM) / (G_REALE / L_REALE))  # scala delle vel. angolari

# conversione nelle unità dell'osservazione (v. codice di LunarLander)
OBS_VX = (VIEWPORT_W / SCALE / 2) / FPS    # 0,2
OBS_VY = (VIEWPORT_H / SCALE / 2) / FPS    # 0,133
OBS_OMEGA = 20.0 / FPS                     # 0,4

SOGLIE = {
    "v_vert": V_VERT_REALE * K_V * OBS_VY,
    "v_oriz": V_ORIZ_REALE * K_V * OBS_VX,
    "angolo": ANGOLO_REALE,
    "omega": OMEGA_REALE * K_OMEGA * OBS_OMEGA,
}
GRANDEZZE = list(SOGLIE)
CARTELLA = Path("results/valutazione_soglie")


def contatto(obs):
    return {"v_vert": abs(obs[3]), "v_oriz": abs(obs[2]),
            "angolo": abs(obs[4]), "omega": abs(obs[5])}


def valuta(compito):
    variante, seed, n_episodi = compito
    torch.set_num_threads(1)
    env = crea_ambiente(FUNZIONI_REWARD.get(variante, reward_A))
    if variante == "euristico":
        scegli = lambda obs: heuristic(env.unwrapped, obs)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(f"models/{variante}/seed_{seed}.zip", device="cpu")
        scegli = lambda obs: int(model.predict(obs, deterministic=True)[0])

    righe = []
    for i in range(n_episodi):
        obs, _ = env.reset(seed=PRIMO_SEED_EPISODI + i)
        stato_contatto = None
        while True:
            prec = obs
            obs, _, terminated, truncated, info = env.step(scegli(obs))
            if stato_contatto is None and (obs[6] == 1.0 or obs[7] == 1.0):
                stato_contatto = contatto(prec)    # ultimo passo prima del contatto
            if terminated or truncated:
                break
        riga = {"variante": variante, "seed": seed, "episodio": PRIMO_SEED_EPISODI + i,
                "esito": info["esito"]}
        for g in GRANDEZZE:
            riga[g] = "" if stato_contatto is None else round(float(stato_contatto[g]), 4)
        riga["consumo_equivalente"] = round(info["accensioni_principale"]
                                            + info["accensioni_laterali"] / 7, 2)
        riga["conforme"] = (info["esito"] == SUCCESSO and stato_contatto is not None
                            and all(stato_contatto[g] <= SOGLIE[g] for g in GRANDEZZE))
        righe.append(riga)
    env.close()
    return righe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--varianti", nargs="+", required=True)
    parser.add_argument("--seed", type=int, nargs="+", required=True)
    parser.add_argument("--episodi", type=int, default=N_EPISODI)
    parser.add_argument("--processi", type=int, default=2)
    args = parser.parse_args()

    print("Soglie al contatto (unità dell'osservazione):")
    print(f"  fattore di scala velocità K_v = {K_V:.3f}, vel. angolari K_ω = {K_OMEGA:.2f}")
    for g, s in SOGLIE.items():
        print(f"  {g:7s} <= {s:.4f}")

    compiti = [(v, s, args.episodi) for v in args.varianti for s in args.seed]
    compiti.append(("euristico", 0, args.episodi))          # riferimento
    with Pool(args.processi) as pool:
        risultati = [r for gruppo in pool.map(valuta, compiti) for r in gruppo]

    CARTELLA.mkdir(parents=True, exist_ok=True)
    with open(CARTELLA / "discese.csv", "w", newline="") as f:
        scrittore = csv.DictWriter(f, fieldnames=list(risultati[0]))
        scrittore.writeheader()
        scrittore.writerows(risultati)

    print(f"\n{args.episodi} discese per modello")
    for v in args.varianti + ["euristico"]:
        tutte = [r for r in risultati if r["variante"] == v]
        ok = [r for r in tutte if r["esito"] == SUCCESSO]
        print(f"\n{v}: {len(ok)} atterraggi riusciti su {len(tutte)} discese")
        if not ok:
            continue
        print(f"  consumo equivalente medio degli atterraggi riusciti: "
              f"{np.mean([r['consumo_equivalente'] for r in ok]):.1f}")
        conformi = sum(r["conforme"] for r in ok)
        print(f"  conformi a tutte le soglie: {conformi}/{len(ok)} = {conformi / len(ok):.1%}")
        for g in GRANDEZZE:
            dentro = sum(r[g] != "" and r[g] <= SOGLIE[g] for r in ok)
            valori = [r[g] for r in ok if r[g] != ""]
            print(f"  {g:7s} entro soglia {dentro / len(ok):6.1%}   "
                  f"mediana {np.median(valori):.4f}   95° perc. {np.percentile(valori, 95):.4f}")
        for s in sorted({r["seed"] for r in ok}):
            del_seed = [r for r in ok if r["seed"] == s]
            c = sum(r["conforme"] for r in del_seed)
            print(f"    seed {s}: {c}/{len(del_seed)} conformi")


if __name__ == "__main__":
    main()
