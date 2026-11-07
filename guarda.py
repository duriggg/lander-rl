"""Mostra in una finestra il lander guidato da un modello addestrato.

Uso:
    python guarda.py --modello A                      (models/A/seed_0.zip, 5 episodi)
    python guarda.py --modello B --seed 3             (models/B/seed_3.zip)
    python guarda.py --modello A --episodi 10 --primo-seed 2000

Per ogni episodio stampa passi, reward della variante, reward originale di
Gymnasium ed esito (v. env.py); alla fine un riepilogo degli esiti.
Senza --primo-seed gli episodi partono da un seed casuale, che viene stampato
per poter rivedere esattamente le stesse discese.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from stable_baselines3 import PPO

from env import crea_ambiente
from rewards import FUNZIONI_REWARD

CARTELLA_MODELLI = Path("models")


def percorso_modello(modello, seed):
    """models/<modello>/seed_<seed>.zip, oppure models/<modello>.zip."""
    if (CARTELLA_MODELLI / modello).is_dir():
        return CARTELLA_MODELLI / modello / f"seed_{seed}.zip"
    return CARTELLA_MODELLI / f"{modello}.zip"


def modelli_disponibili():
    nomi = [p.name for p in CARTELLA_MODELLI.iterdir() if p.is_dir()]
    nomi += [p.stem for p in CARTELLA_MODELLI.glob("*.zip")]
    return sorted(nomi)


def main():
    parser = argparse.ArgumentParser(
        description="Mostra il lander guidato da un modello addestrato.")
    parser.add_argument("--modello", required=True,
                        help="variante (A, B, C) o nome di un file .zip in models/")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed di addestramento del modello (default 0)")
    parser.add_argument("--episodi", type=int, default=5,
                        help="numero di episodi da mostrare (default 5)")
    parser.add_argument("--primo-seed", type=int, default=None,
                        help="seed del primo episodio, poi +1, +2, ... (default casuale)")
    args = parser.parse_args()

    file_modello = percorso_modello(args.modello, args.seed)
    if not file_modello.exists():
        sys.exit(f"Modello non trovato: {file_modello}\n"
                 f"Disponibili in {CARTELLA_MODELLI}/: {', '.join(modelli_disponibili())}")

    # La reward della variante serve solo per stamparla; per modelli senza
    # variante si usa la A, che basta a classificare l'esito.
    funzione_reward = FUNZIONI_REWARD.get(args.modello, FUNZIONI_REWARD["A"])
    env = crea_ambiente(funzione_reward, render_mode="human")

    model = PPO.load(file_modello, device="cpu")
    primo_seed = args.primo_seed if args.primo_seed is not None else random.randint(0, 99_999)
    print(f"Modello {file_modello}, {args.episodi} episodi, seed {primo_seed}..."
          f"{primo_seed + args.episodi - 1}")

    esiti = Counter()
    try:
        for i in range(args.episodi):
            seed = primo_seed + i
            obs, _ = env.reset(seed=seed)
            totale, totale_gym = 0.0, 0.0
            while True:
                azione, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(azione))
                totale += reward
                totale_gym += info["reward_gym"]
                if terminated or truncated:
                    break
            esiti[info["esito"]] += 1
            vel = info["vel_impatto"]
            print(f"episodio {i + 1} (seed {seed}): {info['passi']:4d} passi  "
                  f"reward {totale:8.1f}  reward gym {totale_gym:7.1f}  "
                  f"esito {info['esito']:<18} "
                  f"vel. impatto {'-' if vel is None else f'{vel:.2f}'}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        env.close()

    if esiti:
        n = sum(esiti.values())
        print("Riepilogo: " + "  ".join(f"{e} {c}/{n}" for e, c in esiti.most_common()))


if __name__ == "__main__":
    main()
