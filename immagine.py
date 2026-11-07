"""Salva alcuni fotogrammi di una discesa come immagini PNG (per le figure della tesi).

Uso:
    python immagine.py                         (controllore euristico di Gymnasium)
    python immagine.py --modello A --seed 100  (modello models/A/seed_100.zip)
    python immagine.py --episodio 2000 --ogni 10 --fino-a 100

Le immagini vanno in results/fotogrammi/. La finestra viene disegnata a ogni
passo, anche se se ne salva solo una ogni N: così le particelle dei motori
invecchiano correttamente.
"""

import argparse
from pathlib import Path

import imageio
from gymnasium.envs.box2d.lunar_lander import heuristic

from env import crea_ambiente
from rewards import reward_A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modello", default=None,
                        help="variante (A, B, C); senza, guida il controllore euristico")
    parser.add_argument("--seed", type=int, default=100, help="seed di addestramento del modello")
    parser.add_argument("--episodio", type=int, default=0, help="seed della discesa")
    parser.add_argument("--ogni", type=int, default=10, help="salva un fotogramma ogni N passi")
    parser.add_argument("--fino-a", type=int, default=100, help="ultimo passo da salvare")
    args = parser.parse_args()

    env = crea_ambiente(reward_A, render_mode="rgb_array")
    if args.modello:
        from stable_baselines3 import PPO
        model = PPO.load(f"models/{args.modello}/seed_{args.seed}.zip", device="cpu")
        scegli = lambda obs: int(model.predict(obs, deterministic=True)[0])
        nome = f"{args.modello}_seed{args.seed}"
    else:
        scegli = lambda obs: heuristic(env.unwrapped, obs)
        nome = "euristico"

    cartella = Path("results/fotogrammi")
    cartella.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=args.episodio)
    passo = 0
    while True:
        fotogramma = env.render()
        if passo % args.ogni == 0 and passo <= args.fino_a:
            imageio.imwrite(cartella / f"{nome}_ep{args.episodio}_{passo:04d}.png", fotogramma)
        obs, _, terminated, truncated, info = env.step(scegli(obs))
        passo += 1
        if terminated or truncated:
            break
    env.close()
    print(f"Discesa di {passo} passi, esito {info['esito']}; immagini in {cartella}/")


if __name__ == "__main__":
    main()
