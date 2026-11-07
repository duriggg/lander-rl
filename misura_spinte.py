"""Misura il rapporto tra la spinta del motore principale e quella dei laterali.

Idea: a parità di stato iniziale, si confronta la velocità del lander dopo un
passo con un motore acceso e dopo un passo senza motori. La differenza è la
variazione di velocità dovuta al solo motore, proporzionale al suo impulso
(la massa del lander è costante).

Serve a fissare il peso del consumo dei laterali rispetto al principale (v. 2.4:
a parità di impulso specifico il consumo è proporzionale alla spinta).

Uso:  python misura_spinte.py
"""

import gymnasium as gym
import numpy as np

N_SEED = 20
AZIONI = {1: "laterale sinistro", 2: "principale", 3: "laterale destro"}


def velocita_dopo_un_passo(seed, azione):
    """Velocità del lander (unità di Box2D) dopo un passo con l'azione data."""
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    env.step(azione)
    v = np.array(env.unwrapped.lander.linearVelocity)
    env.close()
    return v


variazioni = {a: [] for a in AZIONI}
for seed in range(N_SEED):
    v_senza_motori = velocita_dopo_un_passo(seed, 0)
    for azione in AZIONI:
        dv = velocita_dopo_un_passo(seed, azione) - v_senza_motori
        variazioni[azione].append(np.linalg.norm(dv))

print(f"Variazione di velocità in un passo, media su {N_SEED} condizioni iniziali:")
for azione, nome in AZIONI.items():
    x = np.array(variazioni[azione])
    print(f"  {nome:18s} {x.mean():.4f}  (dev. std. {x.std():.4f})")

principale = np.mean(variazioni[2])
laterale = np.mean(variazioni[1] + variazioni[3])
print(f"\nRapporto principale / laterale: {principale / laterale:.2f}")
print(f"Laterale come frazione del principale: {laterale / principale:.3f}")
