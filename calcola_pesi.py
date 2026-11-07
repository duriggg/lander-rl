"""Calcolo dei pesi delle varianti B e C a partire da misure sul simulatore.

Il peso di ogni termine dello shaping si ottiene in due fasi successive:

    peso finale = coefficiente di normalizzazione (fase 1)
                  × coefficiente di importanza (fase 2)

Fase 1 - Normalizzazione
    Con il controllore euristico di Gymnasium (regola fissa, non appresa) si
    misura il valore tipico (95° percentile) di distanza, velocità, |θ|, |ω|
    durante la discesa. Il coefficiente di normalizzazione è 100 / valore tipico:
    moltiplicato per la grandezza, porta ogni termine a valere 100 al suo
    valore tipico, cioè lo rende confrontabile con gli altri e con il premio
    finale (±100). Per la gamba a terra (binaria, valore tipico 1) vale 100.

Fase 2 - Importanza (criterio di Bryson, adattato a termini lineari)
    Per ogni grandezza si stabilisce la tolleranza, cioè il valore massimo
    accettabile:
    - distanza: semiampiezza della piattaforma (le bandierine, v. env.py);
    - velocità, |θ|, |ω|: si raccoglie lo stato al PRIMO CONTATTO con il suolo
      su molte discese (controllore euristico con una quota crescente di azioni
      casuali, per avere sia appoggi riusciti sia schianti) e si stima con una
      regressione logistica la probabilità di schianto in funzione delle tre
      grandezze. La tolleranza è il valore a cui la probabilità di schianto
      arriva al 50%, con le altre due ai valori mediani di un appoggio riuscito.
    Il coefficiente di importanza è  c · valore tipico / tolleranza:
    - valore tipico / tolleranza (I) dice quanto la grandezza, nel volo tipico,
      si avvicina al suo limite accettabile (I > 1 conta molto, I < 1 poco);
    - c è un fattore di scala comune, scelto perché in media sugli stati
      iniziali il potenziale valga -VALORE_INIZIALE_PHI.
    Per la gamba a terra (senza tolleranza) I = 1, quindi il coefficiente è c.
    Il peso finale risulta c · 100 / tolleranza: ogni termine vale lo stesso
    valore quando la sua grandezza raggiunge la tolleranza (regola di Bryson).

Consumo (variante C): rapporto 1/7 tra laterale e principale
    (misura_spinte.py) e quota del consumo di un atterraggio efficiente.

Uso:  python calcola_pesi.py
"""

import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

from env import SCHIANTO, X_PIATTAFORMA, crea_ambiente
from rewards import reward_A

# --- Parametri (scelte di progetto, da riportare in tesi) -------------------
N_EPISODI_NORMALIZZAZIONE = 1000
PERCENTILE = 95
QUOTE_CASUALI = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # quota di azioni casuali
N_EPISODI_PER_QUOTA = 2000
PROBABILITA_SOGLIA = 0.5          # tolleranza = rischio di schianto del 50%
VALORE_INIZIALE_PHI = 100         # |Φ(s₀)| medio desiderato
TOLLERANZE_DICHIARATE = {}        # es. {"omega": 0.5} per fissarne una a mano
QUOTA_CONSUMO = 0.15
PREMIO_ATTERRAGGIO = 100
RAPPORTO_LATERALE = 1 / 7         # da misura_spinte.py

env = crea_ambiente(reward_A)
rng = np.random.default_rng(0)
GRANDEZZE = ["distanza", "velocita", "theta", "omega"]


def grandezze(obs):
    x, y, vx, vy, theta, omega = obs[:6]
    return {"distanza": np.hypot(x, y), "velocita": np.hypot(vx, vy),
            "theta": abs(theta), "omega": abs(omega)}


# --- Fase 1: normalizzazione -------------------------------------------------
valori = {g: [] for g in GRANDEZZE}
accensioni, iniziali = [], []
for seed in range(N_EPISODI_NORMALIZZAZIONE):
    obs, _ = env.reset(seed=seed)
    iniziali.append(grandezze(obs))
    while True:
        for g, v in grandezze(obs).items():
            valori[g].append(v)
        obs, _, terminated, truncated, info = env.step(heuristic(env.unwrapped, obs))
        if terminated or truncated:
            break
    accensioni.append(info["accensioni_principale"])
tipici = {g: np.percentile(valori[g], PERCENTILE) for g in GRANDEZZE}

# --- Fase 2: stato al primo contatto e rischio di schianto -------------------
contatti, schianti = [], []
for k, quota in enumerate(QUOTE_CASUALI):
    for seed in range(N_EPISODI_PER_QUOTA):
        obs, _ = env.reset(seed=100_000 + 10_000 * k + seed)
        primo = None
        while True:
            if rng.random() < quota:
                azione = int(rng.integers(4))   # azione casuale riproducibile
            else:
                azione = heuristic(env.unwrapped, obs)
            obs, _, terminated, truncated, info = env.step(azione)
            if primo is None and (obs[6] or obs[7] or env.unwrapped.game_over):
                g = grandezze(obs)
                primo = [g["velocita"], g["theta"], g["omega"]]
            if terminated or truncated:
                break
        if primo is not None:
            contatti.append(primo)
            schianti.append(info["esito"] == SCHIANTO)
X, y = np.array(contatti), np.array(schianti, dtype=float)

# regressione logistica P(schianto) = σ(b0 + b1·|v| + b2·|θ| + b3·|ω|)
A = np.c_[np.ones(len(X)), X]
b = np.zeros(A.shape[1])
for _ in range(100):                       # metodo di Newton
    p = 1 / (1 + np.exp(-A @ b))
    H = A.T @ (A * (p * (1 - p))[:, None]) + 1e-8 * np.eye(len(b))
    b += np.linalg.solve(H, A.T @ (y - p))

mediane_ok = np.median(X[y == 0], axis=0)  # appoggio riuscito "tipico"
logit_soglia = np.log(PROBABILITA_SOGLIA / (1 - PROBABILITA_SOGLIA))
tolleranze = {"distanza": X_PIATTAFORMA}
for i, nome in enumerate(["velocita", "theta", "omega"]):
    altri = sum(b[1 + j] * mediane_ok[j] for j in range(3) if j != i)
    if b[1 + i] > 0:
        tolleranze[nome] = (logit_soglia - b[0] - altri) / b[1 + i]
    else:
        tolleranze[nome] = None              # nessun effetto misurabile
tolleranze.update(TOLLERANZE_DICHIARATE)

# Coefficienti delle due fasi e pesi finali
normalizzazione = {g: 100 / tipici[g] for g in GRANDEZZE}
normalizzazione["gamba"] = 100.0                     # binaria, valore tipico 1
importanza_I = {g: (tipici[g] / tolleranze[g] if tolleranze[g] else 0.0) for g in GRANDEZZE}
importanza_I["gamba"] = 1.0

# c: |Φ(s₀)| medio = VALORE_INIZIALE_PHI (all'inizio le gambe non toccano)
phi0_senza_c = np.mean([sum(normalizzazione[g] * importanza_I[g] * s0[g] for g in GRANDEZZE)
                        for s0 in iniziali])
c = VALORE_INIZIALE_PHI / phi0_senza_c
importanza = {g: c * importanza_I[g] for g in importanza_I}
pesi = {g: normalizzazione[g] * importanza[g] for g in importanza}
TERMINI = GRANDEZZE + ["gamba"]

# --- Stampa ------------------------------------------------------------------
print(f"Valori tipici ({PERCENTILE}° percentile, {N_EPISODI_NORMALIZZAZIONE} discese euristiche):")
for g in GRANDEZZE:
    print(f"  {g:9s} {tipici[g]:6.3f}")

print(f"\nTolleranze - {len(y)} primi contatti, schianti {y.mean():.0%}")
print(f"  regressione logistica: b0={b[0]:.2f}  b_v={b[1]:.2f}  b_theta={b[2]:.2f}  b_omega={b[3]:.2f}")
print(f"  appoggio riuscito mediano: |v|={mediane_ok[0]:.3f}  |θ|={mediane_ok[1]:.3f}  |ω|={mediane_ok[2]:.3f}")
for g in GRANDEZZE:
    origine = "piattaforma" if g == "distanza" else ("dichiarata" if g in TOLLERANZE_DICHIARATE else f"rischio di schianto {PROBABILITA_SOGLIA:.0%}")
    print(f"    {g:9s} {tolleranze[g]:.3f}  ({origine})")

print(f"\nFattore di scala c = {c:.4f}  (|Φ(s₀)| medio = {VALORE_INIZIALE_PHI})")
print(f"\n{'termine':9s} {'fase 1':>9s} {'I':>8s} {'fase 2':>9s} {'peso finale':>12s}")
for g in TERMINI:
    print(f"{g:9s} {normalizzazione[g]:9.1f} {importanza_I[g]:8.4f} {importanza[g]:9.4f} {pesi[g]:12.2f}")

print("\nDa copiare in rewards.py:")
print("COEFF_NORMALIZZAZIONE = {" + ", ".join(f'"{g}": {normalizzazione[g]:.1f}' for g in TERMINI) + "}")
print("COEFF_IMPORTANZA = {" + ", ".join(f'"{g}": {importanza[g]:.4f}' for g in TERMINI) + "}")
media_acc = np.mean(accensioni)
peso_principale = QUOTA_CONSUMO * PREMIO_ATTERRAGGIO / media_acc
print(f"\nConsumo (accensioni medie del principale {media_acc:.0f}):")
print(f"  principale {peso_principale:.3f}   laterale {peso_principale * RAPPORTO_LATERALE:.3f}")
