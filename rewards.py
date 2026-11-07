"""Le funzioni di reward confrontate nella tesi (v. 2.5 e 4.3).

Ogni funzione ha la stessa firma:
    reward(obs_prec, azione, obs, esito) -> float
- obs_prec, obs: stato prima e dopo il passo, (x, y, vx, vy, θ, ω, cL, cR)
- azione: 0 nessun motore, 1 laterale sinistro, 2 principale, 3 laterale destro
- esito: None durante l'episodio, altrimenti l'esito finale (v. env.py)
"""

import math

from env import FUORI_AREA, FUORI_PIATTAFORMA, SCHIANTO, SUCCESSO, TIMEOUT

# Fattore di sconto di PPO (train.py). Lo shaping della variante B non lo usa:
# è la differenza semplice Φ(s') − Φ(s) (v. shaping più sotto).
GAMMA = 0.999

# --- Variante A: reward sparsa -------------------------------------------
# Reward diversa da zero solo a fine episodio (v. 2.5.1).
# Ogni fallimento vale -200: il doppio del premio di atterraggio, così uno
# schianto non viene compensato dallo shaping guadagnato cadendo verso la
# piattaforma (varianti B e C). Con -100 l'agente della variante C imparava
# a lasciarsi cadere a motori spenti (prove pilota).
PREMI_FINALI_A = {
    SUCCESSO: +100.0,
    SCHIANTO: -200.0,
    FUORI_AREA: -200.0,
    FUORI_PIATTAFORMA: -200.0,
    TIMEOUT: -200.0,
}


def reward_A(obs_prec, azione, obs, esito):
    if esito is None:
        return 0.0
    return PREMI_FINALI_A[esito]


# --- Variante B: A + shaping potential-based (v. 2.5.2) -------------------
# Il peso di ogni termine si ottiene in due fasi successive (v. calcola_pesi.py):
#   peso = coefficiente di normalizzazione (fase 1) × coefficiente di importanza (fase 2)
# Fase 1: 100 / valore tipico della grandezza (95° percentile su 1000 discese del
#         controllore euristico): ogni termine vale 100 al suo valore tipico.
# Fase 2: c · valore tipico / tolleranza (criterio di Bryson). Tolleranze:
#         distanza 0,200 (piattaforma); velocità 0,593, |θ| 1,252 rad, |ω| 11,87
#         (rischio di schianto del 50% al primo contatto). c = 0,1259, scelto
#         perché il potenziale iniziale valga in media -100 (l'opposto del
#         premio di atterraggio).
# Gamba a terra: binaria, valore tipico 1 e importanza I = 1.
COEFF_NORMALIZZAZIONE = {"distanza": 72.1, "velocita": 101.7, "theta": 285.3,
                         "omega": 396.0, "gamba": 100.0}
COEFF_IMPORTANZA = {"distanza": 0.8735, "velocita": 0.2087, "theta": 0.0353,
                    "omega": 0.0027, "gamba": 0.1259}
PESI_SHAPING = {k: COEFF_NORMALIZZAZIONE[k] * COEFF_IMPORTANZA[k]
                for k in COEFF_NORMALIZZAZIONE}
# risultato: distanza 63,0 · velocità 21,2 · |θ| 10,1 · |ω| 1,1 · gamba 12,6


def potenziale(obs):
    """Φ(s): quanto è favorevole lo stato. È massimo (vicino a +20) con il lander
    fermo, dritto e appoggiato sulla piattaforma; diminuisce allontanandosene."""
    x, y, vx, vy, theta, omega, gamba_sx, gamba_dx = obs
    p = PESI_SHAPING
    return (- p["distanza"] * math.hypot(x, y)
            - p["velocita"] * math.hypot(vx, vy)
            - p["theta"] * abs(theta)
            - p["omega"] * abs(omega)
            + p["gamba"] * (gamba_sx + gamba_dx))


def shaping(obs_prec, obs, esito):
    """F = Φ(s') − Φ(s): variazione del potenziale tra un passo e il successivo,
    calcolata a ogni passo, compreso l'ultimo.

    - Il potenziale dello stato finale NON viene azzerato: la somma dello
      shaping su un episodio vale Φ(s_finale) − Φ(s₀), quindi conta anche come
      finisce la discesa (uno schianto veloce lontano dalla piattaforma pesa
      più di un appoggio lento al centro).
    - Si usa la differenza semplice, senza il fattore γ: restare fermi in uno
      stato dà shaping nullo (con γ < 1 e Φ negativo darebbe invece un piccolo
      premio a ogni passo, cioè un incentivo a galleggiare).
    Si rinuncia così alla garanzia formale di Ng et al. (1999) in cambio di un
    segnale più informativo (scelta di progetto, v. tesi 4.3).
    """
    return potenziale(obs) - potenziale(obs_prec)


def reward_B(obs_prec, azione, obs, esito):
    return reward_A(obs_prec, azione, obs, esito) + shaping(obs_prec, obs, esito)


# --- Variante C: B + consumo, multi-obiettivo (v. 2.5.3) -----------------
# Penalità per ogni passo con un motore acceso (proxy del propellente, v. 2.4).
# Principale: un atterraggio efficiente (≈ 94 accensioni, calcola_pesi.py)
# costa il 15% del premio di atterraggio → 0,15 · 100 / 94 ≈ 0,160.
# (Prove pilota con il limite superiore: al 30% 3 seed su 5 imparavano a
#  lasciarsi cadere a motori spenti; al 15% gli schianti quasi spariscono.)
# Laterale: 1/7 del principale, rapporto delle spinte (misura_spinte.py).
PESI_CONSUMO = {
    2: 0.160,           # motore principale
    1: 0.160 / 7,       # laterale sinistro (≈ 0,023)
    3: 0.160 / 7,       # laterale destro
}


def consumo(azione):
    return PESI_CONSUMO.get(int(azione), 0.0)


def reward_C(obs_prec, azione, obs, esito):
    return reward_B(obs_prec, azione, obs, esito) - consumo(azione)


# Elenco delle varianti disponibili, usato da train.py e lancia.py
FUNZIONI_REWARD = {
    "A": reward_A,
    "B": reward_B,
    "C": reward_C,
}
