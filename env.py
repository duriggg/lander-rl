"""Ambiente LunarLander-v3 con reward sostituita ed esito di ogni episodio.

Il wrapper fa due cose a ogni passo:
1. calcola la reward con la funzione scelta (variante A, B o C, in rewards.py)
   al posto di quella predefinita di Gymnasium;
2. a fine episodio classifica l'esito con un criterio unico (v. tesi, 4.4)
   e raccoglie le metriche indipendenti dalla reward.
"""

import gymnasium as gym

# Bordi della piattaforma nelle coordinate dell'osservazione (le bandierine).
# Dal codice di LunarLander: piattaforma tra x = 8 e x = 12 su una larghezza
# di 20, centro in 10, quindi |x| <= 0.2 una volta normalizzato.
X_PIATTAFORMA = 0.2

# Limite superiore dell'area. Gymnasium chiude l'episodio solo se il lander
# esce dai lati; senza questo limite può salire oltre la finestra e restarci
# fino al timeout. Il lander parte a y = 1,40-1,42 (bordo superiore della
# finestra: 1,41) e la sola spinta iniziale casuale può alzarlo fino a 1,53,
# quindi il limite è posto poco sopra, a 1,6.
Y_MAX = 1.6

# Possibili esiti di un episodio
SUCCESSO = "successo"                # a riposo sulla piattaforma, senza schianto
FUORI_PIATTAFORMA = "fuori_piattaforma"  # a riposo, ma fuori dalla piattaforma
SCHIANTO = "schianto"                # il corpo del lander tocca il suolo
FUORI_AREA = "fuori_area"            # uscito lateralmente (|x| >= 1) o dall'alto (y >= Y_MAX)
TIMEOUT = "timeout"                  # 1000 passi senza nessuno dei casi sopra


def classifica_esito(obs, terminated, truncated, game_over):
    """Restituisce l'esito dell'episodio, oppure None se non è finito."""
    x = obs[0]

    if truncated and not terminated:
        return TIMEOUT
    if not terminated:
        return None
    if game_over:
        return SCHIANTO
    if abs(x) >= 1.0 or obs[1] >= Y_MAX:
        return FUORI_AREA
    # Rimane un solo caso: il lander è fermo a riposo al suolo, senza che il
    # corpo l'abbia toccato, quindi poggia sulle gambe. Non si usano i flag
    # c_L, c_R dell'ultimo passo: a riposo il motore fisico può azzerarli.
    if abs(x) <= X_PIATTAFORMA:
        return SUCCESSO
    return FUORI_PIATTAFORMA


class LanderRewardWrapper(gym.Wrapper):
    """Sostituisce la reward di LunarLander e registra l'esito dell'episodio.

    funzione_reward(obs_prec, azione, obs, esito) -> float
    `esito` vale None durante l'episodio e una delle costanti sopra all'ultimo passo.
    """

    def __init__(self, env, funzione_reward):
        super().__init__(env)
        self.funzione_reward = funzione_reward

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_prec = obs
        self.passi = 0
        self.accensioni_principale = 0
        self.accensioni_laterali = 0
        self.vel_impatto = None      # velocità al primo contatto di una gamba
        self.contatto_sx = None      # passo del primo contatto della gamba sinistra
        self.contatto_dx = None      # passo del primo contatto della gamba destra
        return obs, info

    def step(self, azione):
        obs, reward_gym, terminated, truncated, info = self.env.step(azione)
        self.passi += 1
        if obs[1] >= Y_MAX:          # uscito dall'alto: fine episodio (fuori area)
            terminated = True

        # Proxy del consumo: passi con un motore acceso (carburante illimitato)
        if azione == 2:
            self.accensioni_principale += 1
        elif azione in (1, 3):
            self.accensioni_laterali += 1

        # Velocità d'impatto: modulo della velocità al primo contatto con il suolo
        if self.vel_impatto is None and (obs[6] == 1.0 or obs[7] == 1.0):
            self.vel_impatto = float((obs[2] ** 2 + obs[3] ** 2) ** 0.5)

        # Requisito (e): passo del primo contatto di ciascuna gamba
        if self.contatto_sx is None and obs[6] == 1.0:
            self.contatto_sx = self.passi
        if self.contatto_dx is None and obs[7] == 1.0:
            self.contatto_dx = self.passi

        esito = classifica_esito(obs, terminated, truncated,
                                 self.env.unwrapped.game_over)
        reward = self.funzione_reward(self.obs_prec, azione, obs, esito)
        self.obs_prec = obs

        info["reward_gym"] = reward_gym   # la reward originale, solo per confronto
        if esito is not None:
            info["esito"] = esito
            info["passi"] = self.passi
            info["accensioni_principale"] = self.accensioni_principale
            info["accensioni_laterali"] = self.accensioni_laterali
            info["vel_impatto"] = self.vel_impatto
            info["x_finale"] = float(obs[0])
            info["angolo_finale"] = float(obs[4])
            # Passi tra il contatto della prima e della seconda gamba (None se
            # una gamba non ha mai toccato): misura la simultaneità, req. (e)
            if self.contatto_sx is not None and self.contatto_dx is not None:
                info["ritardo_gambe"] = abs(self.contatto_sx - self.contatto_dx)
            else:
                info["ritardo_gambe"] = None
            # Distingue il lander che galleggia da quello già a terra ma non fermo
            info["timeout_a_terra"] = esito == TIMEOUT and obs[6] == 1.0 and obs[7] == 1.0

        # Nel problema definito in tesi il timeout è un esito (insuccesso), non
        # un'interruzione artificiale: lo si segnala come fine vera dell'episodio,
        # così PPO non stima un "seguito" che non esiste.
        if esito == TIMEOUT:
            terminated, truncated = True, False
        return obs, reward, terminated, truncated, info


def crea_ambiente(funzione_reward, render_mode=None):
    """Crea LunarLander-v3 (configurazione predefinita) con la reward scelta."""
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    return LanderRewardWrapper(env, funzione_reward)
