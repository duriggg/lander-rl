# lander-rl

Codice della tesi di laurea triennale *Reinforcement Learning per l'aerospazio: progettazione e confronto di funzioni di reward per lander con atterraggio autonomo*.

Un agente PPO impara ad atterrare nell'ambiente `LunarLander-v3` di Gymnasium. L'algoritmo e gli iperparametri restano fissi; cambia solo la funzione di reward, in tre varianti:

| Variante | Funzione di reward | Idea |
|---|---|---|
| A | reward sparso | +100 per l'atterraggio sulla piattaforma, −200 per ogni insuccesso (schianto, fuori piattaforma, fuori area, timeout) |
| B | A + shaping | aggiunge la variazione di un potenziale Φ(s′) − Φ(s) basato su distanza, velocità, inclinazione, velocità angolare e contatto delle gambe |
| C | B + consumo (multi-obiettivo) | sottrae una penalità per ogni passo con un motore acceso (principale 0,160, laterale 0,023) |

I pesi di B e C non sono scelti a mano: li calcola `calcola_pesi.py` a partire da misure sul simulatore (normalizzazione sui valori tipici e importanza con la regola di Bryson). Il metodo è descritto nelle appendici della tesi.

## Installazione

Testato su Windows con Python 3.12.10, su CPU.

```
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python smoke_test.py
```

`requirements.txt` fissa le versioni usate (Gymnasium 1.4.0, Stable-Baselines3 2.9.0, PyTorch 2.14.1 per CPU) e indica a pip l'indice da cui scaricare PyTorch. L'ultimo comando controlla che l'installazione funzioni.

## File

### Ambiente e funzioni di reward

| File | Cosa fa |
|---|---|
| `env.py` | Costruisce l'ambiente `LunarLander-v3` con la funzione di reward scelta al posto di quella di Gymnasium. A fine discesa classifica l'esito (successo, fuori piattaforma, schianto, fuori area, timeout) e registra le metriche usate nell'analisi: passi, accensioni dei motori, velocità d'impatto, posizione e angolo finali. Il timeout è trattato come fine dell'episodio. Aggiunge un limite superiore: se il lander sale oltre y = 1,6 l'episodio termina come fuori area (Gymnasium controlla solo l'uscita laterale). |
| `rewards.py` | Le funzioni di reward A, B e C, con i pesi ricavati da `calcola_pesi.py`. |

### Calcolo dei parametri

| File | Cosa fa |
|---|---|
| `calcola_pesi.py` | Calcola i pesi dello shaping (variante B) in due fasi: normalizzazione sui valori tipici di 1000 discese del controllore euristico di Gymnasium, e importanza secondo la regola di Bryson, con tolleranze stimate da una regressione logistica sul rischio di schianto. Calcola anche la penalità sul consumo (variante C). Stampa i risultati intermedi e i valori da copiare in `rewards.py`. |
| `misura_spinte.py` | Misura nel simulatore l'impulso del motore principale e quello dei laterali; il rapporto (≈ 7) fissa il costo dei laterali nella variante C. |

### Addestramento

| File | Cosa fa |
|---|---|
| `train.py` | Addestra PPO su una variante con un seed (iperparametri di RL Baselines3 Zoo). Ogni 50.000 passi, e una volta a fine addestramento sul modello finale, valuta la policy su 20 discese fisse; salva il modello in `models/` e la curva in `results/curve/`. |
| `lancia.py` | Lancia gli addestramenti di una variante su più seed, fino a 5 in parallelo; l'output di ciascuno va in `logs/`. |

### Analisi e figure

| File | Cosa fa |
|---|---|
| `evaluate.py` | Valutazione finale: ogni modello guida il lander su 1000 discese fisse (seed 10000–10999) con policy deterministica. Salva esito, durata, accensioni e velocità d'impatto di ogni discesa e stampa la media (± deviazione standard sui seed) per variante. |
| `valuta_soglie.py` | Verifica degli atterraggi riusciti rispetto ai limiti di contatto del Lunar Module Apollo (Rogers, 1972), riportati alla scala del simulatore con la similitudine di Froude (Appendice C della tesi). Sulle stesse 1000 discese di `evaluate.py` registra velocità verticale, velocità orizzontale, inclinazione e velocità angolare nell'ultimo passo prima del contatto, e il consumo equivalente (principale + laterali/7). Include il controllore euristico di Gymnasium come riferimento. |
| `riepilogo.py` | Tabella per seed di una variante: esiti dell'ultima valutazione periodica e miglior tasso di successo raggiunto. |
| `grafico_curve.py` | Curve di apprendimento in `results/figures/` a 300 dpi: per ogni variante la composizione degli esiti nel tempo (`esiti_<variante>.png`) e il successo di ogni seed con la media (`successo_<variante>.png`), il successo medio delle varianti sovrapposte (`successo_medio.png`) e una figura riassuntiva (`curve_ABC.png`). |
| `guarda.py` | Mostra in una finestra il lander guidato da un modello addestrato; per ogni discesa stampa reward, esito e velocità d'impatto. |
| `immagine.py` | Salva alcuni fotogrammi di una discesa come PNG, guidata dal controllore euristico o da un modello addestrato. |

### Controllo dell'installazione

| File | Cosa fa |
|---|---|
| `smoke_test.py` | Stampa le versioni delle librerie e fa una discesa con azioni casuali. |

## Riprodurre gli esperimenti

1. Calcolo dei pesi (i valori ottenuti sono già riportati in `rewards.py`):
   ```
   python calcola_pesi.py
   ```
2. Addestramento di una variante su 10 seed da 1.000.000 di passi:
   ```
   python lancia.py --variante A --seed 22 23 24 25 26 27 28 29 30 31
   ```
   Lo stesso per B e C. Per un singolo addestramento: `python train.py --variante A --seed 22`.
3. Valutazione finale (1000 discese per modello, policy deterministica):
   ```
   python evaluate.py --varianti A B C --seed 22 23 24 25 26 27 28 29 30 31
   python valuta_soglie.py --varianti A B C --seed 22 23 24 25 26 27 28 29 30 31
   ```
4. Riepilogo e grafici delle curve:
   ```
   python riepilogo.py --variante A
   python grafico_curve.py --varianti A B C
   ```
5. Visualizzazione di un modello:
   ```
   python guarda.py --modello C --seed 30
   ```

### Seed

| Uso | Seed |
|---|---|
| prove di sviluppo (piloti) | 0–4 |
| addestramenti finali | 22–31 |
| valutazione periodica durante l'addestramento (ogni 50.000 passi) | 2000–2019 |
| valutazione finale (`evaluate.py`, `valuta_soglie.py`) | 10000–10999 |

### Output

- `models/<variante>/seed_<seed>.zip`: modelli addestrati (non inclusi nel repository);
- `results/curve/<variante>_seed<seed>.csv`: valutazioni periodiche;
- `results/valutazione/`: valutazione finale, un file per modello (una riga per discesa) e `riepilogo.csv` (una riga per modello);
- `results/valutazione_soglie/discese.csv`: limiti di contatto e consumo equivalente, una riga per discesa;
- `prove_pilota/`: prove pilota (seed 0–4, 700.000 passi) usate per scegliere la penalità di fallimento, il limite superiore e la quota di consumo (v. `prove_pilota/LEGGIMI.txt`);
- `results/figures/`: grafici;
- `results/train/`: tutti gli episodi di addestramento (non inclusi nel repository);
- `logs/`: output di ogni addestramento lanciato con `lancia.py` (non incluso nel repository).

## Riferimenti

- Towers et al. (2025). *Gymnasium: A Standard Interface for Reinforcement Learning Environments*.
- Raffin et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268).
- Raffin (2020). *RL Baselines3 Zoo*: iperparametri di PPO per `LunarLander-v3`.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*.
- Ng, Harada & Russell (1999). *Policy invariance under reward transformations*.
- Bryson & Ho (1975). *Applied Optimal Control*.
- Rogers (1972). *Apollo Experience Report: Lunar Module Landing Gear Subsystem*. NASA TN D-6850.
