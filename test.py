from tqdm import tqdm
import time
import itertools

# Création d'une animation personnalisée pour la barre de chargement
def spinner_animation():
    spinners = itertools.cycle(['-', '\\', '|', '/'])
    while True:
        yield next(spinners)

# Fonction pour afficher une barre de chargement personnalisée
def custom_progress_bar(total, desc="Chargement", unit="unités", ncols=100):
    spinner = spinner_animation()
    
    with tqdm(total=total, desc=desc, unit=unit, ncols=ncols, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
        for i in range(total):
            # Animation du spinner
            pbar.set_postfix(spin=next(spinner))
            time.sleep(0.1)  # Simule une opération qui prend du temps
            pbar.update(1)

# Utilisation de la barre de chargement personnalisée
custom_progress_bar(100, desc="Téléchargement en cours", unit="unités")