import time
import sys

def affichage_processing(texte, delai=0.5, repetitions=5):
    for _ in range(repetitions):
        for i in range(1, 1000001):  # Simulation d'un calcul
            pass  # Ici vous pouvez insérer votre propre code de calcul
            
        sys.stdout.write('\r' + texte + '   ')
        sys.stdout.flush()
        time.sleep(delai)
        sys.stdout.write('\r' + texte + '.  ')
        sys.stdout.flush()
        time.sleep(delai)
        sys.stdout.write('\r' + texte + '.. ')
        sys.stdout.flush()
        time.sleep(delai)
        sys.stdout.write('\r' + texte + '...')
        sys.stdout.flush()
        time.sleep(delai)
    sys.stdout.write('\r' + texte + '   \n')

# Utilisation de la fonction
affichage_processing("Processing", delai=0.5, repetitions=5)
