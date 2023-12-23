class MaClasse:
    def __init__(self, valeur):
        self.valeur = valeur
    
    def __str__(self):
        return f"MaClasse avec la valeur : {self.valeur}"

# Exemple d'utilisation
objet = MaClasse(42)
print(objet)  # Cela affichera : MaClasse avec la valeur : 42
del objet  # Cela supprimera l'objet de la mémoire
