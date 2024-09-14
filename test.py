import mss

with mss.mss() as sct:
    # Obtenir la liste des moniteurs
    moniteurs = sct.monitors
    for i, monitor in enumerate(moniteurs[1:], start=1):
        print(f"Moniteur {i}: {monitor}")

    # Capture du deuxième moniteur
    moniteur_2 = moniteurs[2]  # Par exemple le deuxième moniteur
    screenshot = sct.grab(moniteur_2)

    # Enregistrer la capture
    mss.tools.to_png(screenshot.rgb, screenshot.size, output="capture_moniteur_2_mss.png")
