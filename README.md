# Tirol Map 3D 🏔️

## Features

- **Resolution:** Unterstützt 5m und 50cm Geländemodelle (M28).
- **Skalierung der Z-Achse:** Ändere (`Z_SCALE`) für eine verstärkte Relief-Darstellung.
- **Export:** In STL oder OBJ.
- **Koordinaten:** EPSG:31254 Koordinatensystem

## Voraussetzungen

- Python version >=3.10
- Python PIP packages:
  - requests
  - numpy
  - numpy-stl
  - rasterio

## Nutzung

1. Suche die gewünschten EPSG:31254 Koordinaten (z.B. auf [epsg.io](https://epsg.io/map)) und  trage sie in die `BBOX` Variable ein.
2. Höhenüberhöhung: `Z_SCALE` (1.0 = Originalgetreu).
3. `main.py` ausführen und das gewünschte Format wählen.

## Rechtlicher Hinweis

Dieses Tool nutzt Daten des Landes Tirol. Es gilt die Nutzung gemäß den ["Richtlinie über Standardentgelte und Standardbedingungen für die Weiterverwendung von Dokumenten des Landes Tirol"](https://www.tirol.gv.at/fileadmin/buergerservice/e-government/opendata/bilder/Dateien/TIWG/Standardbedingungen_Dez2017.pdf).

## Lizenz

MIT License - Copyright (c) 2026 Erik Tóth
