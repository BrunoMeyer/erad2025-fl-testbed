#!/bin/bash

python3 plot.py --dir ERAD\ 2025\ IoT\ -\ LayerSize/ --fwidth 12 --fheight 12 --savefig --figftype png
python3 plot.py --dir ERAD\ 2025\ Docker\ -\ LayerSize/ --fwidth 12 --fheight 12 --savefig --figftype png
python3 plot.py --dir ERAD\ 2025\ -\ LayerSize/ --fwidth 12 --fheight 12 --savefig --figftype png

python3 plot.py --dir ERAD\ 2025\ IoT\ -\ NClients/ --fwidth 12 --fheight 12 --savefig --figftype png
python3 plot.py --dir ERAD\ 2025\ Docker\ -\ NClients/ --fwidth 12 --fheight 12 --savefig --figftype png
python3 plot.py --dir ERAD\ 2025\ -\ NClients/ --fwidth 12 --fheight 12 --savefig --figftype png

python3 plot.py --dir ERAD\ 2025\ IoT\ -\ NumSamples/ --fwidth 12 --fheight 12 --savefig --figftype png
python3 plot.py --dir ERAD\ 2025\ Docker\ -\ NumSamples/ --fwidth 12 --fheight 12 --savefig --figftype png
python3 plot.py --dir ERAD\ 2025\ -\ NumSamples/ --fwidth 12 --fheight 12 --savefig --figftype png