#!/bin/sh
# Download and unpack the stroke-rehab dataset provided by g.tec for the BR41N.IO hackathon.
# Needs bsdtar (ships with macOS; `libarchive-tools` on Debian/Ubuntu).
set -e
mkdir -p data/raw data/stroke-rehab
curl -L -o data/raw/stroke-rehab.rar https://www.gtec.at/downloads_QyTs23/Hackathon/stroke-rehab.rar
bsdtar -xf data/raw/stroke-rehab.rar -C data/stroke-rehab
