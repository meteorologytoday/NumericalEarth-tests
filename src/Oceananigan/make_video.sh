#!/bin/bash

#ffmpeg -i figures/frame-%05d.png -c:v libvpx-vp9 -pix_fmt yuv420p output.mp4



ffmpeg -framerate 12 \
  -pattern_type glob \
  -i 'figures/frame-?????.png' \
  -c:v libopenh264 \
  -vf "scale=3000:-1" \
  -b:v 12M \
  -pix_fmt yuv420p \
  output_openh264.mp4
