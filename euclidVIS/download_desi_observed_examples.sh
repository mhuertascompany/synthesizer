#!/usr/bin/env bash
set -euo pipefail

# Modern macOS scp defaults to SFTP, which is disabled on Teide login nodes.
# Uppercase -O forces the legacy SCP protocol.
remote_host="${REMOTE_HOST:-mhuertas@10.5.22.101}"
remote_spectra_dir="${REMOTE_SPECTRA_DIR:-/home/mhuertas/iac18_aasensio_shared/euclid_dr1/spectra}"
destination="${1:-/Users/marchuertascompany/Documents/data/euclid_desi/spectra}"

mkdir -p "$destination"

targetids=(
  39633451346822358
  39633443742549540
  39627841595250624
  39633286372261925
  39633282664499785
  39627853687423916
  39633438604527060
  39627829511456404
  39633446351408671
)

for targetid in "${targetids[@]}"; do
  filename="TARGETID_${targetid}.fits"
  scp -O "${remote_host}:${remote_spectra_dir}/${filename}" "$destination/"
done

echo "Downloaded ${#targetids[@]} spectra to $destination"
