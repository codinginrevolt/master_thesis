#!/bin/bash

VALIDATE=false

for arg in "$@"
do
  case $arg in
    --validate)
      VALIDATE=true
      shift
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis || { echo "Failed to activate conda environment"; exit 1; }

echo "Generating EOS samples:"
python gpr/generate.py || { echo "Failed to execute generate samples"; exit 1; }

echo "Converting to MR/Tidal curves"
TOV/TOV || { echo "Failed to convert to MR/Tidal curves"; exit 1; }

if ["$VALIDATE" = true]; then
    echo "Validating samples:"
    python gpr/validate_samples.py || { echo "Failed to Validate samples"; exit 1; }

echo "Done and dusted"

