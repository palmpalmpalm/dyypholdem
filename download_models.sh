#!/bin/bash

# Define an array: "file_id destination_folder filename"
files=(
  "1aDIOsaDROQBaMtpXetThSmduGY46FwNT src/game/evaluation hand_ranks.pt"
  "1VixteYtYtdsorWc039Pyl7ZWn6Uq8lTN src/terminal_equity block_matrix.pt"
  "1VQnqGBDwY39oDdgJjsrAk0RuVNfShs6y src/nn/bucketing preflop_buckets.pt"
  "1oePwh3S27UM-URi8bZUqTp_lAYal4RZS src/terminal_equity preflop_equity.pt"
  "19VUnYVzRzHmicGA-P1tQkoNtGdNHvULA src/nn/bucketing ihr_pair_to_bucket.pkl"
  "1AjD1utFjn04v5IHWx1QUFXZWdwNxbgz7 src/nn/bucketing flop_dist_cats.pkl"
  "1gK82FqtSIghEPnkfvPmyoQxaE5O30rzZ src/nn/bucketing turn_dist_cats.pkl"
  "1X6PbbT2m7Dhr--IesIDy3kyPs0mDVuT- src/nn/bucketing river_ihr.pkl"
  "1i9UC1dAE1IKkcYGarYV2ZLg1J1Qdnj7A data/models/preflop-aux final_gpu.tar"
  "1UkcKLZp-AvBZXV0r5WgA6LuqzZI3CHzb data/models/preflop-aux final_cpu.tar"
  "1Q-l7dkld7GsJ35G-sJiZmSDDPrpGsWHg data/models/flop final_gpu.tar"
  "1GNFLsI1IzDGCWFCGSqawZi41ci0PIB66 data/models/flop final_cpu.tar"
  "1rTIyKudfL8OAayPv3CV8fb-0LuLwLaIj data/models/turn final_gpu.tar"
  "1pLzCHukEt6Q9TbtqlK4rq_5ma2lkqJOu data/models/turn final_cpu.tar"
  "1H1LC7Hnqro33m4AoMQL1VD7WCHnKfeDf data/models/river final_gpu.tar"
  "1HxzZUFcNvoThFq_WO_AeemOLlJ0c6C9m data/models/river final_cpu.tar"
)

for file in "${files[@]}"; do
  IFS=' ' read -r file_id dir filename <<< "$file"

  # Create directory if it doesn't exist
  mkdir -p "$dir"

  # Download the file with gdown
  echo "Downloading $filename into $dir..."
  gdown "$file_id" -O "$dir/$filename"
done
