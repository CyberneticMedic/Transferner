#!/usr/bin/env bash
# Background / true-negative imagery for the live-detector retrain (user approval 2026-11-24; D-044).
# Bulk files only; Open Images photos are fetched per image by transferner.data.negatives after selection.
#   COCO 2017 train+val images, instance + stuff annotations (stuff masks name tree/grass/bush/plant/flower/
#     leaves/moss, so plant-free frames can be selected exactly)        ~21 GB
#   Places365 val_large (36,500 scene photos, 100 per category, >= 512 px)  ~2.1 GB
#   Open Images V7 human-verified image labels + class names + boxable image list (ids, licence, author)  ~3.4 GB
# Resumable (curl -C -), 4 transfers at a time, then unpacked in place.
#   bash scripts/fetch_negatives.sh
set -u
: "${TRANSFERNER_DATA:?source ~/ml-env.sh first}"
D=$TRANSFERNER_DATA/raw/negatives
mkdir -p "$D"/{coco,places,openimages}
get() {  # url dest
  for i in 1 2 3 4 5; do
    curl -sSL --fail -C - -o "$2" "$1" && return 0
    echo "retry $i: $1" >&2; sleep 20
  done
  echo "FAILED $1" >&2; return 1
}
export -f get
cat > "$D/urls.tsv" <<EOF
http://images.cocodataset.org/zips/train2017.zip	$D/coco/train2017.zip
http://images.cocodataset.org/zips/val2017.zip	$D/coco/val2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip	$D/coco/annotations_trainval2017.zip
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip	$D/coco/stuff_annotations_trainval2017.zip
https://data.csail.mit.edu/places/places365/val_large.tar	$D/places/val_large.tar
https://data.csail.mit.edu/places/places365/filelist_places365-standard.tar	$D/places/filelist_places365-standard.tar
https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv	$D/openimages/oidv7-train-annotations-human-imagelabels.csv
https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv	$D/openimages/oidv7-class-descriptions.csv
https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv	$D/openimages/train-images-boxable-with-rotation.csv
EOF
awk -F'\t' '{print $1"\n"$2}' "$D/urls.tsv" | xargs -d '\n' -n 2 -P 4 bash -c 'get "$0" "$1" && echo "done $(basename "$1")"'
cd "$D/coco" && for z in *.zip; do unzip -qn "$z" && echo "unzipped $z"; done
cd "$D/coco" && [ -f annotations/stuff_train2017_pixelmaps.zip ] && unzip -qn annotations/stuff_train2017_pixelmaps.zip -d annotations/ 2>/dev/null; true
cd "$D/places" && tar -xf val_large.tar && tar -xf filelist_places365-standard.tar && echo "untarred places"
du -sh "$D"/* && echo "fetch_negatives: finished"
