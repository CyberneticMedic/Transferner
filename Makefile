# Pipeline order. Run on the training box with TRANSFERNER_DATA set.
PY := python -m transferner

data: resolve species tol gbif download dedup splits
# external needs a first splits pass (it picks species by n_train), then re-runs the tail
external: ; $(PY).data.gbif_media --external && $(PY).data.download --only-source gbif_external --procs 2 \
            && $(PY).data.dedup && $(PY).data.splits && $(PY).train.embed_images --splits val_ext train_ext
# boxes (OWLv2 pseudo-labels) runs separately once the detector work starts: make boxes
resolve:  ; $(PY).data.resolve_sources
species:  ; $(PY).data.gbif_species_list && $(PY).data.build_species_list
tol:      ; $(PY).data.tol_select
gbif:     ; $(PY).data.gbif_media
download: ; $(PY).data.download
dedup:    ; $(PY).data.dedup
splits:   ; $(PY).data.splits
boxes:    ; $(PY).data.pseudo_boxes

bioclip:  ; $(PY).train.embed_text && $(PY).train.embed_images && $(PY).train.linear_probe && $(PY).train.calibrate
detector: ; $(PY).train.train_detector && $(PY).quant.qat
quant:    ; $(PY).quant.ptq
export:   ; $(PY).export.litert && $(PY).export.executorch && $(PY).export.bundle
eval:     ; $(PY).eval.evaluate

.PHONY: data external resolve species tol gbif download dedup splits boxes bioclip detector quant export eval
