# Android app (Kotlin) — prototype 0.6

Native Kotlin, Android only (D-016), Jetpack Compose + CameraX + LiteRT 2.4 (`CompiledModel` API). No models
ship in the APK: on first launch the app downloads a versioned bundle from `scripts/serve_bundle.sh` on the
training box and checks every file against the bundle's SHA-256 manifest (D-034).

## What it does
- **Live scanner:** CameraX frames (640×480, RGBA) → upright → squash to the detector's size (448 from bundle
  0.5) → INT8 detector → whole-frame genus and family (time-based smoothing) plus class-agnostic plant boxes.
  The 3:4 frame is shown whole between the top bar and the sheet (0.6). Boxes after the first need a higher
  score, nested boxes are dropped unless the first box fills the frame, and a tracker keeps them steady. A
  plant-presence gate (top-5 genus mass, `tuning.json`) hides boxes and the toxic row on frames that look like
  no known plant. Tap to focus; the light button toggles the torch.
- **Walk mode (0.6, D-044):** screen on and dimmed, 4 frames a second. A forage score (genus probability ×
  share of its sightings that are cited edible species, over the user's forage list) has to stay up for about a
  second; then BioCLIP checks the frame and the phone buzzes twice if it agrees, showing the window where the
  genus scores highest. A long buzz means a genus holding a seriously toxic species stayed in view. The flag
  button marks a plant the app missed. Walks are recorded (JSON lines for every frame, a JPEG a second) for
  tuning and training, and uploaded with the log.
- **Capture or gallery photo:** shortest side to 224 (bicubic, antialiased) → centre crop → BioCLIP 2.5 image
  tower → regional head → calibrated distribution over 4,411 species (duplicated names summed).
- **Results:** a headline level: the species when its probability is at least 0.8, else "<Genus> sp." when the
  genus sum is, else no confident match; one level lower when the photo matches the species names poorly (text
  score), and "maybe not a plant I know, toxicity unknown" at the bottom. A caution banner when cited-toxic
  species and their relatives together reach 0.2% (0.6; stronger at 2%), or a conditional species is in the top
  5 from 2%. On an edible/toxic conflict, or when unsure, the app asks for a second photo of another part and
  combines the two. Each row has a trait chip ("No trait data" when no cited row exists; D-027).
- **Species page:** cited trait text and its source, the other candidates, credits, and a Poison Control
  button for species listed as poisonous if eaten.
- **Log:** a JPEG (two for combined photos), a CSV row and a JSON sidecar per saved capture (the sidecar also
  holds the live model's last frames before the capture). Uploads send the CSV first, then each capture's files.
- **Settings:** this phone's profile and plan, the detector runtime (Auto measures GPU against CPU), the species-model variant and CPU threads, timings, and re-downloading the bundle.

## Phones and fallback (D-035)
The app profiles the phone at first launch (SoC, RAM, whether LiteRT has a GPU) and picks a plan
(`core/DevicePlan.kt`):

| Phone | Species model | Live detector |
|---|---|---|
| Pixel 9 Pro (Tensor G4, 16 GB) and other flagships (Snapdragon 8, Dimensity 9xxx) with ≥ 10 GB | `gpu` on the GPU | GPU or CPU, whichever is faster when measured on the phone |
| Moto G Power 2025 (Dimensity 6300: 2× A76 + 6× A55, Mali-G57 MC2, 8 GB) and any phone with ≥ 6 GB | `cpu` on the CPU (XNNPACK, 4 threads, weight cache) | measured, probably CPU; live capped at 8 fps |
| Phones with < 6 GB (e.g. Moto G 2025, 4 GB) | `cpu-small` on the CPU | measured; 6 fps |

The chain steps down when something fails:
- **Load or run failure:** if loading or running a model throws, the app steps down `gpu` → `cpu` → `cpu-small` → 2 threads.
- **Process killed:** if a native out-of-memory kill takes the whole process down, a crash flag (written before every load and run) makes the next launch step down instead of crashing again.
- **Variant not on the phone:** if the next step needs a variant that isn't installed, the setup screen offers to download it. Shared files are kept.

Other fallbacks:
- **Camera:** if the camera can't run preview, analysis and capture together, the live analysis is dropped and capture still works. If there is no camera, identifying from the gallery still works.
- **Live detector:** it pauses during an identification, so the CPU goes to the species model.
- **Capture size:** stills are taken at about 3 MP rather than the full 50 MP.

## Model I/O contract (bundle written by `src/transferner/export/bundle.py`)

| File | Input | Output |
|---|---|---|
| `detector.tflite` + `detector.json` | int8 `[1,3,S,S]` NCHW (S = 320 to bundle 0.4, 448 from 0.5), normalised with ImageNet mean/std, then quantised with the sidecar's scale and zero point. The app uses a lookup table per channel. | `image_logits [1,G]`, `objectness [1,A]`, `boxes [1,A,4]`, and from 0.6 `region_logits [1,9,G]` (the whole-frame head on 9 overlapping windows; their geometry is the sidecar's `regions`). Output names, dtypes, quantisation, anchors and box coder are in the sidecar. |
| `species_<id>.tflite` + `species.json` | float32 `[1,3,224,224]` NCHW, CLIP mean/std | `[1,1024]` L2-normalised embedding. One file per variant: `gpu` W8A16 (610 MB), `cpu` dynamic INT8 (610 MB), `cpu-small` INT4 + dynamic INT8 (345 MB). Manifest group `species:<id>`, so a phone downloads only its own variant. |
| `head.bin` + `head.json` | embedding | `softmax((scale · e·Wᵀ + b) / T)` over 4,411 species. W is float16, row-major, little-endian. |
| `catalog.json` | — | Per species: names, genus, family, look-alike groups, cited trait rows, thumbnail and its credit |
| `thumbs/<class>.jpg` | — | One CC0 / CC BY 4.0 field photo per species (D-019) |
| `text_emb.bin` (0.6) | — | BioCLIP text embeddings of the species names, float16 `[4411,1024]`, named by `head.json` `text_file` |
| `forage.json` (0.6) | — | Walk-mode genera: detector index, default tier (alert / show / off), weight, label, note |
| `tuning.json` (0.6) | — | Live and capture thresholds; older bundles use the app's defaults |
| `manifest.json`, `../latest.json` | — | Version, then bytes and SHA-256 for every file |

The detector graph is `src/transferner/export/detector_app.py`:
- `image_logits` comes from the whole-photo branch.
- `objectness` = `sigmoid(logsumexp(genus logits) − background logit)`, which is 1 − P(background).
- Decoding and NMS run in the app (`core/Detections.kt`).

## Build
The toolchain lives in user space under `/models/ml/android` (JDK 17, Android SDK, Gradle). No sudo is
needed.
```bash
source /models/ml/android/env.sh
cd ~/Transferner/app
./gradlew assembleRelease          # app/build/outputs/apk/release/app-release.apk (debug-signed, sideload only)
./gradlew testDebugUnitTest        # Kotlin pre/post-processing vs the Python goldens ($TRANSFERNER_DATA/app_golden)
./gradlew testDebugUnitTest -Ptransferner.server=http://127.0.0.1:8765/   # + the bundle download, end to end
```
The goldens come from `export/bundle.py --goldens $TRANSFERNER_DATA/app_golden`. They are data, so they are
not in git. The tests skip when the goldens are absent.

## Install on the phone
1. Run `bash scripts/serve_bundle.sh`. It serves `$TRANSFERNER_DATA/bundle/` on port 8765.
2. On the phone, open `http://<server>:8765/` and install the APK listed there.
3. Open the app. It downloads the bundle from the address on its setup screen.

### Away from home: the tailnet (D-047)
The server listens on every interface, so with Tailscale on the phone the same page and files are at
`http://<machine>.<tailnet>.ts.net:8765/`. From 0.7.1 the app probes the saved address, the home-network default and
the tailnet addresses together and uses the first in that order that answers (Settings shows which). The tailnet
addresses are built in from `app/local.properties`, which is gitignored, so the tailnet name stays out of this
public repo:
```
transferner.tailnet=http://<machine>.<tailnet>.ts.net:8765/,http://<tailscale ip>:8765/
```
