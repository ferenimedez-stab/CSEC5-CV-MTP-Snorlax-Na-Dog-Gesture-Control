# Gesture Control — Vision-Based Autonomous 4WD Robot (SnorlaxTheDog)

Computer Vision group project. The **Midterm** stage covers **SEE → RECOGNIZE**: a YOLO11n detector
recognizes 5 hand gestures from the laptop webcam.

| Gesture | HaGRID class | Robot command (Final) |
|---|---|---|
| Fist | `fist` | STOP |
| Thumbs Up | `like` | FORWARD |
| Open Palm | `palm` | BACKWARD |
| Rock Sign | `rock` | LEFT |
| Call Sign | `call` | RIGHT |

Pipeline: **Webcam → Preprocessing (CLAHE when dark, letterbox 640, normalize) → YOLO11n (ONNX) → Detection → Visual output**

## Layout
```
src/config.py                 classes, commands, thresholds
src/data/capture_webcam.py    capture self-collected images (unseen test set)
src/training/train_colab.ipynb  HaGRID (Kaggle) download, split, training, evaluation, ONNX export (Colab)
src/vision/preprocess.py      preprocessing
src/vision/detector.py        ONNX YOLO detector (onnxruntime)
src/vision/demo_app.py        demo app: live webcam, image mode, pipeline view
src/vision/webcam_detect.py   minimal live webcam viewer
src/vision/predict_images.py  run on a folder of images
dataset/  models/  results/midterm/  docs/
```

## Workflow
1. **Laptop setup**
   ```
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
   PyTorch is not installed locally (Windows Application Control blocks its DLLs on this laptop),
   so the model runs through ONNX Runtime.
2. **Capture webcam images** (each member, different rooms and lighting):
   `.venv\Scripts\python src/data/capture_webcam.py --person <name> --place <tag>`
   Zip `dataset/webcam` as `webcam.zip` and upload it to Google Drive `MyDrive/SnorlaxTheDog_Gestures/`.
3. **Train on Colab**: open `src/training/train_colab.ipynb`, choose a T4 GPU runtime, and run all cells.
4. **Bring results back**: download `MyDrive/SnorlaxTheDog_Gestures/midterm_results.zip`, put `best.onnx`
   in `models/`, and extract `figures/` and `tables/` into `results/midterm/`.
5. **Demo app** (live webcam + instructor images + pipeline view): `.venv\Scripts\python src/vision/demo_app.py`
   Keys: `p` pipeline view, `o` open image(s), `f` open folder, `c` camera, `←/→` previous/next image, `s` screenshot,
   `m` mirror, `l` low-light, `q` quit. Use `--images <file-or-folder>` to start in image mode.
6. **Batch predictions** (annotated copies + CSV): `.venv\Scripts\python src/vision/predict_images.py --source <folder>`

## Dataset
See [dataset/README.md](dataset/README.md). HaGRID Sample 30k 384p from Kaggle (CC BY-SA 4.0), plus optional self-collected webcam images.
