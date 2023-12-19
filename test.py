import os
import fiftyone as fo
from ultralytics import YOLO
from fiftyone import ViewField as F

IMG_DIR = "/dev/shm/pool"
INLCUDE_CLASSES = ["person", "car"]
EXPORT_DIR = "/dev/shm/bdd100k-test"
MODEL = "yolov8l.pt"

ds = fo.load_dataset("bdd-eval")
view = ds.match(F("timeofday.label") == "daytime")
view.distinct("timeofday.label")


view.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
    split="test",
    classes=INLCUDE_CLASSES,
)

tasks = [
    dict(name="default", weights=MODEL),
    dict(
        name="control",
        weights="data/ultralytics/control/weights/best.pt",
    ),
    dict(
        name="experiment",
        weights="data/ultralytics/experiment/weights/best.pt",
    ),
]
task2map = {}
for task in tasks:
    # raise
    model = YOLO(task["weights"])  # load a pretrained model (recommended for training)
    # Train the model
    val_results = model.val(
        data=os.path.join(EXPORT_DIR, "dataset.yaml")
        # , batch_size=1, imgsz=640
    )

    task2map[task["name"]] = val_results.results_dict["metrics/mAP50-95(B)"]


"""
{
    'default': 0.14961648065339903,
    'control': 0.3555776163686223,
    'experiment': 0.4284000874020748,
 }
"""
