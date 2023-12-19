import argparse
import os
import shutil
import sys

import fiftyone as fo
import numpy as np
import supervision as sv
from fiftyone import ViewField as F
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

IMG_DIR = "/dev/shm/pool"
INLCUDE_CLASSES = ["person", "car"]
EXPORT_DIR = "/dev/shm/bdd100k"
MODEL = "yolov8l.pt"

shutil.rmtree(EXPORT_DIR, ignore_errors=True)

if "ipykernel_launcher" in sys.argv[0]:
    SAMPLE_SIZE = 500
    THRESHOLD = 0.0
    NAME = "testrun"
    EPOCHS = 1

else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--name", type=str, default="testrun")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    SAMPLE_SIZE = args.sample_size
    THRESHOLD = args.threshold
    NAME = args.name
    EPOCHS = args.epochs


ds = fo.load_dataset("bdd-train")
Image.open(ds.first().filepath)

model = YOLO(MODEL)  # load an official model

view = ds.match(F("timeofday.label") == "daytime").take(SAMPLE_SIZE, seed=51)
view.distinct("timeofday.label")

samples = []
for sample in view.iter_samples(progress=True):
    imw = sample.metadata.width
    imh = sample.metadata.height
    results = model(sample.filepath)
    # raise
    id2label = results[0].names
    detections = sv.Detections.from_ultralytics(results[0])
    # Export from detections as bounding box data
    bounding_boxes = detections.xyxy

    fo_detections = []
    for (x1, y1, x2, y2), confidence, class_id in zip(
        detections.xyxy, detections.confidence, detections.class_id
    ):
        if id2label[class_id] not in INLCUDE_CLASSES:
            continue

        w = x2 - x1
        h = y2 - y1

        x1n = x1 / imw
        y1n = y1 / imh
        wn = w / imw
        hn = h / imh

        fo_detections.append(
            fo.Detection(
                label=id2label[class_id],
                bounding_box=[x1n, y1n, wn, hn],
                confidence=confidence,
            )
        )
    new_sample = fo.Sample(
        filepath=sample.filepath, detections=fo.Detections(detections=fo_detections)
    )

    samples.append(new_sample)

new_ds = fo.Dataset()
_ = new_ds.add_samples(samples)
new_ds.compute_metadata()

for sample in new_ds.iter_samples(
    progress=True,
):
    imw = sample.metadata.width
    imh = sample.metadata.height

    image = Image.open(sample.filepath)
    arr = np.array(image)

    valid_detections = []
    for det in sample.detections.detections:
        if det.confidence < THRESHOLD:
            x1n, y1n, wn, hn = det.bounding_box
            x1 = int(x1n * imw)
            y1 = int(y1n * imh)
            w = int(wn * imw)
            h = int(hn * imh)
            arr[y1 : y1 + h, x1 : x1 + w] = 0
        else:
            valid_detections.append(det)

    sample["valid_detections"] = fo.Detections(detections=valid_detections)

    from pathlib import Path

    dest = Path(IMG_DIR) / Path(sample.filepath).name
    dest.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray(arr).save(dest)

    sample.filepath = str(dest.absolute())
    sample.save()


train_ids, val_ids = train_test_split(
    new_ds.values("id"), test_size=0.2, random_state=51
)


# Export the splits
for split, ids in zip(["train", "val"], [train_ids, val_ids]):
    split_view = new_ds[ids]
    split_view.export(
        export_dir=EXPORT_DIR,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="valid_detections",
        split=split,
        classes=INLCUDE_CLASSES,
    )


model = YOLO(MODEL)  # load a pretrained model (recommended for training)
# Train the model
results = model.train(
    data=os.path.join(EXPORT_DIR, "dataset.yaml"),
    epochs=EPOCHS,
    imgsz=640,
    project="data/ultralytics",
    name=NAME,
    exist_ok=True,
)
