"""Dataset registry: where each dataset comes from and under which terms.

Each entry lists its archives (URL + expected SHA-256 once recorded), what to
extract, the licence, and what the dataset is used for here. Only datasets
that this project actually evaluates on are registered; the wider catalogue
(including datasets that need registration or could not be reached) is in
docs/DATASET_CATALOG.md.

Checksums are recorded in data/manifests/<id>.json the first time an archive
is downloaded ("trust on first use") and verified on every later download.
Licence fields marked ``verified: False`` still need confirming against the
official terms before any use beyond local evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Archive:
    url: str
    filename: str
    #: "zip" | "tar" | "none" (keep the file as downloaded) | "remote-zip" (extract members via HTTP ranges)
    extract: str = "zip"
    #: For remote-zip: path prefixes of the members to extract (selected subsets only).
    members: tuple[str, ...] = ()
    #: Delete the archive after a verified extraction (saves disk; the checksum stays in the manifest).
    delete_after_extract: bool = True


@dataclass(frozen=True, slots=True)
class Licence:
    annotations: str
    images: str
    commercial_use: str
    verified: bool
    note: str = ""


@dataclass(frozen=True, slots=True)
class DatasetSource:
    id: str
    name: str
    version: str
    homepage: str
    paper: str
    tasks: tuple[str, ...]
    licence: Licence
    archives: tuple[Archive, ...]
    #: Why this copy: mirrors are used when the official host is unreachable from this environment.
    acquisition_note: str
    #: The adapter in app.datasets.adapters that reads it.
    adapter: str
    #: Approximate download size in bytes (to check storage before downloading).
    approx_bytes: int
    extra: dict[str, str] = field(default_factory=dict)


ULTRALYTICS_RELEASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0"

SOURCES: dict[str, DatasetSource] = {
    "coco_val2017": DatasetSource(
        id="coco_val2017",
        name="COCO 2017 validation (detection)",
        version="2017",
        homepage="https://cocodataset.org/",
        paper="T.-Y. Lin et al., Microsoft COCO: Common Objects in Context, ECCV 2014, arXiv:1405.0312",
        tasks=("detection", "instance-segmentation"),
        licence=Licence(
            annotations="CC BY 4.0 (COCO Consortium)",
            images="Flickr terms of use; the COCO Consortium does not own the image copyrights",
            commercial_use="annotations yes (attribution); images depend on each Flickr licence",
            verified=True,
        ),
        archives=(Archive(f"{ULTRALYTICS_RELEASE}/coco2017val.zip", "coco2017val.zip"),),
        acquisition_note=(
            "images.cocodataset.org is not reachable from the build environment; this is the Ultralytics "
            "GitHub-release repackaging of the official val2017 images and instances_val2017.json. "
            "Integrity is checked against the official counts (5000 images, 36781 instances, 80 categories)."
        ),
        adapter="coco",
        approx_bytes=818_322_941,
    ),
    "visdrone_det_val": DatasetSource(
        id="visdrone_det_val",
        name="VisDrone2019-DET validation",
        version="2019",
        homepage="https://github.com/VisDrone/VisDrone-Dataset",
        paper="P. Zhu et al., Detection and Tracking Meet Drones Challenge, IEEE TPAMI 2021, arXiv:2001.06303",
        tasks=("detection", "small-objects"),
        licence=Licence(
            annotations="VisDrone terms (academic research)",
            images="VisDrone terms (academic research)",
            commercial_use="no (research use) - to be confirmed against the official terms",
            verified=False,
        ),
        archives=(Archive(f"{ULTRALYTICS_RELEASE}/VisDrone2019-DET-val.zip", "VisDrone2019-DET-val.zip"),),
        acquisition_note="Official downloads are on Google Drive/Baidu (unreachable here); Ultralytics GitHub-release copy of the original files.",
        adapter="visdrone",
        approx_bytes=81_638_851,
    ),
    "lvis_v1_labels": DatasetSource(
        id="lvis_v1_labels",
        name="LVIS v1 annotations (YOLO-format repackaging) on COCO 2017 images",
        version="v1",
        homepage="https://www.lvisdataset.org/",
        paper="A. Gupta, P. Dollar, R. Girshick, LVIS: A Dataset for Large Vocabulary Instance Segmentation, CVPR 2019, arXiv:1908.03195",
        tasks=("vocabulary-analysis",),
        licence=Licence(
            annotations="CC BY 4.0",
            images="COCO images (Flickr terms)",
            commercial_use="annotations yes (attribution)",
            verified=False,
        ),
        archives=(
            Archive(f"{ULTRALYTICS_RELEASE}/lvis-labels-segments.zip", "lvis-labels-segments.zip"),
            Archive(
                "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/lvis.yaml",
                "lvis.yaml",
                extract="none",
                delete_after_extract=False,
            ),
        ),
        acquisition_note=(
            "dl.fbaipublicfiles.com (official LVIS json) is not reachable here; the Ultralytics repackaging "
            "keeps the LVIS category order, names come from its lvis.yaml. Used only to count which annotated "
            "objects fall outside the detectors' vocabulary; images are the COCO ones."
        ),
        adapter="yolo",
        approx_bytes=520_852_930,
    ),
    "openimages_v5_val_boxes": DatasetSource(
        id="openimages_v5_val_boxes",
        name="Open Images V5 validation box annotations",
        version="v5 (boxes) / v7 (class names)",
        homepage="https://storage.googleapis.com/openimages/web/index.html",
        paper="A. Kuznetsova et al., The Open Images Dataset V4, IJCV 2020, arXiv:1811.00982",
        tasks=("vocabulary-analysis",),
        licence=Licence(
            annotations="CC BY 4.0 (Google LLC)",
            images="listed as CC BY 2.0",
            commercial_use="annotations yes (attribution)",
            verified=True,
        ),
        archives=(
            Archive(
                "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
                "validation-annotations-bbox.csv",
                extract="none",
                delete_after_extract=False,
            ),
            Archive(
                "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
                "oidv7-class-descriptions-boxable.csv",
                extract="none",
                delete_after_extract=False,
            ),
            Archive(
                "https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json",
                "bbox_labels_600_hierarchy.json",
                extract="none",
                delete_after_extract=False,
            ),
        ),
        acquisition_note="Annotations only (no images): used to measure the vocabulary gap of COCO-80 detectors.",
        adapter="openimages",
        approx_bytes=25_117_112,
    ),
    "kitti_tracking": DatasetSource(
        id="kitti_tracking",
        name="KITTI object tracking (training split, selected sequences)",
        version="2012",
        homepage="https://www.cvlibs.net/datasets/kitti/eval_tracking.php",
        paper="A. Geiger, P. Lenz, R. Urtasun, Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite, CVPR 2012",
        tasks=("tracking",),
        licence=Licence(
            annotations="CC BY-NC-SA 3.0",
            images="CC BY-NC-SA 3.0",
            commercial_use="no (non-commercial)",
            verified=True,
        ),
        archives=(
            Archive("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip", "data_tracking_label_2.zip"),
            Archive(
                "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip",
                "data_tracking_image_2.zip",
                extract="remote-zip",
                # Selected training sequences (see docs/EVALUATION_PROTOCOL.md for the choice).
                members=(
                    "training/image_02/0000/",
                    "training/image_02/0002/",
                    "training/image_02/0003/",
                    "training/image_02/0012/",
                    "training/image_02/0016/",
                ),
            ),
        ),
        acquisition_note=(
            "MOTChallenge is not reachable here. KITTI's 15.8 GB image archive is read with HTTP range "
            "requests so that only the selected training sequences are downloaded."
        ),
        adapter="kitti_tracking",
        approx_bytes=600_000_000,
    ),
    "visa": DatasetSource(
        id="visa",
        name="VisA (Visual Anomaly) dataset",
        version="20220922",
        homepage="https://github.com/amazon-science/spot-diff",
        paper="Y. Zou et al., SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation, ECCV 2022, arXiv:2207.14315",
        tasks=("anomaly-detection", "anomaly-segmentation"),
        licence=Licence(
            annotations="CC BY 4.0",
            images="CC BY 4.0",
            commercial_use="yes (attribution)",
            verified=True,
            note="LICENSE-DATASET shipped inside the official archive is CC BY 4.0.",
        ),
        archives=(
            Archive(
                "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
                "VisA_20220922.tar",
                extract="tar",
            ),
            Archive(
                "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv",
                "1cls.csv",
                extract="none",
                delete_after_extract=False,
            ),
        ),
        acquisition_note="Official S3 archive; the official one-class split (1cls.csv) defines train (normal only) and test.",
        adapter="visa",
        approx_bytes=1_929_840_640,
    ),
}


def get(dataset_id: str) -> DatasetSource:
    try:
        return SOURCES[dataset_id]
    except KeyError:
        raise KeyError(f"unknown dataset '{dataset_id}' (known: {', '.join(sorted(SOURCES))})") from None
