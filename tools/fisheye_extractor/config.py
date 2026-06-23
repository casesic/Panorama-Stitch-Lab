from __future__ import annotations

DS_PARAMS = {
    "xi": 0.02541467,
    "alpha": 0.58707823,
    "fx": 819.659819122500,
    "fy": 819.389361330000,
    "cx": 1455.78851018250,
    "cy": 1435.04276513250,
}

DS_PARAMS_CAM2 = {
    "xi": -0.12464126,
    "alpha": 0.56540615,
    "fx": 681.49808088,
    "fy": 682.11759351,
    "cx": 1445.73197064,
    "cy": 1444.25576211,
}

DS_PARAMS_CAM4 = {
    "xi": -0.15331803,
    "alpha": 0.55745717,
    "fx": 656.78496421,
    "fy": 655.70192322,
    "cx": 1440.93901552,
    "cy": 1443.33089179,
}

VIEWS = [
    {"yaw": 70, "pitch": 0, "name": "right"},
    {"yaw": -70, "pitch": 0, "name": "left"},
    {"yaw": 70, "pitch": -25, "name": "lower_right"},
    {"yaw": -70, "pitch": -25, "name": "lower_left"},
]

FOV_DEG = 50
OUT_SIZE = (640, 640)
FISHEYE_SIZE = (2880, 2880)
THETA_THRESHOLD = 60.0
MIN_VISIBLE_KPS = 14
BBOX_COMPLETENESS_THRESHOLD = 0.7
MIN_BBOX_AREA_RATIO = 0.01
DEDUP_INTERVAL_SEC = 2.0
SAMPLE_FPS = 2.0
DEVICE = "cuda"
