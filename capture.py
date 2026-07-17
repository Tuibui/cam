#!/usr/bin/env python3
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import yaml

DEFAULTS = {
    "capture": {
        "interval": 10.0,
        "output": "dataset",
        "resolution": "1080",
        "quality": 95,
        "prefix": "img",
        "max_images": 0,
    },
    "camera": {
        "fps": 10,
        "focus": "auto",
        "exposure": "auto",
        "white_balance": "auto",
        "brightness": 0,
        "contrast": 0,
        "saturation": 0,
        "sharpness": 1,
        "luma_denoise": 1,
        "chroma_denoise": 1,
        "anti_banding": "50hz",
    },
}

RESOLUTIONS = {
    "1080": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
    "12mp": dai.ColorCameraProperties.SensorResolution.THE_12_MP,
}

ANTI_BANDING = {
    "off": dai.CameraControl.AntiBandingMode.OFF,
    "auto": dai.CameraControl.AntiBandingMode.AUTO,
    "50hz": dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
    "60hz": dai.CameraControl.AntiBandingMode.MAINS_60_HZ,
}


def parse_args():
    p = argparse.ArgumentParser(description="OAK-D-Lite interval capture for dataset collection")
    p.add_argument("-c", "--config", type=str, default="config.yaml")
    p.add_argument("-i", "--interval", type=float, default=None)
    p.add_argument("-o", "--output", type=str, default=None)
    p.add_argument("-r", "--resolution", type=str, default=None, choices=["1080", "4k", "12mp"])
    p.add_argument("-n", "--max-images", type=int, default=None)
    p.add_argument("-q", "--quality", type=int, default=None)
    p.add_argument("--prefix", type=str, default=None)
    p.add_argument("--preview", action="store_true")
    return p.parse_args()


def load_config(args) -> dict:
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}

    path = Path(args.config)
    if path.exists():
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        for section in ("capture", "camera"):
            for key, value in (user.get(section) or {}).items():
                if key in cfg[section]:
                    cfg[section][key] = value
                else:
                    print(f"[WARN] Unknown config key '{section}.{key}' ignored")
        print(f"[INFO] Loaded config  : {path}")
    else:
        print(f"[INFO] Config '{path}' not found, using defaults")

    for key in ("interval", "output", "resolution", "quality", "prefix", "max_images"):
        cli_val = getattr(args, key)
        if cli_val is not None:
            cfg["capture"][key] = cli_val

    cfg["capture"]["resolution"] = str(cfg["capture"]["resolution"])
    return cfg


def apply_camera_settings(cam: dai.node.ColorCamera, cam_cfg: dict):
    ctrl = cam.initialControl

    focus = cam_cfg["focus"]
    if focus == "auto":
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
    else:
        ctrl.setManualFocus(int(focus))

    exposure = cam_cfg["exposure"]
    if exposure == "auto":
        ctrl.setAutoExposureEnable()
    else:
        ctrl.setManualExposure(int(exposure["time_us"]), int(exposure["iso"]))

    wb = cam_cfg["white_balance"]
    if wb != "auto":
        ctrl.setManualWhiteBalance(int(wb))

    ctrl.setBrightness(int(cam_cfg["brightness"]))
    ctrl.setContrast(int(cam_cfg["contrast"]))
    ctrl.setSaturation(int(cam_cfg["saturation"]))
    ctrl.setSharpness(int(cam_cfg["sharpness"]))
    ctrl.setLumaDenoise(int(cam_cfg["luma_denoise"]))
    ctrl.setChromaDenoise(int(cam_cfg["chroma_denoise"]))

    anti = cam_cfg["anti_banding"]
    if anti is False:
        anti = "off"
    ctrl.setAntiBandingMode(ANTI_BANDING[str(anti).lower()])


def build_pipeline(cfg: dict) -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(RESOLUTIONS[cfg["capture"]["resolution"]])
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(int(cfg["camera"]["fps"]))

    apply_camera_settings(cam, cfg["camera"])

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.video.link(xout.input)

    return pipeline


def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    numbers = []
    for d in base.iterdir():
        parts = d.name.split("_")
        if d.is_dir() and len(parts) >= 2 and parts[0] == "run" and parts[1].isdigit():
            numbers.append(int(parts[1]))
    run_no = max(numbers, default=0) + 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{run_no:03d}_{stamp}"
    run_dir.mkdir()
    return run_dir


def main():
    args = parse_args()
    cfg = load_config(args)
    cap = cfg["capture"]

    out_dir = next_run_dir(Path(cap["output"]))

    print(f"[INFO] Saving to      : {out_dir.resolve()}")
    print(f"[INFO] Interval       : {cap['interval']} s")
    print(f"[INFO] Resolution     : {cap['resolution']}")
    print(f"[INFO] Max images     : {cap['max_images'] if cap['max_images'] else 'unlimited'}")
    print(f"[INFO] Focus          : {cfg['camera']['focus']}")
    print(f"[INFO] Exposure       : {cfg['camera']['exposure']}")
    print(f"[INFO] White balance  : {cfg['camera']['white_balance']}")
    print("[INFO] Press Ctrl+C to stop.")

    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    pipeline = build_pipeline(cfg)
    interval = float(cap["interval"])
    count = 0

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        next_shot = time.monotonic()

        try:
            while True:
                frame_msg = q.tryGet()
                if frame_msg is None:
                    time.sleep(0.01)
                    continue

                frame = frame_msg.getCvFrame()

                if args.preview:
                    small = cv2.resize(frame, (640, 360))
                    cv2.imshow("OAK-D-Lite preview (q to quit)", small)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                now = time.monotonic()
                if now >= next_shot:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    fname = out_dir / f"{cap['prefix']}_{ts}.jpg"
                    cv2.imwrite(str(fname), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, int(cap["quality"])])
                    count += 1
                    print(f"[{count}] saved {fname.name}")

                    if cap["max_images"] and count >= int(cap["max_images"]):
                        print("[INFO] Reached max images, stopping.")
                        break

                    next_shot += interval
                    if next_shot < now:
                        next_shot = now + interval

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
        finally:
            if args.preview:
                cv2.destroyAllWindows()

    print(f"[DONE] {count} images saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
