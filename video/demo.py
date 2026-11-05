"""Phase 1 live demo: webcam -> detect -> track -> overlay, single process.

This is the "something runnable at the end of every phase" deliverable
(plan.md). It is intentionally single-process and synchronous — the full
multiprocess topology (SHM ring, event bus, separate CUDA streams) is the end
state, but a tight capture->detect->track->render loop is what proves the video
path works and lets you watch the frame budget on the HUD.

Run:
    uv run --extra video python -m video.demo                 # webcam 0
    uv run --extra video python -m video.demo --source clip.mp4
    uv run --extra video python -m video.demo --model yolo11s.pt --imgsz 640
    uv run --extra video python -m video.demo --headless --max-frames 100

Press 'q' in the window to quit.
"""

from __future__ import annotations

import argparse
import time

from core.events import VisionEvent
from core.metrics import timed
from sinks.overlay import FpsMeter, draw_hud, draw_tracks
from video.capture import VideoCapture
from video.detect import Detector
from video.track import IouTracker


def process_frame(detector: Detector, tracker: IouTracker, frame, ts_mono_ns: int,
                  seq: int) -> tuple[VisionEvent, dict[str, float]]:
    """Run one frame through detect -> track and return the VisionEvent plus
    per-stage millisecond timings. Pure logic, no I/O — this is what the tests
    exercise headless."""
    stage_ms: dict[str, float] = {}

    t0 = time.perf_counter()
    with timed("detect"):
        dets = detector.detect(frame)
    stage_ms["detect"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    with timed("track"):
        tracks = tracker.update(dets, ts_mono_ns)
    stage_ms["track"] = (time.perf_counter() - t0) * 1e3

    return VisionEvent(ts_mono_ns=ts_mono_ns, seq=seq, tracks=tracks), stage_ms


def run(args: argparse.Namespace) -> None:
    import cv2

    source: int | str = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    detector = Detector(model=args.model, conf=args.conf, imgsz=args.imgsz,
                        device=args.device)
    tracker = IouTracker(iou_thresh=args.iou, max_age=args.max_age,
                         new_track_conf=args.new_track_conf)
    fps_meter = FpsMeter()

    writer = None
    print(f"opening source {source!r} with model {args.model} on {args.device} ...")
    with VideoCapture(source) as cap:
        w, h = cap.frame_size
        print(f"stream {w}x{h}. press 'q' to quit.")
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, 30.0, (w, h))

        n = 0
        for cf in cap.frames():
            event, stage_ms = process_frame(detector, tracker, cf.frame,
                                             cf.ts_mono_ns, cf.seq)
            fps = fps_meter.tick(time.perf_counter())

            frame = draw_tracks(cf.frame, event.tracks)
            frame = draw_hud(frame, fps=fps, n_tracks=len(event.tracks),
                             stage_ms=stage_ms, seq=cf.seq)

            if writer is not None:
                writer.write(frame)
            if not args.headless:
                cv2.imshow("rt-perception — Phase 1", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            n += 1
            if args.max_frames and n >= args.max_frames:
                break
            if args.headless and n % 30 == 0:
                total = sum(stage_ms.values())
                print(f"  frame {n:5d}  fps {fps:5.1f}  objects {len(event.tracks):2d}  "
                      f"detect {stage_ms['detect']:5.1f}ms  track {stage_ms['track']:4.1f}ms  "
                      f"total {total:5.1f}ms")

    if writer is not None:
        writer.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"done. {n} frames.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase 1 webcam detection demo")
    ap.add_argument("--source", default="0", help="camera index, file path, or RTSP URL")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO11 weights (n/s/m/l/x) or path")
    ap.add_argument("--conf", type=float, default=0.35, help="detection confidence floor")
    ap.add_argument("--imgsz", type=int, default=640, help="inference resolution")
    ap.add_argument("--device", default="cpu", help="cpu | cuda:0 ...")
    ap.add_argument("--iou", type=float, default=0.3, help="tracker association IoU")
    ap.add_argument("--max-age", type=int, default=30, help="frames a track coasts unmatched")
    ap.add_argument("--new-track-conf", type=float, default=0.5, help="conf to spawn a track")
    ap.add_argument("--save", default=None, help="write annotated video to this path")
    ap.add_argument("--headless", action="store_true", help="no window (prints stats)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=forever)")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
