import imageio.v2 as imageio
import numpy as np
import os
from glob import glob

def pngs_to_mp4(
    frames_dir,
    out_mp4,
    fps=5,
    max_width=1920,   # Full HD target
):
    """
    Convert existing PNG frames to MP4:
      - Downscales large images
      - Forces even width & height (libx264 requirement)
    """

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    pngs = glob(os.path.join(frames_dir, "*.png"))
    if not pngs:
        raise RuntimeError(f"No PNGs found in {frames_dir}")

    # Sort numerically: 0000.png, 0001.png, ...
    pngs = sorted(pngs, key=lambda x: int(os.path.basename(x).split(".")[0]))

    with imageio.get_writer(
        out_mp4,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=10,
    ) as writer:

        for p in pngs:
            frame = imageio.imread(p)

            h, w = frame.shape[:2]

            # -------------------------------
            # Downscale if needed
            # -------------------------------
            if w > max_width:
                scale = max_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)

                # simple & fast subsampling (good enough for video)
                frame = frame[
                    np.linspace(0, h - 1, new_h).astype(int)[:, None],
                    np.linspace(0, w - 1, new_w).astype(int)
                ]

            # -------------------------------
            # Enforce EVEN dimensions
            # -------------------------------
            h, w = frame.shape[:2]
            frame = frame[: h - h % 2, : w - w % 2]

            writer.append_data(frame)
            print(f"✔ {os.path.basename(p)} -> {frame.shape[1]}x{frame.shape[0]}")

    print(f"\n✅ MP4 written to: {out_mp4}")


if __name__ == "__main__":
    FRAMES_DIR = (
        "/lus/flare/projects/Prob_AI/kanadsen/myrepos/"
        "dgn4cfd_tum/examples/ARO/outputs_pressure_norm/"
        "outputs_train/frames_all"
    )

    OUT_MP4 = (
        "/lus/flare/projects/Prob_AI/kanadsen/myrepos/"
        "dgn4cfd_tum/examples/ARO/outputs_pressure_norm/"
        "outputs_train/train_all_8k.mp4"
    )

    pngs_to_mp4(
        FRAMES_DIR,
        OUT_MP4,
        fps=5,
        max_width=7680,   # safe & clean
    )
