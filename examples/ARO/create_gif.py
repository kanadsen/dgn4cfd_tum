import imageio
import os
from glob import glob

def make_video(frames_root, out_path, fps=5):
    """
    Convert all PNG frames in `frames_root` into an MP4 video.
    Frames are sorted numerically by filename.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Match all PNGs
    imgs = glob(os.path.join(frames_root, "*.png"))
    if len(imgs) == 0:
        raise ValueError(f"No PNGs found in {frames_root}")

    # Sort numerically: 0000.png, 0001.png, ..., 0010.png
    imgs = sorted(imgs, key=lambda x: int(os.path.basename(x).split(".")[0]))

    # Use ffmpeg to write MP4
    with imageio.get_writer(out_path, mode="I", fps=fps, codec="libx264", quality=8) as writer:
        for img in imgs:
            frame = imageio.imread(img)
            writer.append_data(frame)
            print(f"Appending frame: {img}")

    print(f"✅ Video written to: {out_path}")

if __name__ == "__main__":
    frames_root = "/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_pressure_norm/outputs_train/frames_all"
    out_path = "/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_pressure_norm/outputs_train/frames_all/train_pred.mp4"
    make_video(frames_root, out_path, fps=5)
