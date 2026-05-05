from pathlib import Path

import cv2
import numpy as np


IMAGE_PATH = Path(
    r"data_process/Output/20260311_riding/raft_stitch/frame_00513/"
    r"raft_cam4_to_cam2_unrotated_frame_00513.png"
)
MASK_PATH = Path(r"data_process/Output/20260311_riding/raft_stitch/unrotated_repair_mask.png")

WINDOW_NAME = "draw unrotated repair mask"
OVERLAY_ALPHA = 0.45
INITIAL_BRUSH_RADIUS = 12
MIN_BRUSH_RADIUS = 1
MAX_BRUSH_RADIUS = 200
MIN_VIEW_SCALE = 0.1
MAX_VIEW_SCALE = 8.0
DISPLAY_MAX_WIDTH = 1600
DISPLAY_MAX_HEIGHT = 900


image: np.ndarray | None = None
mask: np.ndarray | None = None
brush_radius = INITIAL_BRUSH_RADIUS
drawing = False
draw_value = 255
view_scale = 1.0
view_cx = 0.0
view_cy = 0.0
display_width = DISPLAY_MAX_WIDTH
display_height = DISPLAY_MAX_HEIGHT
panning = False
last_pan_x = 0
last_pan_y = 0


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def clamp_view_center() -> None:
    global view_cx, view_cy

    height, width = image.shape[:2]
    margin_x = width * 0.05
    margin_y = height * 0.05
    view_cx = clamp(view_cx, -margin_x, width - 1 + margin_x)
    view_cy = clamp(view_cy, -margin_y, height - 1 + margin_y)


def reset_view() -> None:
    global view_scale, view_cx, view_cy

    height, width = image.shape[:2]
    view_scale = 1.0
    view_cx = (width - 1) / 2.0
    view_cy = (height - 1) / 2.0


def init_display_size() -> None:
    global display_width, display_height

    height, width = image.shape[:2]
    fit_scale = min(DISPLAY_MAX_WIDTH / width, DISPLAY_MAX_HEIGHT / height, 1.0)
    display_width = max(1, int(round(width * fit_scale)))
    display_height = max(1, int(round(height * fit_scale)))


def get_crop_rect() -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    effective_scale = max(view_scale, 1.0)
    crop_w = min(width, max(1, int(round(width / effective_scale))))
    crop_h = min(height, max(1, int(round(height / effective_scale))))

    x0 = int(round(view_cx - crop_w / 2.0))
    y0 = int(round(view_cy - crop_h / 2.0))
    x0 = max(0, min(width - crop_w, x0))
    y0 = max(0, min(height - crop_h, y0))
    return x0, y0, x0 + crop_w, y0 + crop_h


def make_full_overlay() -> np.ndarray:
    display = image.copy()
    red = np.zeros_like(display)
    red[:, :, 2] = 255
    selected = mask > 0
    if np.any(selected):
        overlay = cv2.addWeighted(display, 1.0 - OVERLAY_ALPHA, red, OVERLAY_ALPHA, 0.0)
        display[selected] = overlay[selected]
    return display


def make_display() -> np.ndarray:
    full_overlay = make_full_overlay()
    x0, y0, x1, y1 = get_crop_rect()
    crop = full_overlay[y0:y1, x0:x1]
    display = cv2.resize(crop, (display_width, display_height), interpolation=cv2.INTER_AREA)

    lines = [
        f"zoom={view_scale:.2f}x  brush={brush_radius}px  center=({view_cx:.0f}, {view_cy:.0f})",
        "left draw | right erase | middle drag pan | wheel zoom",
        "s save | c clear | r reset view | +/- brush | q quit",
    ]
    for index, text in enumerate(lines):
        y = 28 + index * 26
        cv2.putText(display, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(display, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
    return display


def display_to_image_xy(x: int, y: int) -> tuple[int, int] | None:
    if x < 0 or y < 0 or x >= display_width or y >= display_height:
        return None

    height, width = image.shape[:2]
    x0, y0, x1, y1 = get_crop_rect()
    image_x = x0 + (x / max(1, display_width - 1)) * (x1 - x0 - 1)
    image_y = y0 + (y / max(1, display_height - 1)) * (y1 - y0 - 1)
    image_x = int(round(image_x))
    image_y = int(round(image_y))

    if image_x < 0 or image_y < 0 or image_x >= width or image_y >= height:
        return None
    return image_x, image_y


def draw_at(x: int, y: int) -> None:
    image_xy = display_to_image_xy(x, y)
    if image_xy is None:
        return
    cv2.circle(mask, image_xy, brush_radius, draw_value, thickness=-1, lineType=cv2.LINE_8)


def pan_view(dx: int, dy: int) -> None:
    global view_cx, view_cy

    x0, y0, x1, y1 = get_crop_rect()
    crop_w = x1 - x0
    crop_h = y1 - y0
    view_cx -= dx / max(1, display_width) * crop_w
    view_cy -= dy / max(1, display_height) * crop_h
    clamp_view_center()


def zoom_at(display_x: int, display_y: int, factor: float) -> None:
    global view_scale, view_cx, view_cy

    anchor = display_to_image_xy(display_x, display_y)
    old_scale = view_scale
    view_scale = clamp(view_scale * factor, MIN_VIEW_SCALE, MAX_VIEW_SCALE)
    if anchor is None or view_scale == old_scale:
        clamp_view_center()
        return

    height, width = image.shape[:2]
    effective_scale = max(view_scale, 1.0)
    crop_w = min(width, width / effective_scale)
    crop_h = min(height, height / effective_scale)
    fx = display_x / max(1, display_width - 1)
    fy = display_y / max(1, display_height - 1)
    view_cx = anchor[0] + (0.5 - fx) * crop_w
    view_cy = anchor[1] + (0.5 - fy) * crop_h
    clamp_view_center()


def mouse_wheel_delta(flags: int) -> int:
    if hasattr(cv2, "getMouseWheelDelta"):
        return cv2.getMouseWheelDelta(flags)
    return 1 if flags > 0 else -1


def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
    global drawing, draw_value, panning, last_pan_x, last_pan_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        draw_value = 255
        draw_at(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        draw_value = 0
        draw_at(x, y)
    elif event == cv2.EVENT_MBUTTONDOWN:
        panning = True
        last_pan_x = x
        last_pan_y = y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            draw_value = 255
            draw_at(x, y)
        elif flags & cv2.EVENT_FLAG_RBUTTON:
            draw_value = 0
            draw_at(x, y)
    elif event == cv2.EVENT_MOUSEMOVE and panning:
        pan_view(x - last_pan_x, y - last_pan_y)
        last_pan_x = x
        last_pan_y = y
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        drawing = False
    elif event == cv2.EVENT_MBUTTONUP:
        panning = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        if mouse_wheel_delta(flags) > 0:
            zoom_at(x, y, 1.25)
        else:
            zoom_at(x, y, 0.8)


def load_mask(mask_path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not mask_path.exists():
        return np.zeros(shape, dtype=np.uint8)

    loaded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if loaded is None:
        print(f"Warning: failed to read existing mask, starting empty: {mask_path}")
        return np.zeros(shape, dtype=np.uint8)
    if loaded.shape != shape:
        raise ValueError(f"Mask size {loaded.shape} does not match image size {shape}: {mask_path}")
    return np.where(loaded > 0, 255, 0).astype(np.uint8)


def main() -> None:
    global image, mask, brush_radius

    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {IMAGE_PATH}")

    mask = load_mask(MASK_PATH, image.shape[:2])
    init_display_size()
    reset_view()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, display_width, display_height)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print(f"Image: {IMAGE_PATH}")
    print(f"Mask:  {MASK_PATH}")
    print("Controls: left drag draw, right drag erase, middle drag pan, wheel zoom")
    print("          s save, c clear, r reset view, +/- brush, q quit")

    while True:
        cv2.imshow(WINDOW_NAME, make_display())
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
            binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            cv2.imwrite(str(MASK_PATH), binary_mask)
            print(f"Saved mask: {MASK_PATH}")
        elif key == ord("c"):
            mask.fill(0)
            print("Cleared mask.")
        elif key == ord("r"):
            reset_view()
            print("Reset view.")
        elif key in (ord("+"), ord("=")):
            brush_radius = min(MAX_BRUSH_RADIUS, brush_radius + 1)
            print(f"Brush radius: {brush_radius}")
        elif key == ord("-"):
            brush_radius = max(MIN_BRUSH_RADIUS, brush_radius - 1)
            print(f"Brush radius: {brush_radius}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
