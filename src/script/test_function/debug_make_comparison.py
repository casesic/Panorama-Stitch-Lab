from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FRAME_INDICES = [513, 514, 515, 516, 517, 518]

INPUT_SINGLE_TEMPLATE = Path(
    r"data_process/Output/20260311_riding/vis/single_erp/vis_dual_{idx:05d}.jpg",
)
INPUT_SHARED_TEMPLATE = Path(
    r"data_process/Output/20260311_riding/vis/dual_erp_0/vis_dual_{idx:05d}.jpg",
)
INPUT_SHARED_CONF_TEMPLATE = Path(
    r"data_process/Output/20260311_riding/vis/dual_erp_0.57/vis_dual_{idx:05d}.jpg",
)
OUTPUT_DIR = Path(r"data_process/Output/20260311_riding/vis/comparison")

CAM2_CROP = (1781, 457, 2224, 815)
CAM4_CROP = (702, 1961, 1121, 2304)

COLUMN_GAP = 8
STACK_GAP = 4
CANVAS_BG = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
TITLE_TOP = 16
TITLE_BOTTOM = 12
FOOTER_TOP = 12
FOOTER_BOTTOM = 16
JPEG_QUALITY = 95

COLUMN_TITLES = [
    "单独shape",
    "共享shape",
    "共享shape+置信度限制",
]

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable font, preferring common Chinese system fonts."""
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def read_image(path: Path) -> Image.Image:
    """Read one image as RGB."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def crop_image(image: Image.Image, crop_box: tuple[int, int, int, int]) -> Image.Image:
    """Crop one image with a fixed crop box."""
    return image.crop(crop_box)


def build_column(image: Image.Image) -> Image.Image:
    """Build one column by stacking the cam2 and cam4 crops."""
    cam2_crop = crop_image(image, CAM2_CROP)
    cam4_crop = crop_image(image, CAM4_CROP)

    width = max(cam2_crop.width, cam4_crop.width)
    height = cam2_crop.height + STACK_GAP + cam4_crop.height
    canvas = Image.new("RGB", (width, height), CANVAS_BG)
    canvas.paste(cam2_crop, ((width - cam2_crop.width) // 2, 0))
    canvas.paste(cam4_crop, ((width - cam4_crop.width) // 2, cam2_crop.height + STACK_GAP))
    return canvas


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """Measure one text block."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def build_comparison_image(frame_idx: int) -> Image.Image:
    """Build the three-column comparison canvas for one frame."""
    single_image = read_image(Path(str(INPUT_SINGLE_TEMPLATE).format(idx=frame_idx)))
    shared_image = read_image(Path(str(INPUT_SHARED_TEMPLATE).format(idx=frame_idx)))
    shared_conf_image = read_image(Path(str(INPUT_SHARED_CONF_TEMPLATE).format(idx=frame_idx)))

    columns = [
        build_column(single_image),
        build_column(shared_image),
        build_column(shared_conf_image),
    ]
    column_widths = [column.width for column in columns]
    column_height = columns[0].height

    title_font = load_font(size=34)
    footer_font = load_font(size=30)
    probe = Image.new("RGB", (16, 16), CANVAS_BG)
    probe_draw = ImageDraw.Draw(probe)

    title_height = max(measure_text(probe_draw, title, title_font)[1] for title in COLUMN_TITLES)
    footer_text = f"frame {frame_idx}"
    footer_height = measure_text(probe_draw, footer_text, footer_font)[1]

    canvas_width = sum(column_widths) + COLUMN_GAP * (len(columns) - 1)
    canvas_height = (
        TITLE_TOP
        + title_height
        + TITLE_BOTTOM
        + column_height
        + FOOTER_TOP
        + footer_height
        + FOOTER_BOTTOM
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), CANVAS_BG)
    draw = ImageDraw.Draw(canvas)

    x = 0
    content_top = TITLE_TOP + title_height + TITLE_BOTTOM
    for title, column in zip(COLUMN_TITLES, columns):
        title_width, _ = measure_text(draw, title, title_font)
        title_x = x + (column.width - title_width) // 2
        draw.text((title_x, TITLE_TOP), title, fill=TEXT_COLOR, font=title_font)
        canvas.paste(column, (x, content_top))
        x += column.width + COLUMN_GAP

    footer_width, _ = measure_text(draw, footer_text, footer_font)
    footer_x = (canvas.width - footer_width) // 2
    footer_y = content_top + column_height + FOOTER_TOP
    draw.text((footer_x, footer_y), footer_text, fill=TEXT_COLOR, font=footer_font)
    return canvas


def main() -> None:
    """Generate comparison images for the configured frame indices."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for frame_idx in FRAME_INDICES:
        comparison = build_comparison_image(frame_idx)
        output_path = OUTPUT_DIR / f"comparison_{frame_idx:05d}.jpg"
        comparison.save(output_path, quality=JPEG_QUALITY)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
