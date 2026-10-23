"""Build Stanley site images from the logo and the banner.

Memes are photo types (caption, split, labels, comic, badge, film strip),
not the polka-dot poster set this page used before.
"""
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import cv2
import os

ROOT = r"E:\pack\solana\STANLEY"
FONT = os.path.join(ROOT, "assets", "LilitaOne-Regular.ttf")
SRC_LOGO = os.path.join(ROOT, "logo.png")
SRC_BANNER = os.path.join(ROOT, "banner.png")

INK = (27, 31, 75, 255)
YELLOW = (255, 210, 31, 255)
PINK = (255, 45, 149, 255)
PAPER = (251, 247, 239, 255)
WHITE = (255, 255, 255, 255)
ORANGE = (255, 186, 83, 255)
BLACK = (8, 8, 12, 255)

# Banner panels, inset so the orange-board lettering stays out of the photos.
BOX = {
    "look": (4, 4, 284, 250),
    "soap": (4, 260, 284, 496),
    "rear": (294, 4, 505, 250),
    "sit": (294, 260, 462, 496),
    "up": (972, 6, 1216, 250),
    "blep": (1042, 262, 1216, 494),
    "tap": (1228, 4, 1496, 250),
    "chin": (1228, 260, 1496, 496),
}


def font(size):
    return ImageFont.truetype(FONT, size)


def text_size(draw, text, fnt, sw=0):
    box = draw.textbbox((0, 0), text, font=fnt, stroke_width=sw)
    return box[2] - box[0], box[3] - box[1], box


def stamp(draw, xy, text, fnt, fill, stroke=BLACK, sw=8):
    draw.text(xy, text, font=fnt, fill=fill, stroke_width=sw, stroke_fill=stroke)


def stamp_center(draw, y, text, fnt, width, fill, stroke=BLACK, sw=8):
    tw, th, box = text_size(draw, text, fnt, sw)
    x = (width - tw) // 2 - box[0]
    stamp(draw, (x, y - box[1]), text, fnt, fill, stroke, sw)
    return th


def banner_word(base, xy, text, fnt, fill=YELLOW, sw=10, shift=(7, 7)):
    """Yellow word with the hard offset shadow used on the banner."""
    draw = ImageDraw.Draw(base)
    x, y = xy
    draw.text((x + shift[0], y + shift[1]), text, font=fnt, fill=(55, 42, 20, 255))
    draw.text((x, y), text, font=fnt, fill=fill, stroke_width=sw, stroke_fill=INK)
    return base


def banner_word_center(base, y, text, fnt, fill=YELLOW, sw=10, shift=(7, 7)):
    draw = ImageDraw.Draw(base)
    tw, th, box = text_size(draw, text, fnt, sw)
    x = (base.width - tw) // 2 - box[0]
    banner_word(base, (x, y), text, fnt, fill, sw, shift)
    return th


def save_webp(im, path, quality=86):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if im.mode == "RGBA":
        im.save(path, "WEBP", quality=quality, method=6)
    else:
        im.convert("RGB").save(path, "WEBP", quality=quality, method=6)


def cover(im, w, h, focus=(0.5, 0.45)):
    im = im.convert("RGB")
    ratio = max(w / im.width, h / im.height)
    nw, nh = max(1, int(im.width * ratio)), max(1, int(im.height * ratio))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    x = int((nw - w) * focus[0])
    y = int((nh - h) * focus[1])
    x = max(0, min(nw - w, x))
    y = max(0, min(nh - h, y))
    return im.crop((x, y, x + w, y + h))


def contain_pad(im, w, h, bg=(255, 176, 70)):
    im = im.convert("RGB")
    ratio = min(w / im.width, h / im.height)
    nw, nh = int(im.width * ratio), int(im.height * ratio)
    fitted = im.resize((nw, nh), Image.Resampling.LANCZOS)
    blur = im.resize((w, h), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(22))
    overlay = Image.new("RGB", (w, h), bg)
    blur = Image.blend(blur, overlay, 0.28)
    blur.paste(fitted, ((w - nw) // 2, (h - nh) // 2))
    return blur


def trim(im, pad=6):
    bbox = im.split()[-1].getbbox()
    if not bbox:
        return im
    x0, y0, x1, y1 = bbox
    return im.crop((
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(im.width, x1 + pad),
        min(im.height, y1 + pad),
    ))


def fit(im, w, h, scale=0.92, valign="center"):
    im = trim(im, 4)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    max_w, max_h = int(w * scale), int(h * scale)
    ratio = min(max_w / im.width, max_h / im.height)
    nw, nh = max(1, int(im.width * ratio)), max(1, int(im.height * ratio))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    x = (w - nw) // 2
    if valign == "end":
        y = h - nh - int(h * (1 - scale) / 2)
    elif valign == "start":
        y = int(h * (1 - scale) / 2)
    else:
        y = (h - nh) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def grab_alpha(rgb, rect=None):
    bgr = cv2.cvtColor(np.array(rgb.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    if rect is None:
        m = max(2, int(min(h, w) * 0.02))
        rect = (m, m, w - 2 * m, h - 2 * m)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    ff = alpha.copy()
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 255)
    alpha = cv2.bitwise_or(alpha, cv2.bitwise_not(ff))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 20).astype(np.uint8), 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        alpha = np.where(labels == keep, alpha, 0).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    return rgba


def shade_bars(im, top=180, bot=210, strength=200):
    im = im.convert("RGBA")
    h, w = im.size
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    px = overlay.load()
    for i in range(top):
        a = int(strength * (1 - i / top) ** 1.35)
        for x in range(w):
            px[x, i] = (0, 0, 0, a)
    for i in range(bot):
        a = int(strength * (1 - i / bot) ** 1.15)
        y = h - 1 - i
        for x in range(w):
            px[x, y] = (0, 0, 0, a)
    return Image.alpha_composite(im, overlay)


def circle_photo(im, size):
    tile = cover(im, size, size, (0.48, 0.46)).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    tile.putalpha(mask)
    return tile


def panel(banner, name):
    return banner.crop(BOX[name])


def meme_caption(logo):
    """Classic top-and-bottom caption meme."""
    im = shade_bars(cover(logo, 1080, 1080, (0.5, 0.46)), 200, 230, 210)
    d = ImageDraw.Draw(im)
    stamp_center(d, 48, "STANLEY", font(108), 1080, WHITE, sw=14)
    stamp_center(d, 900, "THE CAT", font(96), 1080, WHITE, sw=13)
    stamp_center(d, 1010, "$STANLEY", font(40), 1080, YELLOW, sw=6)
    return im.convert("RGB")


def meme_split(logo, sit):
    """Two-panel comparison."""
    w, h = 1280, 720
    gap = 10
    left_w = (w - gap) // 2
    im = Image.new("RGB", (w, h), (27, 31, 75))
    im.paste(cover(logo, left_w, h, (0.48, 0.42)), (0, 0))
    im.paste(cover(sit, w - left_w - gap, h, (0.42, 0.42)), (left_w + gap, 0))
    im = im.convert("RGBA")
    bar = Image.new("RGBA", im.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(bar)
    bd.rounded_rectangle((28, 620, 360, 688), radius=18, fill=(8, 8, 12, 210))
    bd.rounded_rectangle((w - 360, 620, w - 28, 688), radius=18, fill=(8, 8, 12, 210))
    bd.rounded_rectangle((w // 2 - 150, 24, w // 2 + 150, 92), radius=18, fill=(255, 210, 31, 245))
    im = Image.alpha_composite(im, bar)
    d = ImageDraw.Draw(im)
    stamp_center_in(d, 28, 360, 632, "THE STRETCH", font(36), WHITE, sw=0)
    stamp_center_in(d, w - 360, w - 28, 632, "THE SIT", font(36), WHITE, sw=0)
    stamp_center_in(d, w // 2 - 150, w // 2 + 150, 36, "$STANLEY", font(40), INK, sw=0)
    return im.convert("RGB")


def stamp_center_in(draw, x0, x1, y, text, fnt, fill, stroke=BLACK, sw=0):
    tw, th, box = text_size(draw, text, fnt, sw)
    x = x0 + (x1 - x0 - tw) // 2 - box[0]
    stamp(draw, (x, y - box[1]), text, fnt, fill, stroke, sw)


def meme_labels(logo):
    """Callout meme: three labels on the stretch."""
    im = cover(logo, 1152, 864, (0.5, 0.48)).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    # leader lines land on the back, the face, and the long paw
    calls = [
        ((70, 70, 430, 150), (340, 250), "ORANGE BACK"),
        ((700, 70, 1080, 150), (1000, 360), "THE LOOK"),
        ((620, 690, 1100, 770), (980, 800), "LONG PAW"),
    ]
    for (x0, y0, x1, y1), (tx, ty), label in calls:
        ax, ay = (x0 + x1) // 2, (y0 + y1) // 2
        if ty > y1:
            ay = y1
        elif ty < y0:
            ay = y0
        d.line(((ax, ay), (tx, ty)), fill=(255, 255, 255, 230), width=6)
        d.ellipse((tx - 8, ty - 8, tx + 8, ty + 8), fill=YELLOW)
        d.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=(8, 8, 12, 220))
    im = Image.alpha_composite(im, overlay)
    d = ImageDraw.Draw(im)
    stamp_center_in(d, 70, 430, 86, "ORANGE BACK", font(36), YELLOW, sw=0)
    stamp_center_in(d, 700, 1080, 86, "THE LOOK", font(36), YELLOW, sw=0)
    stamp_center_in(d, 620, 1100, 706, "LONG PAW", font(36), YELLOW, sw=0)
    return im.convert("RGB")


def frame_photo(src, w, h):
    """Show the whole photo, with a blurred fill behind portrait frames."""
    src = src.convert("RGB")
    ratio = min(w / src.width, h / src.height)
    nw, nh = max(1, int(src.width * ratio)), max(1, int(src.height * ratio))
    fitted = src.resize((nw, nh), Image.Resampling.LANCZOS)
    blur = cover(src, w, h, (0.5, 0.5)).filter(ImageFilter.GaussianBlur(14))
    dim = Image.new("RGB", (w, h), (12, 14, 28))
    blur = Image.blend(blur, dim, 0.35)
    blur.paste(fitted, ((w - nw) // 2, (h - nh) // 2))
    return blur


def meme_comic(look, logo, tap):
    """Three-panel vertical comic."""
    w, h = 864, 1152
    im = Image.new("RGB", (w, h), (27, 31, 75))
    panels = [
        (look, "THE LOOK"),
        (logo, "THE STRETCH"),
        (tap, "THE TAP"),
    ]
    ph = 340
    gap = 8
    for i, (src, label) in enumerate(panels):
        y = 16 + i * (ph + gap)
        tile = frame_photo(src, w - 32, ph - 64)
        im.paste(tile, (16, y))
        d = ImageDraw.Draw(im)
        d.rectangle((16, y + ph - 64, w - 16, y + ph), fill=(8, 8, 12))
        stamp_center_in(d, 16, w - 16, y + ph - 52, label, font(36), YELLOW, sw=0)
    d = ImageDraw.Draw(im)
    stamp_center(d, 1068, "$STANLEY", font(48), w, YELLOW, sw=0)
    return im.convert("RGB")


def meme_badge(logo):
    """Orange circle card, the banner's own type, as a square."""
    w = h = 1000
    im = Image.new("RGBA", (w, h), ORANGE)
    banner_word_center(im, 36, "STANLEY", font(108), sw=12, shift=(8, 8))
    banner_word_center(im, 160, "THE CAT", font(64), sw=8, shift=(6, 6))
    circ = circle_photo(logo, 520)
    ring = Image.new("RGBA", im.size, (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    cx, cy = w // 2, 530
    rd.ellipse((cx - 276, cy - 276, cx + 276, cy + 276), fill=INK)
    im = Image.alpha_composite(im, ring)
    im.paste(circ, (cx - 260, cy - 260), circ)
    banner_word_center(im, 860, "$STANLEY", font(72), sw=9, shift=(6, 6))
    return im.convert("RGB")


def meme_film(look, soap, up, tap):
    """Four-frame film strip from the banner photos."""
    w, h = 1400, 788
    im = Image.new("RGBA", (w, h), PAPER)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w, 118), fill=ORANGE)
    banner_word(im, (36, 22), "THE BOARD", font(58), sw=8, shift=(5, 5))
    banner_word(im, (980, 28), "$STANLEY", font(48), sw=7, shift=(5, 5))
    frames = [
        (look, "LOOK", (0.5, 0.42)),
        (soap, "SOAP", (0.5, 0.45)),
        (up, "UP", (0.5, 0.48)),
        (tap, "TAP", (0.5, 0.4)),
    ]
    fw, fh = 300, 500
    gap = 28
    total = 4 * fw + 3 * gap
    x = (w - total) // 2
    y = 160
    for src, label, focus in frames:
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((x - 6, y - 6, x + fw + 6, y + fh + 6), radius=18, fill=INK)
        tile = cover(src, fw, fh - 70, focus).convert("RGBA")
        mask = Image.new("L", tile.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, fw, fh - 70), radius=12, fill=255)
        tile.putalpha(mask)
        im.paste(tile, (x, y), tile)
        d = ImageDraw.Draw(im)
        stamp_center_in(d, x, x + fw, y + fh - 58, label, font(36), YELLOW, sw=0)
        x += fw + gap
    return im.convert("RGB")


def thumb_for(im, tw, th):
    return cover(im, tw, th, (0.5, 0.42))


def draw_solana(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    shapes = [
        ([(64.6, 3.8), (73.8, 0), (391.2, 0), (395.8, 11.1), (333.1, 73.8), (323.9, 77.6), (6.5, 77.6), (1.9, 66.5)], (153, 69, 255, 255)),
        ([(333.1, 120.1), (323.9, 116.3), (6.5, 116.3), (1.9, 127.4), (64.6, 190.1), (73.8, 193.9), (391.2, 193.9), (395.8, 182.8)], (135, 82, 243, 255)),
        ([(64.6, 237.9), (73.8, 234.1), (391.2, 234.1), (395.8, 245.2), (333.1, 307.9), (323.9, 311.7), (6.5, 311.7), (1.9, 300.6)], (20, 241, 149, 255)),
    ]
    pad = size * 0.14
    s = min((size - 2 * pad) / 398, (size - 2 * pad) / 312)
    ox = (size - 398 * s) / 2
    oy = (size - 312 * s) / 2
    for pts, color in shapes:
        draw.polygon([(ox + x * s, oy + y * s) for x, y in pts], fill=color)
    return img


def oval_face(im, size=640, focus=(0.5, 0.45)):
    tile = cover(im, size, size, focus).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((18, 18, size - 19, size - 19), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(4))
    tile.putalpha(mask)
    return tile


def oval_tall(im, w=520, h=760, focus=(0.5, 0.45)):
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    side = int(min(w, h) * 0.92)
    face = oval_face(im, side, focus)
    canvas.paste(face, ((w - side) // 2, (h - side) // 2), face)
    return canvas


def on_orange(im, w=640, h=480):
    bg = Image.new("RGBA", (w, h), ORANGE)
    fitted = fit(im, w, h, 0.9, "center")
    bg.paste(fitted, (0, 0), fitted)
    return bg.convert("RGB")


def main():
    logo = Image.open(SRC_LOGO).convert("RGB")
    banner = Image.open(SRC_BANNER).convert("RGB")
    shots = {name: panel(banner, name) for name in BOX}

    brand = os.path.join(ROOT, "imagery", "brand")
    save_webp(logo.resize((512, 512), Image.Resampling.LANCZOS), os.path.join(brand, "logo.webp"), 90)
    logo.resize((512, 512), Image.Resampling.LANCZOS).save(os.path.join(brand, "icon-512.png"), "PNG")
    logo.resize((192, 192), Image.Resampling.LANCZOS).save(os.path.join(brand, "icon-192.png"), "PNG")
    logo.resize((180, 180), Image.Resampling.LANCZOS).save(os.path.join(brand, "apple-touch-icon.png"), "PNG")
    logo.resize((32, 32), Image.Resampling.LANCZOS).save(os.path.join(brand, "favicon-32.png"), "PNG")
    logo.resize((256, 256), Image.Resampling.LANCZOS).save(
        os.path.join(ROOT, "favicon.ico"), sizes=[(16, 16), (32, 32), (48, 48), (64, 64)]
    )
    contain_pad(banner, 1200, 630).save(os.path.join(brand, "og.jpg"), "JPEG", quality=88, optimize=True)
    save_webp(draw_solana(256), os.path.join(brand, "sol.webp"), 90)

    scenes = os.path.join(ROOT, "imagery", "scenes")
    save_webp(banner, os.path.join(scenes, "hero.webp"), 86)
    save_webp(logo.resize((500, 500), Image.Resampling.LANCZOS), os.path.join(scenes, "mark.webp"), 88)
    save_webp(banner, os.path.join(scenes, "season2.webp"), 86)

    stretch = grab_alpha(logo, (24, 64, 452, 420))

    altai = os.path.join(ROOT, "imagery", "altai")
    pieces = {
        "head": oval_tall(shots["look"], focus=(0.48, 0.46)),
        "love": oval_tall(shots["blep"], focus=(0.4, 0.55)),
        "x": oval_tall(shots["up"], focus=(0.55, 0.5)),
        "body": fit(stretch, 520, 740, 0.98, "end"),
        "sit": fit(stretch, 480, 740, 0.96, "end"),
        "dex": fit(stretch, 480, 740, 0.96, "end"),
        "x-bite": fit(stretch, 640, 640, 0.92),
        "dex-bite": oval_face(shots["look"], 640, (0.5, 0.45)),
    }
    for name, im in pieces.items():
        save_webp(im, os.path.join(altai, name + ".webp"))
    nail = os.path.join(ROOT, "imagery", "nailoong")
    for name in pieces:
        Image.open(os.path.join(altai, name + ".webp")).save(
            os.path.join(nail, name + ".webp"), "WEBP", quality=86, method=6
        )

    memes = {
        1: meme_caption(logo),
        2: meme_split(logo, shots["sit"]),
        3: meme_labels(logo),
        4: meme_comic(shots["look"], logo, shots["tap"]),
        5: meme_badge(logo),
        6: meme_film(shots["look"], shots["soap"], shots["up"], shots["tap"]),
    }
    thumbs = {1: (640, 640), 2: (1138, 640), 3: (853, 640), 4: (480, 640), 5: (640, 640), 6: (1138, 640)}
    meme_dir = os.path.join(ROOT, "imagery", "memes")
    thumb_dir = os.path.join(meme_dir, "thumbs")
    cycle = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 1, 20: 2}
    for n, src_n in cycle.items():
        img = memes[src_n]
        save_webp(img, os.path.join(meme_dir, f"{n}.webp"), 86)
        tw, th = thumbs[src_n]
        save_webp(thumb_for(img, tw, th), os.path.join(thumb_dir, f"{n}.webp"), 80)

    prev = os.path.join(ROOT, "assets")
    on_orange(stretch, 900, 520).save(os.path.join(prev, "preview-cutouts.jpg"), "JPEG", quality=88)
    memes[1].save(os.path.join(prev, "preview-meme1.jpg"), "JPEG", quality=88)
    memes[4].save(os.path.join(prev, "preview-meme4.jpg"), "JPEG", quality=88)
    print("done", {k: v.size for k, v in memes.items()})


if __name__ == "__main__":
    main()
