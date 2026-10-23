"""Build Altaius site images from the logo, banner, and keyed cutouts."""
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import os

ROOT = r"E:\pack\solana\Altai"
FONT = os.path.join(ROOT, "assets", "LilitaOne-Regular.ttf")
SRC_LOGO = os.path.join(ROOT, "logo-1.png")
SRC_BANNER = os.path.join(ROOT, "banner-1.png")
HEAD = r"C:\Users\Alita\.cursor\projects\e-pack-solana-Altai\assets\altaius-head.jpg"
BODY = r"C:\Users\Alita\.cursor\projects\e-pack-solana-Altai\assets\altaius-body.jpg"
SHIELD = r"C:\Users\Alita\.cursor\projects\e-pack-solana-Altai\assets\altaius-shield.jpg"

INK = (27, 31, 75, 255)
YELLOW = (255, 210, 31, 255)
PINK = (255, 45, 149, 255)
PAPER = (251, 247, 239, 255)
WHITE = (255, 255, 255, 255)


def font(size):
    return ImageFont.truetype(FONT, size)


def key_green(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    score = np.minimum(g - r, g - b)
    greenish = (g > 80) & (g > r + 18) & (g > b + 18)
    # 0 = fully green background, 255 = subject
    alpha = np.where(greenish, np.clip((70 - score) * (255.0 / 36.0), 0, 255), 255).astype(np.uint8)
    # despill green fringe on the subject edge
    edge = greenish | ((g > r + 8) & (g > b + 8) & (g > 60))
    g2 = g.copy()
    cap = np.maximum(r, b)
    g2 = np.where(edge & (g2 > cap), cap, g2)
    out = np.dstack([
        np.clip(r, 0, 255).astype(np.uint8),
        np.clip(g2, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8),
        alpha,
    ])
    im = Image.fromarray(out, "RGBA")
    # soften the matte a touch so jpeg fringe does not sparkle
    a = im.split()[-1].filter(ImageFilter.GaussianBlur(0.6))
    im.putalpha(a)
    return im


def trim(im, pad=8):
    bbox = im.split()[-1].getbbox()
    if not bbox:
        return im
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(im.width, x1 + pad)
    y1 = min(im.height, y1 + pad)
    return im.crop((x0, y0, x1, y1))


def fit(im, w, h, scale=0.92, valign="center"):
    im = trim(im, 4)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    max_w = int(w * scale)
    max_h = int(h * scale)
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


def save_webp(im, path, quality=86):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, "WEBP", quality=quality, method=6)


def cover(im, w, h, focus=(0.5, 0.42)):
    im = im.convert("RGB")
    ratio = max(w / im.width, h / im.height)
    nw, nh = int(im.width * ratio), int(im.height * ratio)
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    x = int((nw - w) * focus[0])
    y = int((nh - h) * focus[1])
    x = max(0, min(nw - w, x))
    y = max(0, min(nh - h, y))
    return im.crop((x, y, x + w, y + h))


def contain_pad(im, w, h, bg=(18, 12, 8)):
    """Fit a wide banner into a taller frame without cutting the tiger."""
    im = im.convert("RGB")
    ratio = min(w / im.width, h / im.height)
    nw, nh = int(im.width * ratio), int(im.height * ratio)
    fitted = im.resize((nw, nh), Image.Resampling.LANCZOS)
    blur = im.resize((w, h), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(18))
    overlay = Image.new("RGB", (w, h), bg)
    blur = Image.blend(blur, overlay, 0.35)
    x = (w - nw) // 2
    y = (h - nh) // 2
    blur.paste(fitted, (x, y))
    return blur


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


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1], box


def stamp(draw, xy, text, fnt, fill, stroke=INK, sw=10, shadow=None):
    x, y = xy
    if shadow:
        draw.text((x + shadow[0], y + shadow[1]), text, font=fnt, fill=shadow[2])
    draw.text((x, y), text, font=fnt, fill=fill, stroke_width=sw, stroke_fill=stroke)


def dots(base, step=22, radius=2.2, color=(255, 45, 149, 90), box=None):
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = box or (0, 0, base.width, base.height)
    y = y0 + 10
    while y < y1:
        x = x0 + 10
        while x < x1:
            d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            x += step
        y += step
    return Image.alpha_composite(base, overlay)


def rounded_paste(base, im, box, radius=36, shadow=(10, 10, PINK)):
    x, y, w, h = box
    if shadow:
        sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        sd.rounded_rectangle((x + shadow[0], y + shadow[1], x + w + shadow[0], y + h + shadow[1]), radius=radius, fill=shadow[2])
        base = Image.alpha_composite(base, sh)
    tile = cover(im, w, h).convert("RGBA")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    tile.putalpha(mask)
    base.paste(tile, (x, y), tile)
    return base


def meme_square():
    w = h = 1080
    im = Image.new("RGBA", (w, h), PAPER)
    im = dots(im, step=26, radius=3, color=(255, 45, 149, 110), box=(0, 0, 560, h))
    logo = Image.open(SRC_LOGO)
    im = rounded_paste(im, logo, (500, 90, 500, 900), radius=48, shadow=(16, 16, PINK))
    d = ImageDraw.Draw(im)
    stamp(d, (48, 250), "ALTAIUS", font(92), YELLOW, sw=12, shadow=(8, 8, PINK))
    stamp(d, (48, 390), "$Altaius", font(78), YELLOW, sw=10, shadow=(7, 7, PINK))
    stamp(d, (48, 540), "MAXIMUS", font(54), PINK, stroke=INK, sw=8)
    return im.convert("RGB")


def meme_wide_chain():
    banner = Image.open(SRC_BANNER).convert("RGBA")
    im = banner.resize((1280, 720), Image.Resampling.LANCZOS)
    bar = Image.new("RGBA", im.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(bar)
    bd.rounded_rectangle((36, 580, 520, 684), radius=22, fill=(27, 31, 75, 230))
    im = Image.alpha_composite(im, bar)
    d = ImageDraw.Draw(im)
    stamp(d, (58, 598), "ON SOLANA", font(48), YELLOW, sw=0)
    return im.convert("RGB")


def meme_gold():
    logo = cover(Image.open(SRC_LOGO), 1152, 864, focus=(0.48, 0.42)).convert("RGBA")
    bar = Image.new("RGBA", logo.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(bar)
    bd.rectangle((0, 700, 1152, 864), fill=(27, 31, 75, 220))
    im = Image.alpha_composite(logo, bar)
    d = ImageDraw.Draw(im)
    stamp(d, (48, 730), "HE STANDS", font(62), YELLOW, sw=0)
    stamp(d, (48, 798), "IN THE GOLD", font(42), WHITE, sw=0)
    return im.convert("RGB")


def meme_portrait():
    w, h = 864, 1152
    im = Image.new("RGBA", (w, h), PAPER)
    im = dots(im, step=24, radius=2.6, color=(255, 45, 149, 100))
    head = trim(key_green(HEAD), 2)
    arr = np.array(head)
    start = int(arr.shape[0] * 0.82)
    ramp = np.linspace(1, 0, arr.shape[0] - start)
    arr[start:, :, 3] = (arr[start:, :, 3].astype(np.float32) * ramp[:, None]).astype(np.uint8)
    head = Image.fromarray(arr, "RGBA")
    ratio = min(760 / head.width, 780 / head.height)
    nw, nh = int(head.width * ratio), int(head.height * ratio)
    head = head.resize((nw, nh), Image.Resampling.LANCZOS)
    im.paste(head, ((w - nw) // 2, 20), head)
    d = ImageDraw.Draw(im)
    stamp(d, (40, 900), "ALTAIUS", font(78), YELLOW, sw=10, shadow=(7, 7, PINK))
    stamp(d, (40, 990), "MAXIMUS", font(64), YELLOW, sw=9, shadow=(6, 6, PINK))
    stamp(d, (40, 1072), "$Altaius", font(42), PINK, stroke=INK, sw=6)
    return im.convert("RGB")


def meme_mark():
    w = h = 1000
    im = Image.new("RGBA", (w, h), PAPER)
    im = dots(im, step=28, radius=3, color=(255, 45, 149, 80))
    logo = Image.open(SRC_LOGO)
    im = rounded_paste(im, logo, (150, 40, 700, 700), radius=40, shadow=(14, 14, PINK))
    d = ImageDraw.Draw(im)
    stamp(d, (48, 800), "ALTAIUS", font(86), YELLOW, sw=11, shadow=(8, 8, PINK))
    stamp(d, (48, 900), "$Altaius", font(52), INK, stroke=YELLOW, sw=6)
    return im.convert("RGB")


def meme_arena():
    banner = Image.open(SRC_BANNER).convert("RGB")
    im = cover(banner, 1400, 788, focus=(0.5, 0.45)).convert("RGBA")
    bar = Image.new("RGBA", im.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(bar)
    bd.rounded_rectangle((36, 36, 620, 150), radius=24, fill=(27, 31, 75, 225))
    im = Image.alpha_composite(im, bar)
    d = ImageDraw.Draw(im)
    stamp(d, (58, 52), "THE WIDE ARENA", font(42), YELLOW, sw=0)
    stamp(d, (58, 100), "$Altaius", font(32), WHITE, sw=0)
    return im.convert("RGB")


def thumb_for(im, tw, th):
    return cover(im, tw, th, focus=(0.5, 0.4))


def main():
    logo = Image.open(SRC_LOGO).convert("RGB")
    banner = Image.open(SRC_BANNER).convert("RGB")

    save_webp(logo.resize((512, 512), Image.Resampling.LANCZOS), os.path.join(ROOT, "imagery", "brand", "logo.webp"), 90)
    logo.resize((512, 512), Image.Resampling.LANCZOS).save(os.path.join(ROOT, "imagery", "brand", "icon-512.png"), "PNG")
    logo.resize((192, 192), Image.Resampling.LANCZOS).save(os.path.join(ROOT, "imagery", "brand", "icon-192.png"), "PNG")
    logo.resize((180, 180), Image.Resampling.LANCZOS).save(os.path.join(ROOT, "imagery", "brand", "apple-touch-icon.png"), "PNG")
    logo.resize((32, 32), Image.Resampling.LANCZOS).save(os.path.join(ROOT, "imagery", "brand", "favicon-32.png"), "PNG")
    ico = logo.resize((256, 256), Image.Resampling.LANCZOS)
    ico.save(os.path.join(ROOT, "favicon.ico"), sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])

    og = contain_pad(banner, 1200, 630)
    og.save(os.path.join(ROOT, "imagery", "brand", "og.jpg"), "JPEG", quality=88, optimize=True)

    sol = draw_solana(256)
    save_webp(sol, os.path.join(ROOT, "imagery", "brand", "sol.webp"), 90)

    save_webp(banner, os.path.join(ROOT, "imagery", "scenes", "hero.webp"), 86)
    save_webp(logo.resize((500, 500), Image.Resampling.LANCZOS), os.path.join(ROOT, "imagery", "scenes", "mark.webp"), 88)
    save_webp(banner, os.path.join(ROOT, "imagery", "scenes", "season2.webp"), 86)

    head = key_green(HEAD)
    body = key_green(BODY)
    shield = key_green(SHIELD)

    head_fit = fit(head, 584, 819, 0.96)
    body_fit = fit(body, 405, 740, 0.98, valign="end")
    shield_sq = fit(shield, 640, 640, 0.94)
    head_sq = fit(head, 640, 640, 0.96)

    altai = os.path.join(ROOT, "imagery", "altai")
    save_webp(head_fit, os.path.join(altai, "head.webp"))
    save_webp(head_fit, os.path.join(altai, "love.webp"))
    save_webp(head_fit, os.path.join(altai, "x.webp"))
    save_webp(body_fit, os.path.join(altai, "body.webp"))
    save_webp(body_fit, os.path.join(altai, "sit.webp"))
    save_webp(body_fit, os.path.join(altai, "dex.webp"))
    save_webp(shield_sq, os.path.join(altai, "x-bite.webp"))
    save_webp(head_sq, os.path.join(altai, "dex-bite.webp"))

    nail = os.path.join(ROOT, "imagery", "nailoong")
    for name in ("head", "love", "x", "body", "sit", "dex", "x-bite", "dex-bite"):
        src = os.path.join(altai, name + ".webp")
        dst = os.path.join(nail, name + ".webp")
        Image.open(src).save(dst, "WEBP", quality=86, method=6)

    memes = {
        1: meme_square(),
        2: meme_wide_chain(),
        3: meme_gold(),
        4: meme_portrait(),
        5: meme_mark(),
        6: meme_arena(),
    }
    thumbs = {
        1: (640, 640),
        2: (720, 405),
        3: (720, 540),
        4: (480, 640),
        5: (640, 640),
        6: (720, 405),
    }
    meme_dir = os.path.join(ROOT, "imagery", "memes")
    thumb_dir = os.path.join(meme_dir, "thumbs")
    cycle = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 1, 20: 2}
    for n, src_n in cycle.items():
        img = memes[src_n]
        save_webp(img, os.path.join(meme_dir, f"{n}.webp"), 86)
        tw, th = thumbs[src_n]
        save_webp(thumb_for(img, tw, th), os.path.join(thumb_dir, f"{n}.webp"), 80)

    # preview cutouts on the site orange so edges can be checked
    prev = Image.new("RGBA", (900, 740), (255, 122, 44, 255))
    b = fit(body, 360, 700, 0.98)
    h = fit(head, 280, 400, 0.96)
    prev.paste(b, (20, 20), b)
    prev.paste(h, (420, 40), h)
    prev.convert("RGB").save(os.path.join(ROOT, "assets", "preview-cutouts.jpg"), "JPEG", quality=88)
    memes[1].save(os.path.join(ROOT, "assets", "preview-meme1.jpg"), "JPEG", quality=88)
    memes[4].save(os.path.join(ROOT, "assets", "preview-meme4.jpg"), "JPEG", quality=88)
    print("done")


if __name__ == "__main__":
    main()
