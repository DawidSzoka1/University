"""Generuje krótkie filmy demonstracyjne (animacja terminala) do danych testowych."""
import io
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FFMPEG = str(Path.home() / ".cache/ms-playwright/ffmpeg-1011/ffmpeg-linux")
FONT = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 26)
TITLE = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 20)
W, H, FPS = 640, 360, 10
BG, FG, PROMPT, MUTED = (17, 24, 39), (255, 251, 235), (217, 119, 6), (156, 163, 175)


def frames(title, script):
    """script: lista (komenda, wynik) — komenda „pisana” znak po znaku."""
    done = []
    def draw(lines, cursor):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 34], fill=(31, 41, 55))
        d.text((16, 7), title, font=TITLE, fill=MUTED)
        y = 52
        for kind, text in lines[-9:]:
            if kind == "cmd":
                d.text((16, y), "$", font=FONT, fill=PROMPT)
                d.text((44, y), text, font=FONT, fill=FG)
            else:
                d.text((16, y), text, font=FONT, fill=MUTED)
            y += 34
        if cursor is not None:
            d.rectangle([cursor[0], cursor[1], cursor[0] + 14, cursor[1] + 28], fill=FG)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=88); return buf.getvalue()
    for _ in range(8):
        yield draw(done, (44, 52))
    for command, output in script:
        for i in range(len(command) + 1):
            y = 52 + 34 * min(len(done), 8)
            yield draw(done + [("cmd", command[:i])], (44 + 15.6 * i, y))
        done.append(("cmd", command))
        for _ in range(4):
            yield draw(done, None)
        for line in output:
            done.append(("out", line))
        for _ in range(12):
            yield draw(done, None)
    for _ in range(15):
        yield draw(done, None)


def encode(path, title, script):
    # Minimalny ffmpeg z Playwrighta ma tylko protokół file: — strumień MJPEG przez plik tymczasowy
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mjpeg", delete=False) as stream:
        for frame in frames(title, script):
            stream.write(frame)
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-r", str(FPS),
         "-i", f"file:{stream.name}", "-c:v", "vp8", "-b:v", "400k", "-r", str(FPS), f"file:{path}"],
        check=True,
    )
    Path(stream.name).unlink()
    print(path, Path(path).stat().st_size, "B")


out = Path(sys.argv[1])
encode(out / "python_hello_world.webm", "Terminal — pierwszy program w Pythonie", [
    ("python3 --version", ["Python 3.13.0"]),
    ('echo \'print("Hello, world!")\' > hello.py', []),
    ("python3 hello.py", ["Hello, world!"]),
])
encode(out / "linux_pierwsze_polecenia.webm", "Terminal — pierwsze polecenia w Linuksie", [
    ("pwd", ["/home/uczen"]),
    ("mkdir projekt && cd projekt", []),
    ("touch notatki.txt && ls -l", ["-rw-r--r-- 1 uczen uczen 0 notatki.txt"]),
])
