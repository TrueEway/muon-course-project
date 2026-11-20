"""Render every report page with Poppler and make six-page review sheets."""
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / 'tmp' / 'pdfs'
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pdftoppm', '-r', '110', '-png', str(ROOT / 'output' / 'pdf' / 'main.pdf'),
                    str(target / 'page')], check=True)
    pages = sorted(target.glob('page-*.png'))
    for offset in range(0, len(pages), 6):
        sheet = Image.new('RGB', (1200, 1170), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, page in enumerate(pages[offset:offset + 6]):
            with Image.open(page) as source:
                thumb = source.convert('RGB')
                thumb.thumbnail((384, 550))
                x, y = (index % 3) * 400 + 8, (index // 3) * 585 + 24
                sheet.paste(thumb, (x, y))
                draw.text((x, y - 18), page.stem, fill='black')
        sheet.save(target / f'review-{offset // 6 + 1}.png')
    print(f'Rendered {len(pages)} pages into {target}')


if __name__ == '__main__':
    main()
