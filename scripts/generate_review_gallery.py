"""Generate audit/review/gallery.html for accepted candidate manual inspection.

Requirements:
- Show all accepted candidate crops
- Responsive grid layout
- Dark theme
- Candidate image preview
- Candidate ID
- Detection method
- Bounding box dimensions
- Click image to open full size (lightbox / new tab)
- Sort by candidate_id
- Read-only operation
"""

import json
import os
from typing import Any


def main() -> None:
    review_dir = os.path.join("audit", "review")
    accepted_dir = os.path.join(review_dir, "accepted")
    output_html_path = os.path.join(review_dir, "gallery.html")

    candidates: list[dict[str, Any]] = []

    if os.path.exists(accepted_dir):
        json_files = [f for f in os.listdir(accepted_dir) if f.endswith(".json")]
        for jf in json_files:
            jpath = os.path.join(accepted_dir, jf)
            with open(jpath, encoding="utf-8") as f:
                data = json.load(f)

            # Match corresponding png image file
            png_name = jf.replace(".json", ".png")
            png_path = os.path.join(accepted_dir, png_name)
            if os.path.exists(png_path):
                data["img_rel_path"] = f"accepted/{png_name}"
            else:
                data["img_rel_path"] = ""

            candidates.append(data)

    # Fallback to audit/detections.json if review/accepted is empty
    if not candidates and os.path.exists(os.path.join("audit", "detections.json")):
        with open(os.path.join("audit", "detections.json"), encoding="utf-8") as f:
            det_data = json.load(f)
        for fr in det_data.get("frames", []):
            for c in fr.get("candidates", []):
                if c.get("accepted"):
                    c_copy = dict(c)
                    crop_file = c.get("crop_file")
                    if crop_file and os.path.exists(os.path.join("audit", "crops", crop_file)):
                        c_copy["img_rel_path"] = f"../crops/{crop_file}"
                    else:
                        c_copy["img_rel_path"] = ""
                    candidates.append(c_copy)

    # Sort candidates strictly by candidate_id ascending
    candidates.sort(key=lambda c: (c.get("candidate_id", 0), c.get("method", "")))

    print(f"Collected {len(candidates)} accepted candidates for review gallery.")

    # Generate HTML Cards
    cards_html = []
    for c in candidates:
        cid = c.get("candidate_id", 0)
        method = c.get("method", "contour")
        bbox = c.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        area = c.get("area", w * h)
        img_src = c.get("img_rel_path", "")

        # Method badge styling
        method_badge_class = "badge-contour"
        if method == "combat_base":
            method_badge_class = "badge-combat-base"
        elif method == "hsv":
            method_badge_class = "badge-hsv"

        card = f"""
        <div class="card" data-candidate-id="{cid}" data-method="{method}">
            <div class="card-header">
                <span class="candidate-id">ID #{cid:04d}</span>
                <span class="badge {method_badge_class}">{method}</span>
            </div>
            <div class="card-body">
                <div class="image-wrapper" onclick="openLightbox('{img_src}', '{cid}', '{method}', '{w}x{h}')" title="Click to open full size">
                    {'<img src="' + img_src + '" alt="Candidate ' + str(cid) + '" loading="lazy" />' if img_src else '<div class="no-img">No Image Available</div>'}
                    <div class="zoom-overlay">🔍 Click to enlarge</div>
                </div>
                <div class="meta-info">
                    <div class="meta-row">
                        <span class="meta-label">BBox (X, Y):</span>
                        <span class="meta-val">({x}, {y})</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Dimensions (W × H):</span>
                        <span class="meta-val"><strong>{w} × {h} px</strong></span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Area:</span>
                        <span class="meta-val">{area:.0f} px²</span>
                    </div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DTA Detection Audit — Accepted Candidates Gallery</title>
    <style>
        :root {{
            --bg-color: #0f1117;
            --card-bg: #181b24;
            --card-header: #212532;
            --text-primary: #e6e8f0;
            --text-secondary: #9ea4b5;
            --accent-blue: #4fc3f7;
            --border-color: #2b3042;
            --badge-contour-bg: #f57f17;
            --badge-hsv-bg: #2e7d32;
            --badge-combat-bg: #c62828;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }}

        header {{
            max-width: 1400px;
            margin: 0 auto 24px auto;
            padding-bottom: 16px;
            border-bottom: 2px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}

        h1 {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent-blue);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .subtitle {{
            font-size: 0.95rem;
            color: var(--text-secondary);
        }}

        .stats-bar {{
            display: flex;
            gap: 16px;
            background: var(--card-bg);
            padding: 8px 16px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            font-size: 0.9rem;
        }}

        .stats-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .stats-value {{
            font-weight: bold;
            color: var(--accent-blue);
        }}

        .controls-bar {{
            max-width: 1400px;
            margin: 0 auto 20px auto;
            display: flex;
            gap: 12px;
            align-items: center;
        }}

        .search-input {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 0.9rem;
            width: 260px;
        }}

        .search-input:focus {{
            outline: none;
            border-color: var(--accent-blue);
        }}

        .grid-container {{
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 20px;
        }}

        .card {{
            background-color: var(--card-bg);
            border-radius: 10px;
            border: 1px solid var(--border-color);
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            flex-direction: column;
        }}

        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            border-color: #3b4259;
        }}

        .card-header {{
            background-color: var(--card-header);
            padding: 10px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
        }}

        .candidate-id {{
            font-weight: 700;
            font-size: 1.05rem;
            color: #ffffff;
            letter-spacing: 0.5px;
        }}

        .badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #ffffff;
        }}

        .badge-contour {{ background-color: var(--badge-contour-bg); }}
        .badge-hsv {{ background-color: var(--badge-hsv-bg); }}
        .badge-combat-base {{ background-color: var(--badge-combat-bg); }}

        .card-body {{
            padding: 14px;
            display: flex;
            flex-direction: column;
            flex: 1;
        }}

        .image-wrapper {{
            position: relative;
            background-color: #000000;
            border-radius: 6px;
            overflow: hidden;
            height: 140px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            border: 1px solid var(--border-color);
            margin-bottom: 12px;
        }}

        .image-wrapper img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            image-rendering: pixelated;
            transition: transform 0.2s ease;
        }}

        .image-wrapper:hover img {{
            transform: scale(1.08);
        }}

        .zoom-overlay {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.75);
            color: #ffffff;
            font-size: 0.75rem;
            text-align: center;
            padding: 4px;
            opacity: 0;
            transition: opacity 0.2s ease;
        }}

        .image-wrapper:hover .zoom-overlay {{
            opacity: 1;
        }}

        .no-img {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-style: italic;
        }}

        .meta-info {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-size: 0.85rem;
        }}

        .meta-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
            padding-bottom: 4px;
        }}

        .meta-label {{
            color: var(--text-secondary);
        }}

        .meta-val {{
            color: var(--text-primary);
            font-family: monospace;
        }}

        /* Lightbox Overlay */
        .lightbox {{
            display: none;
            position: fixed;
            z-index: 1000;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }}

        .lightbox.active {{
            display: flex;
        }}

        .lightbox-img {{
            max-width: 90%;
            max-height: 80%;
            object-fit: contain;
            border: 2px solid var(--accent-blue);
            border-radius: 8px;
            background: #000;
            image-rendering: pixelated;
        }}

        .lightbox-caption {{
            margin-top: 16px;
            color: #ffffff;
            font-size: 1.1rem;
            text-align: center;
        }}

        .lightbox-close {{
            position: absolute;
            top: 20px;
            right: 30px;
            color: #ffffff;
            font-size: 2rem;
            font-weight: bold;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>DTA Detection Audit — Accepted Gallery</h1>
            <div class="subtitle">Manual inspection of all accepted candidates sorted by candidate_id</div>
        </div>
        <div class="stats-bar">
            <div class="stats-item">Total Accepted: <span class="stats-value">{len(candidates)}</span></div>
        </div>
    </header>

    <div class="controls-bar">
        <input type="text" id="filterInput" class="search-input" placeholder="Filter by ID or method..." onkeyup="filterCards()" />
    </div>

    <div class="grid-container" id="galleryGrid">
        {''.join(cards_html)}
    </div>

    <!-- Lightbox Modal -->
    <div class="lightbox" id="lightboxModal" onclick="closeLightbox()">
        <span class="lightbox-close">&times;</span>
        <img class="lightbox-img" id="lightboxImg" src="" alt="Enlarged candidate" />
        <div class="lightbox-caption" id="lightboxCaption"></div>
    </div>

    <script>
        function openLightbox(src, cid, method, dims) {{
            if (!src) return;
            const modal = document.getElementById('lightboxModal');
            const img = document.getElementById('lightboxImg');
            const caption = document.getElementById('lightboxCaption');

            img.src = src;
            caption.innerHTML = `Candidate #${{cid}} (${{method}}) &bull; Dim: ${{dims}} px`;
            modal.classList.add('active');
        }}

        function closeLightbox() {{
            const modal = document.getElementById('lightboxModal');
            modal.classList.remove('active');
        }}

        function filterCards() {{
            const query = document.getElementById('filterInput').value.toLowerCase();
            const cards = document.querySelectorAll('.card');
            cards.forEach(card => {{
                const cid = card.getAttribute('data-candidate-id');
                const method = card.getAttribute('data-method');
                if (cid.includes(query) || method.toLowerCase().includes(query)) {{
                    card.style.display = 'flex';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}

        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') closeLightbox();
        }});
    </script>
</body>
</html>
"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_document)

    print(f"Successfully generated {output_html_path} with {len(candidates)} candidates.")


if __name__ == "__main__":
    main()
