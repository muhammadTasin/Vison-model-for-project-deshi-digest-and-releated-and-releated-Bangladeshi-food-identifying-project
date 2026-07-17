# README chart assets

The SVG figures in `assets/readme/` are generated from the compact, versioned
Stage-3 reports under `evaluation/reports/`.

Regenerate them with:

```bash
python3 evaluation/render_readme_charts.py
```

The renderer uses only the Python standard library. Do not edit generated SVGs
by hand; update the underlying report or renderer, regenerate, and verify that
the benchmark still totals 95 images, 83 correct predictions, and 12 errors.
