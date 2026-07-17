# Dataset Provenance and Governance

## Current evidence status

The repository was empty before this documentation work. The exported Stage-3
archive contains training and validation manifests, but manifest presence alone
does not prove the original source, license, redistribution permission,
deduplication status, or final inclusion of every dataset considered during the
broader project. No source images are redistributed here.

Accordingly, every source below is recorded conservatively as **considered or
prepared; Stage-3 inclusion not independently verified**. The maintainer should
replace unknown fields only after checking the source landing page, license
file, download metadata, and exact Stage-3 manifest lineage.

## Source register

| Source | URL | License | Images/classes | Stage-3 evidence | Redistribution and attribution |
|---|---|---|---|---|---|
| Mendeley dataset `tczzndbprx`, version 1 | [Landing page](https://data.mendeley.com/datasets/tczzndbprx/1) | Not verified from repository artifacts | Not verified | Considered or prepared; inclusion not independently verified | Do not redistribute until terms and citation are verified |
| Mendeley dataset `xh3ghf3jbg`, version 2 | [Landing page](https://data.mendeley.com/datasets/xh3ghf3jbg/2) | Not verified from repository artifacts | Not verified | Considered or prepared; inclusion not independently verified | Do not redistribute until terms and citation are verified |
| Mendeley dataset `gwrf5gphzw`, version 1 | [Landing page](https://data.mendeley.com/datasets/gwrf5gphzw/1) | Not verified from repository artifacts | Not verified | Considered or prepared; inclusion not independently verified | Do not redistribute until terms and citation are verified |
| Kaggle `wasiffuad/bengali-food` | [Landing page](https://www.kaggle.com/datasets/wasiffuad/bengali-food) | Not verified from repository artifacts | Not verified | Considered or prepared; inclusion not independently verified | Follow the dataset page and Kaggle terms; do not mirror by default |
| Mendeley dataset `r7df84yyyj`, version 1 | [Landing page](https://data.mendeley.com/datasets/r7df84yyyj/1) | Not verified from repository artifacts | Not verified | Considered or prepared; inclusion not independently verified | Do not redistribute until terms and citation are verified |
| Mendeley dataset `j2pnx2mwwk`, version 1 | [Landing page](https://data.mendeley.com/datasets/j2pnx2mwwk/1) | Not verified from repository artifacts | Not verified | Considered or prepared; inclusion not independently verified | Do not redistribute until terms and citation are verified |
| Kaggle `fayroztasnimrowza/bd-desserts` | [Landing page](https://www.kaggle.com/datasets/fayroztasnimrowza/bd-desserts) | Not verified from repository artifacts | Not verified | Considered or prepared; inclusion not independently verified | Follow the dataset page and Kaggle terms; do not mirror by default |

The local archive inspection also found Stage-3 manifest paths referring to a
Kaggle input named `deshi-food-original-image-archives` and an image tree named
`DeshiFoodBD`. Those path names are not sufficient to identify the legal source
or license, so they are not presented as verified dataset attribution.

## Required provenance record

Before the next release, create one immutable record per source with:

- Dataset name, version, owner, landing URL, and retrieval date
- Exact license text or identifier and required citation
- Download checksum and original archive filename
- Image count and class list before and after filtering
- Whether the source was considered, downloaded, tested, attached to a
  notebook, included in Stage 3, or excluded
- Redistribution permission for images, annotations, and derived manifests
- Mapping from original labels to canonical model labels
- Deduplication method and train/validation/benchmark split membership

## Data-quality risks

### Duplicates and leakage

Exact duplicates and visually near-identical files can inflate validation or
benchmark results if copies cross split boundaries. Compare cryptographic file
hashes first, then use perceptual hashes or embeddings for near-duplicate
review. Keep the fixed 95-image benchmark isolated from all training and
checkpoint-selection data.

### Class-name inconsistency

Historical artifacts use spellings such as `bangladeshi_biryani` and
`roshgolla`, whereas the public API uses `bangladeshi_biriyani` and
`roshogolla`. Normalize only through an explicit exact alias map and preserve
the original label in audit metadata.

### Label and domain mismatch

Manually inspect wrong labels, ambiguous mixed plates, raw ingredients filed as
cooked dishes, and restaurant packaging or text that may leak the answer.
Regional recipes and presentation differences need documented label policy.

### License compatibility

A dataset being publicly downloadable does not automatically permit
redistribution, commercial use, derivative datasets, or model training. Check
compatibility among every dataset license, the base-model terms, and the future
repository license. When permission is unclear, distribute download scripts and
checksums rather than images.

## Privacy

Exclude private user photos, faces or personal documents captured incidentally,
location metadata, and personal nutrition records unless explicit consent and
governance are in place. Strip unnecessary EXIF metadata and provide a deletion
process for contributed data.
