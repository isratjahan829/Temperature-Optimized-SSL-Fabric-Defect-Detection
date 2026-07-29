# Dataset

## Source

The corpus was collected directly from the **Chenab Textile** production line and
is published by the authors at <https://tinyurl.com/488uhkhy>
(~2 800 images). Unlike TILDA, TIANCHI or AITEX, the images are *as-produced*:
unflattened fabric, varied angles and backlighting, real noise and blur, and
visible background elements from the production setup. Plain, regularly printed
and irregularly printed fabrics are all represented. A portion of the minority
classes was supplemented with visually similar public Roboflow datasets.

## Expected layout

One directory per class; any of `.jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp` is read.

```
data/fabric/
├── baekra/
├── color_issues/
├── contamination/
├── cut/
├── gray_stitch/
├── selvet/
└── stain/
```

Directory names are the class labels. `FabricDefectDataset` sorts them
alphabetically, so the label ordering is deterministic across runs.

## Classes

| Class | Description |
| --- | --- |
| `baekra` | Weave/structural irregularity spanning a large region of the fabric |
| `color_issues` | Non-uniform colour: spots, discoloration, shade variation |
| `contamination` | Foreign matter (fibre, dust, oil) embedded in the fabric |
| `cut` | Torn or severed yarn creating a linear break |
| `gray_stitch` | Off-colour stitching left in the finished fabric |
| `selvet` | Defective selvedge along the fabric edge |
| `stain` | Localised discoloration from oil, water or dye — the majority class |

## Class imbalance

`stain` accounts for more than half the corpus while `baekra`, `gray_stitch` and
`selvet` each stay under 10 %. Three mechanisms in this repository counter that:

1. **Augmentation** — rotation, flips, scaling, colour jitter/drop, Gaussian
   noise and blur (`fabricssl.data.transforms`).
2. **Inverse-frequency sampling** — `data.balanced_sampler: true` draws minority
   classes more often (`class_balanced_sampler`).
3. **Weighted loss** — class-weighted cross-entropy during fine-tuning, with
   `FocalLoss` available for harder, rarer defects.

Splits are always **stratified**, so every class keeps its proportion in
train/val/test. The paper compares 75/15/10, 80/10/10 and 70/20/10; all three
are available via `data.split` or `fabricssl.pipeline.split_sweep`.

## Synthetic stand-in

When the real corpus is unavailable, generate a structurally identical
replacement:

```bash
python -m fabricssl.cli make-sample-data --root data/sample --images-per-class 40 --image-size 96
```

Each class gets a woven-looking background with a class-specific defect
primitive (blob, horizontal line, rectangle, vertical line, arc). It exists to
exercise the code paths and CI — **never** to produce comparable accuracy
numbers.

## Preprocessing

| Stage | Transform |
| --- | --- |
| SSL views | RandomResizedCrop(0.3–1.0), H/V flip, rotation ±30°, ColorJitter(0.8/0.8/0.8/0.2), grayscale p=0.2, Gaussian blur, Gaussian noise, optional Sobel blend |
| Fine-tuning | RandomResizedCrop(0.7–1.0), horizontal flip, rotation ±15°, mild jitter |
| Evaluation | Resize(1.14×) → CenterCrop → normalise |

Normalisation uses ImageNet statistics
(`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`).

## Ethics and licensing

The images come from an operating factory. Redistribute them only under the
terms set by the dataset authors, and remove any frames that identify workers
before sharing.
