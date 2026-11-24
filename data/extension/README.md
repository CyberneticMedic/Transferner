# Cultivated perennials for the label set

The user's answer recorded in D-044: fields and woods are the usual setting, but the label set should
also take **some cultivated species, perennials only, no annual crops**. Of the yard plants the user
photographed, most were recognised (fig, pomegranate, feijoa, rosemary, mint, asparagus, daylily,
muscadine, tomato, yaupon). Olive and pothos are not in the label set: the capture model named them
*Buxus sempervirens* (0.39) and *Ipomoea tricolor* (0.30).

The label set takes a species once it has 20 regional GBIF records (`configs/species.yaml`). Planted
species that rarely escape stay under that line (audit findings `capture_model-2`, `data_domain-03`,
`capture_open_set-1`, `capture_open_set-5` in `$TRANSFERNER_DATA/ops/audit-20261124.json`). This
directory lists the ones worth adding. **It does not change the label set**: the lead integrates it.

| File | What it holds |
|---|---|
| `perennial_candidates.csv` | Hand-curated input: 244 names, with common name, group, kind, habit and hardiness zone, curated priority, and why each one matters. |
| `cultivated_perennials.csv` | **Kept: 120 species** to add, safety-relevant toxic ones first. |
| `cultivated_perennials_excluded.csv` | The other 121, each with an `excluded_reason`. |
| `src/transferner/data/cultivated_perennials.py` | Builds the two output files from the input. Re-runs use the caches: `python -m transferner.data.cultivated_perennials [--min-tol 300] [--no-fetch]` |

## Criteria

A candidate is kept when all of these hold:

1. **Perennial and grown outdoors in SC/NC/GA/TN (USDA zones 6-9).** It must be a tree, shrub,
   vine, herbaceous perennial, bulb or tuber that a walker meets in yards, gardens, orchards and
   parks. These are excluded:
   - annual crops (pepper, onion, garlic, bean, parsley, dill);
   - houseplants (the lead's rule: only if commonly outdoors);
   - tender plants grown as annuals or in containers (key lime, lemongrass, avocado, passion fruit).

   Evidence comes from USDA PLANTS durations, the NC State page's Life Cycle field and the curated
   habit. A species is also dropped when every source that gives a life cycle (USDA or NCSU) says
   annual only.
2. **Not already a label species.** The same accepted GBIF key counts as in the set. A shared name
   counts only when a strict GBIF match resolves that name to this species. GBIF synonym lists keep
   misapplied and *pro parte* names, which made three false "already in the set" hits:
   - *Buxus sempervirens* is listed under *B. microphylla*;
   - *Vitis labrusca* is listed under *V. vinifera*;
   - *Passiflora incarnata* is listed under *P. edulis*.

   Infraspecific accepted names are rolled up to their species: *Actinidia deliciosa* is
   *A. chinensis* var. *deliciosa*, which is in the set.
3. **At least 300 usable TreeOfLife images**, counted in `raw/tol/plantae.parquet` with DuckDB under
   the same filters and curation tiers that `tol_select` uses for training:
   - names counted: the accepted name, plus each GBIF synonym that GBIF resolves back to the same
     species. For example, *Chrysanthemum indicum* images are not counted for *C. morifolium*;
   - `tol_top_name` is the name with the most images. That is the name BioCLIP saw, which
     `capture_model-2` says text rows should be gated on and built from. For every kept species it is
     the accepted name.

   The audit suggested a gate of about 1,000 images. The task set 300. The median kept species has
   about 2,970 images, and 27 kept species have fewer than 1,000. Those 27 include sago palm (822),
   three *Brugmansia*, *Podocarpus* (849), *Citrus limon* (629) and the hostas, so a 1,000 gate would
   drop several priority-1 toxic species. Check their recall in the stealing test rather than trust
   the count.

## Columns (`cultivated_perennials.csv`)

| Column | Meaning |
|---|---|
| `scientific_name`, `gbif_key` | Accepted GBIF canonical name and key, from a strict match then the accepted usage, as in `build_species_list`. |
| `common_name` | Curated. |
| `group` | `toxic`, `edible` or `both`. This is a **curation category: why a forager would look at the plant**. It is not a claim that any part is safe to eat. An `edible` species whose NCSU page lists poisonous parts becomes `both`. A `toxic` species that NCSU tags as Edible also becomes `both`. |
| `perennial_evidence` | USDA symbol and durations (growth habits), then the NCSU life cycle, then the curated habit and zone. |
| `tol_images` | Usable ToL images (criterion 3). |
| `ncsu_slug` | NC State Plant Toolbox page, `https://plants.ces.ncsu.edu/plants/<slug>/`. A slug without a hyphen is a **genus** page. Hybrids use NCSU's `genus-x-epithet` form. Blank means no page. |
| `ncsu_toxic_parts` | `<severity>: <parts>` from the page's poison section, with `; contact dermatitis` when it says so. `not listed` means the page has no poison section, which is **never evidence of safety**. Blank means no page. A `[genus page]` prefix means the text applies to the genus. |
| `ncsu_edible_parts` | What the page lists, quoted and never extended: `edible <part>` tags, plant type Edible, and the Edibility text (cut at 110 characters). When the same page also lists poisonous parts, the field starts with `SAME PAGE LISTS POISONOUS PARTS (<severity>)`. For example, the yew page says the aril "is OK to eat" while the seed inside it is deadly. |
| `priority` | 1 = high, 3 = low (see below). |
| `note` | Curated reason, then automatic facts: GBIF name changes, "genus absent from the label set", regional GBIF record count, names not counted, few field photos, label-synonym conflicts. |
| `tol_top_name` | Extra column: the ToL name with the most images (criterion 3). |

The excluded file has the same columns plus `excluded_reason`.

## Priority

The table is sorted by priority, then by toxic, both, edible, then by ToL images.

- **1: add first.** A seriously toxic species that is commonly planted in the region (taxine,
  cycasin, colchicine, aconitine, tropane alkaloids, cardiac glycosides, grayanotoxins). Also any
  `toxic` or `both` species whose NCSU severity is High (the script raises these automatically),
  and the user's two field misses, olive and pothos.
- **2:** a common edible perennial, or a moderately toxic or contact-toxic common ornamental.
- **3:** occasional or rare in the region, or low severity.

Kept: 120 species. Priority 1: 19 (17 toxic, 1 both, 1 edible). Priority 2: 39. Priority 3: 62. By
group: 53 toxic, 25 both, 42 edible.

## What is excluded (121)

- **70 are already in the label set.** Most of the toxic examples the task named are already in:
  oleander, lily of the valley, foxglove, nandina, privets, *Buxus sempervirens*, English and Irish
  ivy, Chinese and Japanese wisteria, *Hydrangea macrophylla* and *H. paniculata*, *Pieris*, cherry
  laurel, horse chestnut, tung oil tree, the common daffodils, red spider lily, snowdrop, summer
  snowflake, Italian arum, rue, coontie and plum yew.
  Some edibles are in too: jujube, raspberry, hazelnut, kousa dogwood, tea, lemon balm, sage,
  flowering quince and pindo palm (*Butia odorata*).
- **25 have fewer than 300 ToL images.** These are the ones to fetch images for later:
  - *Daphne odora* 180, *Taxus × media* 172, *Aconitum carmichaelii* 171;
  - the Southern Indica and Kurume azaleas: *R. × pulchrum* 159, *R. mucronatum* 241,
    *R. yedoense* 129;
  - *Prunus salicina* 294, *Citrus reticulata* 205 (satsuma), kumquat 187, *Musa basjoo* 187.
- 14 houseplants, including dumb cane, Monstera, Philodendron, peace lily, aloe, rubber plant,
  poinsettia and crown of thorns. Several have more than 1,000 images and calcium oxalate or
  irritant latex, so add them if the lead's rule changes.
- 4 tender plants and 6 annual crops.
- 2 names with no GBIF species: *Rhododendron obtusum* (Kurume azalea) and *Citrus unshiu* (satsuma).

## Notes for integration

- **Trait rows are needed before the caution banner can fire.** 37 kept toxic or both species are in
  genera absent from the label set, so D-038's "relative" rule cannot cover them. Examples are
  *Taxus*, *Cycas*, *Brugmansia*, *Colchicum*, *Laburnum*, *Duranta*, *Cascabela*, *Epipremnum*,
  *Caladium*, *Alocasia* and *Podocarpus*.
  - 82 kept species have an NCSU species page and 27 a genus page, all now cached under
    `raw/ncsu/pages/`. The other 11 have none, such as *Ribes*, *Cydonia*, *Morus nigra*,
    *Xanthosoma* and *Ligustrum ovalifolium*.
  - `traits_ncsu.parse` and `traits_build` read only label-set targets, so they need these species
    added to `targets.parquet` first.
- **Two label species list a kept species as a GBIF synonym:** *Vitis vinifera* under
  *V. aestivalis*, and *Prunus lusitanica* under *P. caroliniana*. D-045's
  `manifests/homonym_synonyms.parquet` already marks both `drop`, and today's `tol_candidates.parquet`
  files every image of those two label species under the label species' own name. `tol_select` also
  skips any synonym that is an accepted label name. Once added, these species keep their own
  images.
- **19 kept species have fewer than 300 ToL field photos; the rest are herbarium specimens.** The note
  says so, for example *Buxus microphylla* (41 field photos of 347), *Citrus limon*, the hostas and
  *Narcissus × incomparabilis*. Text rows work from what BioCLIP saw. An image pull for these needs
  other sources, because `tol_select` caps specimens at 30% and the detector never trains on them
  (D-020).
- **Adding text rows enlarges the closed set; it is not open-set rejection** (`capture_model-2`).
  Measure the in-set take on regional val with the audit's protocol before shipping. The audit
  measured 0.11% of in-set photos with 39 rows at bias 0, and its proposed ceiling is about 0.2%.
  The bias needs several logit units to have any effect.
- **NCSU data source:** `traits_ncsu.fetch_pages` (split out of `fetch` so that `targets.parquet` is
  not overwritten) fetched 134 pages at 1 request per second on 2026-11-24. `sitemap_slugs()` also
  returns slugs of non-plant URLs. Their "genus pages" (`ligustrum`, `elaeagnus`) return 404, so
  this module matches `/plants/<slug>/` URLs only.
