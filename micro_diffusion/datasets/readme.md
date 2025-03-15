# Downloading Datasets / Precomputing Latents for training:

**Datasets:**
**Training:**  
- [JourneyDB](https://journeydb.github.io/)
- [SA1B](https://ai.meta.com/datasets/segment-anything/)
- [DiffusionDB](https://github.com/poloclub/diffusiondb)  
**Validation:**  
- [COCO](https://cocodataset.org/#home)

## Dataset Pipeline Illustration:
**JourneyDB**:
Download 1% of the dataset
datadir is the directory where the dataset will be downloaded and converted to MDS shards
``` bash
    bash scripts/get_jdb_dataset.sh ./datadir small 8 
```
Download entire dataset
``` bash
    bash scripts/get_jdb_dataset.sh ./datadir all 8 
```
Precompute latents and store in MDS shards
``` bash
    bash scripts/precompute_jdb.sh 1 ./datadir 16
```
The arguments are number of gpus, directory where to find mds shards and store latent mds shards, batch size for a single forward pass through pre-trained openclip and vae.

Recommend looking at the python scripts for the respective datasets for higher degree of modularity.