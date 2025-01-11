[中文文档](README_CN.md)

Solved [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux) model pollution problem and supported using with TeaCache.

Must uninstall or disable `ComfyUI-PuLID-Flux` and other PuLID-Flux nodes before install this plugin. Due to certain reasons, I used the same node name `ApplyPulidFlux`.


## Preview (Image with WorkFlow)
![save api extended](examples/PuLID_with_teacache.png)

## Install

- Manual
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_PuLID_Flux_ll.git
    cd ComfyUI_PuLID_Flux_ll
    pip install -r requirements.txt
    # restart ComfyUI
```

## Model
Please see [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)


## Nodes
- PulidFluxModelLoader
  - See [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- PulidFluxInsightFaceLoader
  - See [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- PulidFluxEvaClipLoader
  - See [ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)
- ApplyPulidFlux
  - Solved the model pollution problem of the original plugin ComfyUI-PuLID-Flux
  - `attn_mask` may not work correctly (I have no idea how to apply it, I have tried multiple methods and the results have been satisfactory)
- FixPulidFluxPatch
  - If you want use with [TeaCache](https://github.com/ali-vilab/TeaCache), must link it after node `ApplyPulidFlux`, and link node [`FluxForwardOverrider` and `ApplyTeaCachePatch`](https://github.com/lldacing/ComfyUI_Patches_ll) after it.

## Thanks

[ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID)

[ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)

