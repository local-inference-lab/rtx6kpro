# Archived serving guides

Use the [model index](../../README.md#start-here) for supported launch instructions.
These snapshots preserve complete commands, image versions and performance
tables for reproducing a deployment or comparing releases. Commands in an
archive do not follow the floating beta image tag.

## Karmic Kraken beta, September 19, 2026

Image: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260919-cfc67a15ebc3daf7`.

- [Shared Docker options](karmic-kraken-beta-20260919-cfc67a15ebc3daf7/docs/unified-vllm-docker.md)
- [GLM-5.3-Flash](karmic-kraken-beta-20260919-cfc67a15ebc3daf7/models/glm-5.3-flash.md)
- [GLM-5.3-Flash Spark TP2](karmic-kraken-beta-20260919-cfc67a15ebc3daf7/models/glm-5.3-flash-spark-tp2.md)
- [Qwen3.8-Flash-Next](karmic-kraken-beta-20260919-cfc67a15ebc3daf7/models/qwen38-flash-next.md)
- [DeepSeek V4 Flash](karmic-kraken-beta-20260919-cfc67a15ebc3daf7/models/deepseek-v4-flash.md)
- [DeepSeek V4 Flash Vision](karmic-kraken-beta-20260919-cfc67a15ebc3daf7/models/deepseek-v4-flash-vision.md)
- [DeepSeek V4.1 Flash](karmic-kraken-beta-20260919-cfc67a15ebc3daf7/models/deepseek-v4.1-flash.md)
- [Performance report](karmic-kraken-beta-20260919-cfc67a15ebc3daf7/benchmarks/karmic-kraken-serving.md)

Performance tables retain their original hardware, clock and image labels;
archiving a page does not reassign measurements to this image. The
[manifest](karmic-kraken-beta-20260919-cfc67a15ebc3daf7/manifest.json) records the
source revision, original page identities and archive transformations.

For recipes predating these snapshots, use the
[model-directory history](https://github.com/local-inference-lab/rtx6kpro/commits/master/models)
or the historical release links in each archived guide.
