# Changelog

## 0.99.9post1
### What's Changed
* Changes data documentation by @PedroConrado in <https://github.com/IBM/terratorch/pull/437)>
* Adding badges for the README by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/484>
* Update WxCTutorialDownscaling.ipynb for terratorch 0.99.9 support by @romeokienzler in <https://github.com/IBM/terratorch/pull/491>
* Skipping automatic tests when the modifications are for documentation and other files outside the core.  by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/464>
* add release scripts by @romeokienzler in <https://github.com/IBM/terratorch/pull/489>
* Badge for coverage by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/495>
* Periodical synchronization between documentation from the working branch with `main` by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/482>
* Basic support for visualizing models (as graphs) by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/459>
* Improve/tests by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/498>
* Updating the tests workflows to install granitewxc and prithviwxc from PyPI. by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/494>
* Removing unnecessary module by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/502>
* allow users to run iterate using terratorch command by @leotizzei in <https://github.com/IBM/terratorch/pull/483>
* Update cli_tools.py, make subcommand optional by @romeokienzler in <https://github.com/IBM/terratorch/pull/501>
* graph break fix by @kaoutar55 in <https://github.com/IBM/terratorch/pull/509>

### New Contributors
* @leotizzei made their first contribution in <https://github.com/IBM/terratorch/pull/483>
* @kaoutar55 made their first contribution in <https://github.com/IBM/terratorch/pull/509>

**Full Changelog**: <https://github.com/IBM/terratorch/compare/0.99.9...0.99.9post1>

## 0.99.9
### What's Changed
* unpin versions, update author list pyproject.toml by @romeokienzler in <https://github.com/IBM/terratorch/pull/408>
* Trying to solve issue with replicated input arguments during model (ViT) instantiation/loading by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/415>
* fix-376 by @PedroConrado in <https://github.com/IBM/terratorch/pull/412>
* fix padding by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/397>
* Allowing the segmentation task to output multiple class labels by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/393>
* chore: Updated logic to update the new model versions value `backbone_pretrained` to `false`  by @WanjiruCate in <https://github.com/IBM/terratorch/pull/418>
* add requirements_dist.txt for pypi publishing by @romeokienzler in <https://github.com/IBM/terratorch/pull/406>
* add pin_requirements.py for release by @romeokienzler in <https://github.com/IBM/terratorch/pull/405>
* add requirements_test.txt needed for running tests by @romeokienzler in <https://github.com/IBM/terratorch/pull/407>
* multicrop HF version by @PedroConrado in <https://github.com/IBM/terratorch/pull/419>
* adds integration tests for datamodules by @PedroConrado in <https://github.com/IBM/terratorch/pull/432>
* Fix padding for decoders by @blumenstiel in <https://github.com/IBM/terratorch/pull/439>
* weights_only=True for all the occurences of torch.load by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/391>
* Update CONTRIBUTING.md by @romeokienzler in <https://github.com/IBM/terratorch/pull/446>
* Invoking gc for all these tests by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/445>
* fix model for backwards compatibility by @jaionet in <https://github.com/IBM/terratorch/pull/443>
* Adjusting the weights keys when necessary by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/417>
* Updating README by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/434>
* Pinning albumentations and updating eurosat by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/452>
* Disabling stackability test when requested by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/451>
* timm must be bounded in some way by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/477>
* Fixing links and cleaning not necessary info by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/466>
* pinning albumentations==1.4.6 by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/463>
* Padding as input transform by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/461>
* Removing hardcoded paths.  by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/458>
* Fixing issues with `interpolate_pos_encoding` in prithvi by @daniszw in <https://github.com/IBM/terratorch/pull/471>
* ModelCheckpoint must be defined in the config dict, not during the parsing. by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/454>
* Dealing with encoder outputs with dimension > 3 when using the reshaper neck.  by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/468>
* [WIP] fix 479 by @romeokienzler in <https://github.com/IBM/terratorch/pull/480>
* Removing unnecessary steps and passing extra arguments for tiled inference by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/469>
* 201 downscaling by @romeokienzler in <https://github.com/IBM/terratorch/pull/472>

### New Contributors
* @WanjiruCate made their first contribution in <https://github.com/IBM/terratorch/pull/418>
* @daniszw made their first contribution in <https://github.com/IBM/terratorch/pull/471>

**Full Changelog**: <https://github.com/IBM/terratorch/compare/0.99.8...0.99.9>

## 0.99.8
### What's Changed
* Improve/tests by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/300>
* info not debug by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/306>
* PRs send to `dev` also must be tested by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/308>
* Base task for terratorch  by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/309>
* Fixing by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/311>
* Bump h5py from 3.10.0 to 3.12.1 by @dependabot in <https://github.com/IBM/terratorch/pull/175>
* Bump actions/checkout from 3 to 4 by @dependabot in <https://github.com/IBM/terratorch/pull/34>
* Bump actions/setup-python from 4 to 5 by @dependabot in <https://github.com/IBM/terratorch/pull/35>
* Fix timm pretrained error by @blumenstiel in <https://github.com/IBM/terratorch/pull/318>
* Remove fallback by error with pretrained weights by @blumenstiel in <https://github.com/IBM/terratorch/pull/320>
* Fix base task `on_test_epoch_end` by @fmartiescofet in <https://github.com/IBM/terratorch/pull/319>
* Testing finetuning for more Prithvi-2 backbones by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/322>
* Update contribution_process.md by @romeokienzler in <https://github.com/IBM/terratorch/pull/326>
* [WIP] Add torchgeo models by @paolofraccaro in <https://github.com/IBM/terratorch/pull/233>
* quickfix select_patch_embed_weights.py by @romeokienzler in <https://github.com/IBM/terratorch/pull/346>
* Update README.md by @biancazadrozny in <https://github.com/IBM/terratorch/pull/353>
* increasing timeout for unit tests and fixing issues with tests by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/351>
* Update README.md by @biancazadrozny in <https://github.com/IBM/terratorch/pull/354>
* [WIP] Yaml Schema generation added  by @jaionet in <https://github.com/IBM/terratorch/pull/296>
* Fix: Remove duplicated methods by @fmartiescofet in <https://github.com/IBM/terratorch/pull/356>
* Feat: Implement Terratorch UNet decoder by @fmartiescofet in <https://github.com/IBM/terratorch/pull/357>
* Add deprecation warning for `scale_modules` by @fmartiescofet in <https://github.com/IBM/terratorch/pull/358>
* adds predict to datamodules by @PedroConrado in <https://github.com/IBM/terratorch/pull/337>
* Info not debug by @fmartiescofet in <https://github.com/IBM/terratorch/pull/360>
* Improve/docs by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/324>
* Fix timm config loading for prithvi by @blumenstiel in <https://github.com/IBM/terratorch/pull/372>
* 201modularize wxc by @romeokienzler in <https://github.com/IBM/terratorch/pull/328>
* Feat: Implement option to have multiple learning rates by @fmartiescofet in <https://github.com/IBM/terratorch/pull/329>
* Testing the repository using the installation from pyproject for torchgeo models.  by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/365>
* pinning jsonargparse by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/379>
* Update branch by @blumenstiel in <https://github.com/IBM/terratorch/pull/381>
* Refactor prithvi by @blumenstiel in <https://github.com/IBM/terratorch/pull/377>
* add smoke.py and an integration tests folder getting picked up by the IBM Research GPU cluster atm by @romeokienzler in <https://github.com/IBM/terratorch/pull/347>
* add cli support for wxc gravity wave by @romeokienzler in <https://github.com/IBM/terratorch/pull/380>
* Feat: Implement multiple test dataloaders in all tasks by @fmartiescofet in <https://github.com/IBM/terratorch/pull/330>
* Feat: Add `compute_statistics` subcommand by @fmartiescofet in <https://github.com/IBM/terratorch/pull/336>
* Adding padding at the input when necessary by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/342>
* Freeze/head/decoder by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/378>
* consolidate requirements by @romeokienzler in <https://github.com/IBM/terratorch/pull/389>
* Unpin exact `jsonargparse` version by @fmartiescofet in <https://github.com/IBM/terratorch/pull/394>
* Fix prithvi by @blumenstiel in <https://github.com/IBM/terratorch/pull/398>
* Fix/pyproject by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/396>

### New Contributors
* @paolofraccaro made their first contribution in <https://github.com/IBM/terratorch/pull/233>
* @jaionet made their first contribution in <https://github.com/IBM/terratorch/pull/296>

**Full Changelog**: <https://github.com/IBM/terratorch/compare/0.99.8rc1...0.99.8>

## 0.99.7
### What's Changed
* Fix: Remove filter warnings in clay by @fmartiescofet in <https://github.com/IBM/terratorch/pull/238>
* fix 239 by @romeokienzler in <https://github.com/IBM/terratorch/pull/240>
* Refactor: Use fused attention for Clay by @fmartiescofet in <https://github.com/IBM/terratorch/pull/248>
* [WIP] fix 247 by @romeokienzler in <https://github.com/IBM/terratorch/pull/250>
* fixes typo at sen1floods11 dataset by @PedroConrado in <https://github.com/IBM/terratorch/pull/254>
* Fix/checkpoint by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/255>
* Other ways to define custom modules. by @Joao-L-S-Almeida in <https://github.com/IBM/terratorch/pull/251>

**Full Changelog**: <https://github.com/IBM/terratorch/compare/0.99.6...0.99.7>
