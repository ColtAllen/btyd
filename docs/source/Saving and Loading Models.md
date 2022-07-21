# Model Persistence

Model persistence is helpful for saving and loading BTYD models for reuse as well as logging to a registry server such as `mlflow`. After a model has been fit, its inference data can be saved in either JSON or CSV format. The format will be inferred automatically from the filename.

## Fit model

```python
from btyd import BetaGeoModel
from btyd.datasets import load_cdnow_summary

data = load_cdnow_summary(index_col=[0])
bgm = BetaGeoModel()
bgm.fit(data)
bgm
"""<btyd.BetaGeoModel: Parameters {'alpha': 4.5, 'r': 0.2, 'a': 0.8, 'b': 2.4} estimated with 2357 customers.>"""
```

## Saving and Loading Models


```python
# Save inference data of a fitted model as a JSON:
bgm.save('path/to/file.json')

# Load model inference data from a JSON into a new or existing model:
bgm = bgm.load('path/to/file.json')
bgm_new = BetaGeoModel()
bgm_new._idata = bgm.load('path/to/file.json')

bgm_new
"""<btyd.BetaGeoModel: Parameters {'alpha': 4.5, 'r': 0.2, 'a': 0.8, 'b': 2.4} estimated with 2357 customers.>"""
```
