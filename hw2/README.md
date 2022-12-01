# Parameters tuning

All the preprocessing and training parameters can be modified
when invoking the training script.

To see all of them, type

```python train.py -h```

and execute accordingly.

The dataset is the Mini Speech Command (MSC), and it must
be represented with a folder containing these 3 sub-folders:
- ```msc-train```
- ```msc-val```
- ```msc-test```
This folder must be provided to 
the script ```train.py``` through
 the parameter ```--dataset```.

