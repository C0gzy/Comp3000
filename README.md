# Comp3000

> [!WARNING]
> This project uses [Git Large File Storage (Git LFS)](https://git-lfs.github.com/) for storing large model files (such as `.h5`, `.keras`, `.pb`).  
> To properly clone or contribute to this repository, make sure you have [Git LFS installed and initialized](https://git-lfs.github.com/) on your system.  
> Otherwise, model files will not be downloaded correctly.

# Coral Reef Health Monitor using classifcation models and deep learning

## current progress:
| Run Number | Loss    | Accuracy | Precision | Recall |
|-------------|---------|-----------|------------|---------|
| 1 | 0.4407 | 0.8173 | 0.8322 | 0.6464 |
| 2 | 3.5399 | 0.8064 | 0.6983 | 0.8579 |
| 3 | 30.6366 | 0.8000 | 0.8294 | 0.5922 |
| 4 | 3.9265 | 0.5885 | 0.4696 | 0.6920 |
| 5 | 0.4589 | 0.7936 | 0.7044 | 0.8426 |
| 6 | 0.6475 | 0.7316 | 0.6154 | 0.7716 |
| 7 | 0.7292 | 0.7827 | 0.6859 | 0.7834 |
| 8 | 0.4818 | 0.7923 | 0.8694 | 0.5296 |
| 9 | 0.4218 | 0.8358 | 0.8494 | 0.6870 |


## Todo
- [ ] Improve Recall and Precision 
- [ ] Add further image Augmentation to improve dataset size
- [ ] Create Web interface
- [ ] Compress model
- [ ] Get more Coral Images
- [ ] break larger images down into 254x254 pixel chunks