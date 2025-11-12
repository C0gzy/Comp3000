# Comp3000

**Disclaimer:**  
This project uses [Git Large File Storage (Git LFS)](https://git-lfs.github.com/) for storing large model files (such as `.h5`, `.keras`, `.pb`).  
To properly clone or contribute to this repository, make sure you have [Git LFS installed and initialized](https://git-lfs.github.com/) on your system.  
Otherwise, model files will not be downloaded correctly.

# Coral Reef Health Monitor using classifcation models and deep learning

## current progress:
| Run Number | Loss    | Accuracy | Precision | Recall |
|-------------|---------|-----------|------------|---------|
| 1 | 0.4407 | 0.8173 | 0.8322 | 0.6464 |
| 2 | 3.5399 | 0.8064 | 0.6983 | 0.8579 |
| 3 | 30.6366 | 0.8000 | 0.8294 | 0.5922 |
| 4 | 3.9265 | 0.5885 | 0.4696 | 0.6920 |
| 5 | 0.4589 | 0.7936 | 0.7044 | 0.8426 |

## Todo
- [ ] Improve Recall and Precision 
- [ ] Add further image Augmentation to improve dataset size
- [ ] Create Web interface
- [ ] Compress model
- [ ] Get more Coral Images
- [ ] break larger images down into 254x254 pixel chunks