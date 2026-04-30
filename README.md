# Comp3000

> [!WARNING]
> This project uses [Git Large File Storage (Git LFS)](https://git-lfs.github.com/) for storing large model files (such as `.h5`, `.keras`, `.pb`).  
> To properly clone or contribute to this repository, make sure you have [Git LFS installed and initialized](https://git-lfs.github.com/) on your system.  
> Otherwise, model files will not be downloaded correctly.

website link : https://comp3000.vercel.app/

# Coral Reef Health Monitor using classifcation models and deep learning

## How to Use

### AI models

The main two ai models used in this program.
[Harsher Ai model](/Source/ModelHistory/V20251114_192040.h5)
[Secondary Model](/Source/ModelHistory/V20251114_153133.h5)

For these two ai models you will require tensorflow and tf_keras. You can also use [ModelTest.py](/Source/ModelTest.py) to test indivdiual images without the frontend interface

### Website

The website uses NextJS and Bun to download and run packages for the website please install bun to continue.


these commands will install the needed packages and then run the webstie
```
bun install

bun run dev
```

### Backend

To run the backend that the website uses please install python3 and install the requirements

```
python3 install -r requirements.txt

pyhon3 main.py
```

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
- [X] Improve Recall and Precision 
- [X] Add further image Augmentation to improve dataset size
- [X] Create Web interface
- [X] Compress model
- [X] Get more Coral Images
- [X] break larger images down into 254x254 pixel chunks