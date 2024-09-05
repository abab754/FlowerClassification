import os
import shutil
import scipy.io

# Load the labels and setid data
labels = scipy.io.loadmat('imagelabels.mat')['labels'][0]
setid = scipy.io.loadmat('setid.mat')
train_ids = setid['trnid'][0]
val_ids = setid['valid'][0]
test_ids = setid['tstid'][0]

# Create directories for organized dataset
for subset in ['train', 'val', 'test']:
    os.makedirs(f'flowers/{subset}', exist_ok=True)
    for i in range(1, 103):  # 102 classes
        os.makedirs(f'flowers/{subset}/class_{i}', exist_ok=True)

# Function to move images to the correct directory
def move_images(ids, subset):
    for img_id in ids:
        label = labels[img_id - 1]  # MATLAB indexing starts at 1
        src = f'jpg/image_{img_id:05d}.jpg'  # Update 'jpg' if your folder name is different
        dst = f'flowers/{subset}/class_{label}/{src.split("/")[-1]}'
        shutil.copyfile(src, dst)

# Move images according to dataset split
move_images(train_ids, 'train')
move_images(val_ids, 'val')
move_images(test_ids, 'test')

print("Images organized into training, validation, and test sets.")
