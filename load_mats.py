import scipy.io

# Load the image labels
labels_data = scipy.io.loadmat('imagelabels.mat')
labels = labels_data['labels'][0]  # Extract the labels array

# Load the dataset split information
setid_data = scipy.io.loadmat('setid.mat')
train_ids = setid_data['trnid'][0]  # Training set IDs
val_ids = setid_data['valid'][0]    # Validation set IDs
test_ids = setid_data['tstid'][0]   # Test set IDs

# Print some information to confirm loading
print(f"Number of images: {len(labels)}")
print(f"Number of training samples: {len(train_ids)}")
print(f"Number of validation samples: {len(val_ids)}")
print(f"Number of test samples: {len(test_ids)}")

# Optional: Print the first few labels to check
print("First few labels:", labels[:10])
print("First few training IDs:", train_ids[:10])
