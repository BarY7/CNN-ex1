import helpers.plot as plot
import hw1.datasets as hw1datasets

# Create the dataset
num_samples = 1000
num_classes = 10
image_size = (3, 32, 32)
ds = hw1datasets.RandomImageDataset(num_samples, num_classes, *image_size)

# You can load individual items from the dataset by indexing
img0, cls0 = ds[0]

# Plot first N images from the dataset with a helper function
fig, axes = plot.dataset_first_n(ds, 9, show_classes=True, nrows=3)