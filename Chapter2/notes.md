# An Example But Not A Template

## 5-flower dataset great to learn but shouldn't be used as a template when creating training dataset

### Why that's the case :

1. Quantity
   - To train ML models from scratch, you'll need MILLIONS of images
2. Data format
   - Storing images as individual JPEGs --> Super inefficient
   - Since training time is spent waiting for the data to be read
   - Recommend using **<u>Tensorflow format</u>**
     - Check out ```tf_format_example.py``` for an example 
3. Content
   - This dataset consists of **found** data
     - Or images that weren't explicitly collected for classification taks
   - Ideally should collect more data purposefully
4. Labeling
   - The 5 flower data set was manually labeled
   - Not pratical for larger datasets

<br>

## Standardize Image Collection

### In practice you can make a machine perception problem easier by **standardizing how the images are collected**

- Example : You can specify your images have to be collected in controlled conditions
  - Flat lighting
  - Consistent zoom
- Important thing to keep in mind
  - Training dataset **<u>has to reflect the conditions your model will be required to make predictions </u>**

<br>

# Reading Image Data
## Four Steps to reading an image --> training ML model
### 1) Read the image data from persistent storage to memory as a sequence of bytes
   - ```img``` is a Tensor that also contains an array of bytes
   - Parse the bytes --> convert them to pixel data
   - Aka : ```Decoding```
     - Decoding involves decoding the pixel values from look up tables
```python
img = tf.io.read(file(filename))
```
### 2) Specify number of color channels 
   - Pixel values will be of ```uint8``` 
   - And range from 0 - 255
```python
# Specifying we only want three out of the four possible color channels 
# RGB and opacity being the fourth
img = img = tf.image.decode_jpeg(img, channels=3)
```
### 3) Convert the pixel values to floats then scale the values to fall between the ranges of [0 - 1]
   - Since ML optimizers are tuned to work well with small numbers 
```python
img = tf.image.convert_image_dtype(img, tf.float32)
```
## 4) Resize the image to desired size
- ML models are built to work with inputs of known sizes
- Since real-world images come in arbitrary sizes 
  - Might need to shrink/crop/expand them to fit desired size
  - Example code on how to resize can be found below
```python
tf.image.resize(img,[256, 128])
```

# ^ These steps aren't set in stone ^
- Either cropping the data or padding it with zeros is okay

<br>

---

<br>

# Visualizing Image Data

## Always visualize the first few images to make sure data is being read in correctly
- Common mistake : read ddata in a way that the images are either rotated or mirrored
- Visualizing the images also super useful to get a sense of how challenging a machine perception problem is
  - Use Matplotlib's ```imshow()``` function to visualize an image
  - Must convert the image that's initally a ```Tensorflow tensor``` --> ```numpy array``` 
    - Done via using : ```numpy() function```

```python
def show_image(filename):
   img = read_and_decode(filename, [IMG_HEIGHT, IMG_WIDTH])
   plt.imshow(img.numpy())
```