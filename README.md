# Dataset Analysis Framework
This framework performs T-SNE on the KITTI dataset.

## Steps done in this framework
Following steps are performed:
1. Turn all KITTI labels to pandas DataFrame
    * As reading from .txt file is slow and tedious, all labels are put into pandas.DataFrame and pickled to Disk
2. Plot Class Distribution across dataset,
    * Since the KITTI dataset is quite unbalanced, the distribution is analyzed.
3. Compute list of images and bounding boxes to perform inference on (class balancement is enforced)
    * The lowest amount of samples per class is 222 (for sitting persons.) To maximize class balancement, 
      222 samples are chosen per class. Since Misc and DontCare classes are not that interesting, those are ignored.
4. Compute feature maps for every image chosen in step 3.
    * The feature maps for each image chosen in step 3) are calculated by using a pre-trained ResNet model from
      Tensorflow Object Detection API.
5. Get features for every single bounding box in step 3.
    * For each bounding box, the center is projected onto the feature map. This pixel is treated as the feature of the bounding box.
6. Compute TSNE on all bounding boxes.
    * TSNE is performed on all bounding boxes.
7. Grab the hard samples.
    * For each class the median is calculated.
    * For each class median, the nearest samples from other classes are treated as hard samples.
 
 ## Run
 There are two ways to run the script
 1. Run the run.py file.
    * Default settings are loaded from settings/run.yaml
 2. Run every script seperately.
    * For each script default scripts are configured and can be found under settings/scripts/.
    
 The second way is especially interesting for tinkering around.
