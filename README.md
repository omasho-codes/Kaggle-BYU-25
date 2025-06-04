# Kaggle-BYU-25

- Made 3D NMS for post processing , LB got fro 0.754 to 0.812 with public YOLO
- Used ONNX export with custom multi-thread image loading , increasing 104s (starter notebook) to 36s for 3 tomograms.  
 

# What I learned 

- Learned to use multi-threading and multi-processing in python and their advantages.
- Got to know about various strengths and flaws in current YOLO11m architecture by analyzing feature maps for different augmented images.
- Developed post-processing steps for merging to 2D inferences by YOLO back to 3D by analyzing scatter plots of 3D coordinates in train     set.

  

