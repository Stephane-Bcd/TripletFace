# -*- coding: utf-8 -*-
"""IA Faces.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mbIyR2GNEpaU1L2aQGpBrNAipxogHW_I

# Initialising data

## 1. import data to google collab
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My\ Drive

ls -la

pwd

"""## 2. Unzip data"""

!unzip -F "Copie de dataset.zip" -d ./dataset

"""## 3. Clone my project repository Triplet Face & pull"""

! git clone https://github.com/Stephane-Bcd/TripletFace.git

cd TripletFace/

! git pull

!apt-get install tree

! tree .

"""# Test current model"""

# Commented out IPython magic to ensure Python compatibility.
# Short preparation (as in the previous section but easier way)

from google.colab import drive
drive.mount('/content/drive')
! sleep 1
# %cd /content/drive
! cd '/content/drive/My Drive' ; ls; rm -rf TripletFace/ ; ls; echo "removed TripletFace git repository"
! cd '/content/drive/My Drive' ; git clone https://github.com/Stephane-Bcd/TripletFace.git -q ; ls; cd 'TripletFace'; echo "git cloned again"; git pull -q; echo "git pulled to:"; pwd
# %cd /content/drive/My\ Drive/TripletFace
! apt-get install tree -y -qq; echo ""; echo "current repository tree:"; tree .  

#setting some paths variables
dataset_home = "/content/drive/My Drive/dataset/dataset"
git_home = "/content/drive/My Drive/TripletFace"
resnet_home = "/content/drive/My Drive/TripletFace/resnet18"
multi_home = "/content/drive/My Drive/TripletFace/multi"
model_path = "/content/drive/My Drive/TripletFace/core"

# installing requirements
! sudo pip3 install -r requirements.txt

import sys
if resnet_home not in sys.path:
  sys.path.append(resnet_home)

import resnet18

# Commented out IPython magic to ensure Python compatibility.
# %cd $multi_home

! python ./train2.py