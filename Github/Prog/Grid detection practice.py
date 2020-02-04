#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


flat_chess = cv2.imread('C:\\Users\\Raja1\\Pictures\\AI & Data science\\Object detection\\flat-chess-boards-903 (2).jpg')


# In[3]:


plt.imshow(flat_chess,cmap='gray')


# In[4]:


found, corners = cv2.findChessboardCorners(flat_chess,(7,7))


# In[5]:


if found:
    print('OpenCV was able to find the corners')
else:
    print("OpenCV did not find corners. Double check your patternSize.")


# In[6]:


corners.shape


# In[7]:


flat_chess_copy = flat_chess.copy()
cv2.drawChessboardCorners(flat_chess_copy, (7, 7), corners, found)


# In[8]:


plt.imshow(flat_chess_copy)


# In[33]:


dots = cv2.imread('C:\\Users\\Raja1\\Pictures\\AI & Data science\\Object detection\\circles-pattern-png-2.png')


# In[34]:


plt.imshow(dots)


# In[49]:


found, corners = cv2.findCirclesGrid(dots, (5,14), cv2.CALIB_CB_ASYMMETRIC_GRID)


# In[50]:


found


# In[ ]:





# In[18]:


dbg_image_circles = dots.copy()
cv2.drawChessboardCorners(dbg_image_circles, (10, 10), corners, found)


# In[19]:


plt.imshow(dbg_image_circles)


# In[ ]:




