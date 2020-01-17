# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:52:23 2020

@author: bened
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))