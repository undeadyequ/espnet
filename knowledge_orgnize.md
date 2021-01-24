---
layout: post
title:  "knowledge_organize"
date:   2019-7-17 12:52:48 +0900
categories: jekyll update
---

## args usage in big project



## The overall process of training Speech data
    - Prepare all types of feed data in files with (name, data) format. (name, raw_file) is required
    - Create feed data iterator, read specific data, set by args, in batch form
        - Read each data element in key-value format
        - Organize data element in batch with specified method
        - Build batch iterator

    - Build middle-level data processing model which compute middle-value used in final model
        - Gather several middle-level data processor
        - 

    - Create trainer which specify the training process
        - Build core-model
        - build other model, optimizer etc, used in core-model


## Sampler
    - Fixed Length
    - Vairable Length
        - descend_descend
 
