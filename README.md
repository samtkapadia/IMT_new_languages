# Interactive Machine Teaching for Teaching New Languages
Interactive Machine Teaching platform for teaching new languages

This repo accompanies Sam Kapadia's 2021 thesis 'Interactive Machine Teaching for Teaching New Languages', submitted to University College London as part of the MSc Computational Statistics and Machine Learning.

The code in this repo can be used as an example for creating a web-paced interactive teaching platform that can teach a student words from a new language. We chose to teach English speakers words from Czech, but the framework presented below can be used for any language. The server-side code is written in Python using the Django web development framework. Some JS and HTML is used for the client-side.

## Learn Czech
Please feel free to try our Interactive Teaching Platform (samkapadia.pythonanywhere.com/teacher)! Here, we will teach you 30 Czech words and then test you on another 30. The whole course should take 10-15 minutes to complete. 

## Adapting to a new language
Users are directed to the Django documentation for setting up a new Django project and hosting a development server locally (https://www.djangoproject.com/). Ensure dependencies listed in IMT_django/requirements.txt are installed. The following steps can be followed to adapt the framework to a new target language:

### Specify target vocabulary
1. Collect N words you would like to teach from target language. 
2. Store N words in /Datasets/[]\_words_[N].npy
3. Create dictionary mapping N words to translation in language of target user. Save this as /Datasets/[]\_[]\_dict.pickle

### Collect/generate teaching examples
4. Acquire N reference (ground-truth) pronunciations (.wav) for your words and store in IMT_django/static/teacher/audio
5. Update /Datasets/audio_paths_[].npy
7. Acquire/generate image of each word and store in IMT_django/static/teacher/images
8. Update /Datasets/image_paths_[].npy

### Create new weight matrix
9. Generate embeddings based on image of each word, and the corresponding weight matrix, using generate_weight_matrix notebook
10. Save new weight matrix as weight_matrix_CONVAUTO.npy and store in /Datasets

11. Ensure filepath referencing is updated in IMT_django/teacher/views.py

## Credits
This project built upon work by Johns et al. (http://visual.cs.ucl.ac.uk/pubs/interactiveMachineTeaching/) [1]. Particular thanks also to Muaz Khan whose repo. (https://github.com/muaz-khan/RecordRTC) helped with the platform's recording functionality.

## References
<a id="1">[1]</a> 
Johns, Edward and Mac Aodha, Oisin and Brostow, Gabriel J. (2015). 
Becoming the Expert - Interactive Multi-Class Machine Teaching 
CVPR
