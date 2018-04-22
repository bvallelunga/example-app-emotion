# Emotion
Quickly determine the emotion on a face. Trained on tens of thousands of images
collected by the Montreal Institute for Learning Algorithms.

**Possible Use Cases**
  * Adding emotion classification to a face detection system
  * Targeted messaging to your users based on their emotional responses
  * Measuring the emotional response of an audience during a movie


## Input Scheme
The input should contain a base64 encoded image of a face. The image must be 
at least 64 x 64 pixels. In order to get the best results, make sure your input 
image is a tight face shot.
``` json
{
  "image": "BASE_64_ENCODED_IMAGE"
}
```

## Output Scheme
The output will map each emotion to a percentage. The percentages measure how confident 
the model is that the face shows that specific emotion. 
 
``` json
{
  "angry": 0.16, 
  "disgust": 0.0, 
  "fear": 0.01, 
  "happy": 0.29, 
  "sad": 0.07, 
  "surprise": 0.01, 
  "neutral": 0.45
}
```


## Training
The model was trained by the [B-IT-BOTS robotics team][1] on over 28,000 images. 


[1]: https://mas-group.inf.h-brs.de/?page_id=622
