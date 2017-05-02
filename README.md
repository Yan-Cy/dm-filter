# DM-filter

*DM filter with CNN Sentence and Word2Vec*

*chen yang*

*2017.4.24*


## Code

* Root

```shell
$Root = $shd-magi-01:/mnt/storage01/chenyang
```


* Main code

  ```shell
  $Root/cnn-sentence
    -> train.py
    -> eval.py
  ```

* Interface

  * Path
    
    ```shell
    $Root/cnn-sentence/cnn_sentence.py  
    ```

  * Usage

    ```python
    class CNN_Sentence(object):
        
        __init__(self,
                batch_size = 256, # Batch Size of each iter
                checkpoint_dir = './runs/1492510012/checkpoints', # Pretrained model
                allow_soft_placement = True,
                log_device_placement = False
                )
        
        predict(self,
                x_raw, # input data, 2 dimention list. Use segChinese.filterChinese to convert Chinese string to list: filterChinese(unicode(sentence, 'utf-8'))
                print_info = False # Print infomation while testing
                )

        evaluate(self,
                y_test, # Ground Truth, 1 dimention np array
                all_predictions # predictions, 1 dimention np array
                )
    ```

## Data Source

* Root

```shell
$Root = $shd-magi-01:/mnt/storage01/chenyang
```

* Training data

  ```shell
  Positive data: $Root/danmu/positive.train
    -> 300w dm sentences
    
  Negative data: $Root/danmu/negative.train
    -> 100w dm sentences
  ```


* Test data

  ```shell
  Positive data: $Root/cnn-sentence/data/positive.test
    -> 150w dm sentences
  Negative data: $Root/cnn-sentence/data/nagetive.test
    -> 50w dm sentences
  ```


## Trained Model

```shell
$Root/cnn-sentence/runs/1492510012/checkpoints/
```

## Results

| Real Type \ Protected Type | Positive DM | Negative DM |
| -------------------------- | ----------- | ----------- |
| Positive DM                | 1428260     | 71740       |
| Negative DM                | 158830      | 341170      |



* Recall （命中率） = $\frac{341170}{500000} = 68.234%$
* Precision （准确率）= $\frac{341170}{341170+71740} = 82.63%$
* Accuracy = 88.4175%



