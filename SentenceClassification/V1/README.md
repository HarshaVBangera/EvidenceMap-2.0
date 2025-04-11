### V1 SentenceClassification configuration options

Below is a list of the configuration options for this module:

- `model`: Any Huggingface model suited for NLP tasks.  
  - This value is required.
- `learning_rate`: Only used for initial training.  
  - Default: 2e-5
- `num_train_epochs`: Maximum number of maximum epochs for initial training.  
  - Default: 60
- `per_device_train_batch_size`: Number of samples to load at a time on each computing device during training. 
  - Default: 4
- `per_device_eval_batch_size`: Number of samples to load at a time on each computing device during evaluation and predictions. 
  - Default: 1
- `weight_decay`: Only used for initial training.  Sets the weight decay.
  - Default: 0.01
- `warmup_steps`: Only used for initial training.  Number of steps to ramp up learning rate.
  - Default: 10