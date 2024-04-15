---
title: Model FineTuning using LoRA
date: 2022-04-15
categories: [NLP, FineTuning, LoRA]
tags: [blog, lora, model fine tuning]
author: saugat
math: true
---

Fine Tuning is the process of taking pre-trained models and passing a new data and updating it's weights with sort of the smaller amount of new data that you're finetuning. It's a powerful tool that we can do with many foundational models, basically in base model that have been pre-trained on a lot of data to acquire good feature representation.

Finetuning Large Language Models (LLM) from scratch is quite resource-intensive, given the large number of parameters these models contain. If not done appropriately, it could lead to the pre-trained model losing some of its base language understanding.

Parameter-efficient fine-tuning (PEFT) methods have emerged as a promising approach to address the challenges of fine-tuning large language models. PEFT techniques aim to adapt models to specific tasks while introducing only a small number of additional parameters. This is particularly useful when dealing with large-scale pre-trained models, as it reduces the computational overhead and memory requirements during fine-tuning. By introducing task-specific parameters in a controlled manner, PEFT methods enable efficient transfer of knowledge from pre-trained models to specific tasks.

![LoRA Architecture](https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/4313422c5f2755897fb8ddfc5b99251358f679647ec0f2d120a3f1ff060defe7?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27lora_diagram.png%3B+filename%3D%22lora_diagram.png%22%3B&response-content-type=image%2Fpng&Expires=1713341813&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMzM0MTgxM319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy80MzEzNDIyYzVmMjc1NTg5N2ZiOGRkZmM1Yjk5MjUxMzU4ZjY3OTY0N2VjMGYyZDEyMGEzZjFmZjA2MGRlZmU3P3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=avJS7Rd5qrwla7VL3MQPksMnfnRKdRi8sAdExDupZH4pzCJrIX0ddn8zkRBmvuDuog-6A-1B9UagSb0FT%7EuxOd35z8xMGxXyTcauQFZrks0PllRQLpi9lzC-JM6aflml%7Eszc0Xhfonp0tiOiN1lBySzJeNd1pyVYWJpmrtgV4edDsFKb085ZgVoj7bxUQzoVugJSw1QBwsnp2wSqc6KMlsAMJJoxaHfIE7zER9kQoYqvCQ1hQ5i%7E7zVdlGUJ8Ox3PvSQjQTSir0xCaGAC5nRvWgYGq5uwf80prKUKbBLy109V7tmTE5HI7NcwkwBx4rQtQECj56mkFUrtJIBK3PyOA__&Key-Pair-Id=KVTP0A1DKRTAX)

LoRA (Low Rank Adaptation) is a PEFT method that decomposes a large matrix into two smaller low-rank matrices in the attention layers. This drastically reduces the number of parameters that need to be fine-tuned. It just allows you to fine tune only small number of extra weights in model, while we freeze most of the parameters of pre-trained networks. Basically, it is a technique used in training language models to make them smaller and faster. It does this by adding a smaller number of new weights to the model and training only these weights, which makes the training process quicker, uses less memory, and results in a smaller model that is easier to save and share.

**Key inspiration for LoRA**
> "Over parameterized models(large models), in fact, reside in low intrinsic dimension" is key inspiration for the paper which means that large models can learn with low dimension inputs.

**<u>Hypothesis</u>**
"Change in weights during model adaptation also has low intrinsic rank" which states that model weight can adapt with very less linearly independent vector. Eg. 1000 dimension can be expressed in terms of 10 linearly independent rows (rank). LoRA allows training some dense layers indirectly by optimizing rank decomposition matrices of dense layers.

For a pre-trained weight matrix $\(W_0 \in \mathbb{R}^{d \times k}\)$, we constrain its update by representing the latter with a low-rank decomposition $\(W_0 + \Delta W = W_0 + BA\), where \(B \in \mathbb{R}^{d \times r}\), \(A \in \mathbb{R}^{r \times k}\)$, and the rank $\(r \leq \min(d, k)\)$.

During training, $\(W_0\)$ is frozen and does not receive gradient updates, while $\(A\) and \(B\)$ contain trainable parameters. Note both $\(W_0\)$ and $\(\Delta W = BA\)$ are multiplied with the same input, and their respective output vectors are summed coordinate-wise. For $\(h = W_0 x\)$, our modified forward pass yields:

$$h = W_0x + \Delta W x = W_0x + BAx$$

**Key Features of LoRA:**
- No additional inference latency(response time between user's input and model's output) unlike conventional adapters
- Higher training throughout
- Performance on par or better than a fully fine-tuned model

**Advantages of LoRA**
- Efficiently train multiple task by switching decomposition matrices
- Lowers the hardware barrier by 3 times since gradient is calculated only for injected matrices
- Simple linear design allows merging the trainable weights with frozen weights results in no additional inference latency
- LoRA can be combined with some other previous adapters methods like previous tuning.

**Conclusion**
- LoRA is a memory and computation efficient strategy which achieves comparable or better performance than fully finetuned model.
- Swappable modules, no additional inference latencies are some key featuers of LoRA

References:
- [LoRA: Low Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/v0.10.0/en/index)
