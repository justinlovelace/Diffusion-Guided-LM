# Diffusion Guided Language Modeling

This is the official code release for the ACL Findings 2024 paper:

**Diffusion Guided Language Modeling**.

by Justin Lovelace, Varsha Kishore, Yiwei Chen, and Kilian Q. Weinberger

**Note: Code will be available soon. Stay tuned for updates!**

### Abstract
Current language models demonstrate remarkable proficiency in text generation. However, for many applications it is desirable to control attributes, such as sentiment, or toxicity, of the generated language---ideally tailored towards each specific use case and target audience. For auto-regressive language models, existing guidance methods are prone to decoding errors that cascade during generation and degrade performance.
In contrast, text diffusion models can easily be guided with, for example, a simple linear sentiment classifier---however they do suffer from significantly higher perplexity than auto-regressive alternatives. In this paper we use a guided diffusion model to produce a latent proposal that steers an auto-regressive language model to generate text with desired properties. Our model inherits the unmatched fluency of the auto-regressive approach and the plug-and-play flexibility of diffusion. We show that it outperforms previous plug-and-play guidance methods across a wide range of benchmark data sets. Further, controlling a new attribute in our framework is reduced to training a single logistic regression classifier.

### Citation
```bibtex
@inproceedings{lovelace2024dglm,
  title={Diffusion Guided Language Modeling},
  author={Lovelace, Justin and Kishore, Varsha and Chen, Yiwei and Weinberger, Kilian Q},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
  year={2024},
  publisher={Association for Computational Linguistics},
}
