## Question answering robustness benchmark

The repo includes code to assess the robustness of language models trained for biomedical question answering (QA).

### Installation

Clone the code repo. Go to the qstab folder, install using the following from the command line

<pre><code class="console"> pip install . </code></pre>

### Run baseline (no attack)

The code are under the scripts folder. Under the `qstab` folder, type the following from the command line

Run [Flan-T5-large](https://huggingface.co/google/flan-t5-large) model on [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf-MPNet-IR) drug-mention questions
<pre><code class="console"> python ./scripts/run_hf_entity.py -grp="drugs" -nq=6000 </code></pre>

Run [MedAlpaca-7B](https://huggingface.co/medalpaca/medalpaca-7b) model on MedQA-USMLE disease-mention questions
<pre><code class="console"> python ./scripts/run_medalpaca_entity.py -grp="diseases" -nq=6000 </code></pre>

Here 6000 is just a large enough number such that all instances are run.

### Run entity substitution attack

The substitution attack targets entities in the distractors of the question. Using a sampling-based approach, the attacker selects a replacement entity to substitute the original one that exist in the instance of the dataset.

* **Random sampling**

Using the Flan-T5-large model on MedQA-USMLE drug-mention questions
<pre><code class="console"> python ./scripts/run_hf_entity.py -ptb=True -ptb_samp="random" -grp="drugs" -nq=6000 </code></pre>

* **Powerscale distance-weighted sampling (PDWS)**

Using the Flan-T5-large model on MedQA-USMLE drug-mention questions
<pre><code class="console"> python ./scripts/run_hf_entity.py -ptb=True -ptb_samp="distance" -grp="drugs" -ndist=20 -nq=6000 </code></pre>

* **Hyperparameter tuning for PDWS**

Using the Flan-T5-large model on MedQA-USMLE drug-mention questions
<pre><code class="console"> bash ./scripts/flant5_large_tune.sh </code></pre>