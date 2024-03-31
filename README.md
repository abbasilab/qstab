## Question answering robustness benchmark

The repo includes code to assess the robustness of language models trained for biomedical question answering (QA).

### Installation

Clone the code repo. Go to the qstab folder, install using the following from the command line

<pre><code class="console"> pip install . </code></pre>

### Run baseline (no attack)

The code are under the scripts folder. Under the qstab folder, type the following from the command line

Run Flan-T5-large model on MedQA-USMLE drug-mention questions
<pre><code class="console"> python ./scripts/run_hf_entity.py -grp="drugs" -nq=6000 </code></pre>

Run MedAlpaca-7B model on MedQA-USMLE disease-mention questions
<pre><code class="console"> python ./scripts/run_medalpaca_entity.py -grp="diseases" -nq=6000 </code></pre>

Here 6000 is just a large enough number such that all instances are run.

### Run entity substitution attack

**Random sampling**

Using the Flan-T5-large model on MedQA-USMLE drug-mention questions
<pre><code class="console"> python ./scripts/run_hf_entity.py -ptb=True -ptb_samp="random" -grp="drugs" -nq=6000 </code></pre>

**Powerscale distance-weighted sampling (PDWS)**

Using the Flan-T5-large model on MedQA-USMLE drug-mention questions
<pre><code class="console"> python ./scripts/run_hf_entity.py -ptb=True -ptb_samp="distance" -grp="drugs" -ndist=20 -nq=6000 </code></pre>

**Hyperparameter tuning for PDWS**

Using the Flan-T5-large model on MedQA-USMLE drug-mention questions
<pre><code class="console"> bash ./scripts/flant5_large_tune.sh </code></pre>