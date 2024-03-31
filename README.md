## Question answering robustness benchmark

The repo includes code to assess the robustness of language models trained for biomedical question answering (QA).

### Installation

Clone the code repo. Go to the qstab folder, install using the following from the command line

<pre><code class="console"> pip install . </code></pre>

### Run code

The code are under the scripts folder. Under the qstab folder, type the following from the command line

Run Flan-T5-large model on MedQA-USMLE drug-mention questions
<pre><code class="console"> python ./scripts/run_hf_entity.py -grp="drugs" -nq=6000 </code></pre>

Run MedAlpaca-7B model on MedQA-USMLE disease-mention questions
<pre><code class="console"> python ./scripts/run_medalpaca_entity.py -grp="diseases" -nq=6000 </code></pre>

Here 6000 is just a large enough number such that all instances are run.