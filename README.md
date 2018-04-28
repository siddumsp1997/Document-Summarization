# Document-Summarization

Data summarization is the process of creating a concise representation of a document. In
extractive graph-based summarization tasks, we need to create a graph representation of the
document and select some of the sentences of the document for forming the summary.

Given 5 topics and 25 documents each within a topic, we have to synthesize a fluent,
well-organized 250-word summary of the documents per topic using the following graph-based
approaches: Text-Rank & Degree centrality Based. 
Successful performance on the task will benefit from a combination of IR and NLP capabilities.
The weighted graph has to be thresholded for three different thresholds: threshold = 0.1, 0.2
and 0.3.

Regarding Evaluation, we will perform Rouge-N (with N values 1 and 2) and Rouge-L evaluation on the
summaries.
"pythonrouge" package is used here, for ROUGE score calculation. 
Here's the link of that library : https://github.com/tagucci/pythonrouge

You can download the 5 topic samples from here: https://drive.google.com/drive/folders/1ApnqvgTQFyrgit2Rwi4u5J-m1vsD99VU




