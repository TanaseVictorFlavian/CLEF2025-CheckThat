# Task 3: Fact-checking numerical claims

This task focuses on verifying claims with numerical quantities and temporal expressions. Numerical claims are defined as those requiring validation of explicit or implicit quantitative or temporal details. Participants must classify each claim as True, False, or Conflicting based on a short list of evidence. The fact-verification task is available in languages Spanish, Arabic and English

Each claim would be provided with top-k BM25 evidence and participants can choose to carefully select from this evidence set or employ re–ranking approaches to improve fact verification performance. The evidence corpus is collected by pooling using multiple advanced claim decomposition approaches and the top-100 Bm25 evidences are retrieved from this pool to provide diverse perspectives required for claim verification.

The task is offered in Arabic, English, and Spanish. 

Information regarding the annotation guidelines can be found in the following papers:

> Venktesh V, Abhijit Anand, Avishek Anand, and Vinay Setty. 2024, _[QuanTemp: A real-world open-domain benchmark for fact-checking numerical claims] (https://doi.org/10.1145/3626772.3657874)_. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 650–660. 



__Table of contents:__

<!-- - [Evaluation Results](#evaluation-results) -->
- [List of Versions](#list-of-versions)
- [Contents of the Task 3 Directory](#contents-of-the-repository)
- [Datasets statistics](#datasets-statistics)
- [Input Data Format](#input-data-format)
- [Output Data Format](#output-data-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Scorers](#scorers)
- [Baselines](#baselines)
- [Credits](#credits)

<!-- ## Evaluation Results

TBA -->

## List of Versions
- [01/01/2025] training data released.

<!-- * **subtask-2A-english**
  - [03/05/2023] (unlabeled) test data are released.
  - [21/02/2023] previously released training data contained also validation data, they are now split in two separate files.
  - [30/01/2023] training data are released.
* **subtask-2A-arabic**
  - [03/05/2023] (unlabeled) test data are released.
  - [10/03/2023] training and validation data are released.
* **subtask-2A-dutch**
  - [03/05/2023] (unlabeled) test data are released.
  - [16/03/2023] training and validation data are released.
* **subtask-2A-german**
  - [03/05/2023] (unlabeled) test data are released.
  - [02/03/2023] training and validation data are released.
* **subtask-2A-italian**
  - [03/05/2023] (unlabeled) test data are released.
  - [21/02/2023] validation data are released.
  - [30/01/2023] training data are released.
* **subtask-2A-turkish**
  - [03/05/2023] (unlabeled) test data are released.
  - [02/03/2023] training and validation data are released.
* **subtask-2A-multilingual**
  - [03/05/2023] (unlabeled) test data are released.
  - [23/03/2023] training and validation data are released. -->

## Contents of the Task 3 Directory

- Main folder: [data](task3/data)
  - Contains a subfolder for each language which contain the data as json format with claims and related meta-data for train and validation sets.
- Main folder: [baseline](clef2025-checkthat-lab/task3/baselines)<br/>
  - Contains a notebook train_using_bm25.py to train a nli model using BM25 retrieved evidence from corpus.
  - Contains a file, inference.py, used to provide predictions using a trained model.
  - Contains a .tsv file for each language (e.g - clef2025-checkthat-lab/task3/output/English/output_example.csv), consisting of the examples of  predictions of the baseline over the dev sets.
  - Contains a README.md file.
- Main folder: [scorer](clef2025-checkthat-lab/task3/baselines/English)<br/>
  - Contains a single file, clef2025-checkthat-lab/task3/baselines/English/eval_veracity_prediction.py, that checks the format of a submission and evaluate the various metrics.
  The corpus / evidence collection can be found at [Evidence collection](https://drive.google.com/drive/folders/1GYzSK0oU2MiaKbyBO3hE8kO4gdmxDjCv?usp=drive_link)
  Alternatively instead of using whole corpus if you want to use BM25 retrieved evidence for original claims, please refer clef2025-checkthat-lab/task3/data/English/bm25_evidence. The bm25 evidence for training claims can be found at [train_bm25](https://drive.google.com/drive/folders/1ZYhjUB_POyBbrAWBxWh5FljyG81Gymuo?usp=drive_link)

The bm25 evidence for decomposed questions for train and validation sets of English can be found at [Decomposed claims BM25 evidence](https://drive.google.com/drive/folders/1j8pr4xIe9B3tdxwy2jHuuc545YiiSPBK?usp=sharing)

BM25 evidence for all train and validation claims together for Spanish. Map the claims to train and validation claim files to get corresponding evidence [Spanish Bm25 evidence] (https://drive.google.com/drive/folders/1MSaijGQ4jafMceFdi4hVacxzMxhNkWSl?usp=sharing)

- [README.md](./README.md) <br/>

## Datasets statistics

-- * **Task 3 - English**
  - train: 9935 claims,
  - dev: 3084 claims,

* **Task 3 - Spanish**
  - train: 1506 claims
  - val: 377 claims

* **Task 3 - Arabic**
  - train: TBA



## Input Data Format

The data will be provided as a json file with multiple fields such as:
> claim <TAB> taxonomy <TAB> doc <TAB> label etc...

Where: <br>
* claim: a given claim<br/>
* taxonomy: claim's taxonomy classified into *temporal*, *statistical*, *comparison* and *interval* <br/>
* doc: Oracle document which is the justification given for the claim verdict by manual fact-checkers
* label: *True*, *Conflicting* and *False*
* Other meta-data such as claim source url, evidence url are given 


The evidence collection provided consists of document ids paired with the corresponding text

The top-100 Bm25 evidence gives the text of top-100 documents from above corpus for each claim or at sub-question level in case of decomposition based approaches.

<!-- **Note:** For English, the training and development (validation) sets will also include a fourth column, "solved_conflict", whose boolean value reflects whether the annotators had a strong disagreement. -->

**Examples:**

An example from dataset
```
{
        "country_of_origin": "usa",
        "label": "Conflicting",
        "url": "https://www.politifact.com/factchecks/2010/aug/08/donald-carcieri/carcieri-says-tax-repeal-spawned-new-business/",
        "lang": "en",
        "claim": "Repealing the sales tax on boats in Rhode Island has spawned 2,000 companies, 7,000 jobs and close to $2 billion a year in sales activity.",
        "doc": "The furor over U.S. Sen. John Kerry\u2019s yacht being docked on the tax-free shores of Rhode Island -- not in Massachusetts where he lives -- has subsided now that the senator\u2019s team has promised he will pay the $500,000 in owed taxes.But Rhode Island Governor Donald Carcieri did his part to make sure the underlying issue of Rhode Island\u2019s tax exemption for boats is not forgotten.In one of the governor\u2019s frequent appearances on Fox News\u2019 \"The Neil Cavuto Show,\" Carcieri touted the positive effect of the sales tax repeal on boat sales, the marine industry and the state\u2019s economy as a whole.\"We abolished the tax on boats back in 1993 and in the intervening period we have over 2,000 little companies that have been spawned, employing almost 7,000 people, generating close to $2 billion a year in sales activity,\" an enthusiastic Carcieri told Cavuto, a conservative pundit from Barrington, R.I., who seems tickled by the Ocean State connection.It\u2019s no wonder Carcieri was excited. With the nation's fourth-highest unemployment rate, Rhode Island rarely gets to brag about any positive job news.And it's clear the sales tax repeal -- enacted under then-Governor Bruce Sundlun -- helped rescue a sector that was taking on water by the early 1990s, having struggled after a boom in previous decades. Following the law's passage, boat registrations in Rhode Island increased every year for two decades before declining slightly during the recent recession.But those are big numbers for a fairly narrow industry and they warrant a careful check.The Carcieri administration told us that prior to the interview, they put out a request to several state agencies for data demonstrating the impact of the tax repeal.The state Economic Development Corporation directed them to two studies by the Rhode Island Marine Trades Association: one from 2008 that focused on the skills gap in the industry and another from 2007 that looked at the effect of the repeal.We got a hold of those reports and it appears the governor is close to correct. The studies collectively find that the Rhode Island marine industry includes 6,600 jobs across 2,300 companies, which generates $1.6 billion a year in sales activity. The numbers don't exactly line up, but they're fairly close to those that Carcieri cited.The sales tax study does not make clear how it arrived at the final tallies and the Marine Trades Association refused to explain its methodology, saying it was confidential.The second study tries harder to explain its approach. It relied on a combination of 136 survey results, as well as U.S. Economic Census numbers and economic modeling, to determine the totals. The researchers started by using the ratio of payroll to sales obtained from the 2002 U.S. Economic Census to determine the number of jobs, companies and total sales in what they call \"core marine trades industries.\" It then appears that they added those numbers to a secondary group of similarly calculated figures that included companies considered to be \"indirect marine trades businesses.\" That math makes sense to us. But then the study offers this caveat: \"In 2006, Rhode Island had approximately 1,700 businesses at least some of whose sales could be attributed to the state's 'direct' marine trades industries. Exactly how much was attributable to marine trades is impossible to determine from employment data reported by the Rhode Island Department of Labor and Training.\"Put another way, there are a lot of companies in this state that may devote some, but not all, of their business to the boating industry. Paul Harden, manager of business and workforce development for the state Economic Development Corporation, offered a hypothetical example of a canvas textile company. It could manufacture boat covers, as well as awnings for shops. One product is clearly connected to the industry, the other is not. That unknown makes it difficult to track the number of jobs created and the business generated within the company specifically for marine work.Based on experience, Harden, who helped oversee the Marine Trades study, said researchers assigned estimated percentages that each subcategory of business devotes to marine sales. He acknowledged that it is an imperfect science.That's where Carcieri's claim gets a little shaky. He suggests the tallies are official, despite the study's stipulation that it is \"impossible\" to pinpoint an exact number.But the bigger issue is that Carcieri says that all of these jobs, businesses and sales activity were \"spawned\" by the 1993 tax repeal on boats, which the study never concludes. It seems clear that the repeal had a positive effect on the industry, but some of the businesses and jobs predated the repeal.\"I don't think you can attribute all the job growth and all the sales activity to the repeal of the sales tax,\" said Harden, of the EDC.Given the study's limits, we tried to verify the numbers independently.We started with the state Division of Taxation. But because boats are not taxed, their sales are not reported to the division, said Tax Administrator David Sullivan. And the department doesn't collect data on marine-related sales.Next we spoke with the state Department of Labor and Training, which collects data on jobs and companies in different industries. Similar to the Marine Trades study, its data shows that in 2009, approximately 4,560 people were employed in \"primary marine industries.\" A second class of \"related industries\" adds about another 1,400 jobs for a total of close to 6,000. But here again, a DLT spokeswoman said it's impossible to know just how many of those related jobs are definitively connected to the marine sector. Realistically, the number probably falls somewhere between 4,560 and 6,000, both of which are lower than the number Carcieri cites.And again, that's total jobs, not those \"spawned\" solely by the sales tax repeal.Carcieri got it right by doing his research before spouting statistics on national television, and the studies he cites offer some measurable data. However, it is impossible to know how precise those numbers are and the governor should have offered that caveat. More troublesome, he should never have said the tax repeal \"spawned\" what amounts to an entire industry. We find this Half True.",
        "taxonomy_label": "statistical",
        "label_original": "half-true"
    }
```

The pipeline consists of claim decomposition, evidence retrieval and stance detection steps.
# Setup
pip install -r requirements.txt

## Output Data Format

The output must be a TSV format with two columns: **claim** and **verdict**.

## Evaluation Metrics

This task is evaluated as a classification task. We will use the F1-macro averaged measure and and classwise F1 scores for evaluating the fact-verification performance and for the ranking of teams.

We will also measure Precision, Recall, and F1 for each class.
<!--
There is a limit of 5 runs (total and not per day), and only one person from a team is allowed to submit runs.

Submission Link: Coming Soon

Evaluation File task3/evaluation/CLEF_-_CheckThat__Task3ab_-_Evaluation.txt -->

## Scorers

To evaluate the output of your model which should be in the output format required, please run the script below:


```
python3 task3/baselines/English/eval_veracity_prediction.py --test_path clef2025-checkthat-lab/task3/data/English/val_claims_quantemp.json --output_path output/output_example.csv
```

where output_example.tsv is the output of your model on the dev set / test set and val_claims_quantemp.json is the dev set. You can replye this with test set when released.



## Baselines

The notebook to train the baseline is provided in the related directory clef2025-checkthat-lab/task3/baselines/English.

The inference script is also provided in relevant directory:
> python3 code/nli_inference/veracity_prediction.py --test_path data/raw_data/test_claims_quantemp.json --bm25_evidence_path task3/data/English/bm25_top_100_claimdecomp.json --base_model roberta-large-mnli --model_path <model_path>/<model_name>/model_weights.zip --questions_path data/decomposed_questions/test/test_claimdecomp.csv --output_path finqa_roberta_claimdecomp

where train_data.tsv is the file to be used for training and dev_data.tsv is the file on which doing the prediction.

The baseline is a RoBERTA based NLI model though participants are free to use any model including LLMs. You can also refer to the paper for the idea.

<!-- ### Task 3: Multi-Class Fake News Detection of News Articles

For this task, we have created a baseline system. The baseline system can be found at https://zenodo.org/record/6362498
 -->

## Submission
TBA
<!-- 
Submission is done through the Codalab platform at: https://codalab.lisn.upsaclay.fr/competitions/18809

-   Make sure that you create one account for each team, and submit it through one account only.
-   The last file submitted to the leaderboard will be considered as the final submission.
-   For subtask 2A, there are 5 languages (Arabic, Bulgarian, English, German and Italian). Moreover, we define a multi-lingual evaluation scenario where we use a balanced sub-sample of all 5 languages to define multi-lingual evaluation splits.
-   Name of the output file has to be `subtask2A_LANG.tsv` where LANG can be arabic, bulgarian, english, german, italian, multilingual.
-   Get sure to set `.tsv` as the file extension; otherwise, you will get an error on the leaderboard.
-   Examples of submission file names should be `subtask2A_arabic.tsv`, `subtask2A_bulgarian.tsv`, `subtask2A_english.tsv`, `subtask2A_german.tsv`, `subtask2A_italian.tsv`, `subtask2A_multilingual.tsv`.
-   You have to zip the tsv into a file with the same name, e.g., `subtask2A_arabic.zip`, and submit it through the codalab page.
-   If you participate in the task for more than one language, for each language you must do a different submission.
-   for each submission, it is required to submit the team name. **Your team name must EXACTLY match the one used during the CLEF registration.**
-   You are allowed to submit max 200 submissions per day for each subtask.
-	Additionally, we ask you to fill out a questionnaire (link will be provided, once the evaluation cycle started started) to provide some details on your approach as we need that information for the overview paper.
-   We will keep the leaderboard private till the end of the submission period, hence, results will not be available upon submission. All results will be available after the evaluation period. --> -->

<!-- Each participant must submit their results as a single .ZIP file. -->


<!--
The file must contain .TSV files, one file for each result.

Participants are allowed to submit up to 2 results for each language.

For each language, one will be the "main" one and considered for the final ranking,
the other will be evaluated but not considered for the final ranking.

The main file must be name as "result_LN_TYPE_TEAMNAME.csv";
where LAN are two letters that identify the language
(must be either EN, AR, NL, DE, IT, TR, or ML for multi-lingual),
TYPE are four characters that indicate the type of submission
(must be either "MAIN", "ALT1"),
TEAMNAME is the identifier name of the team.
-->


## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2025/task3/