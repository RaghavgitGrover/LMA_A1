# RAG Benchmark Report: Baseline vs. Re-Ranking on The Constitution of India

## 1. Executive Retrieval Metrics (Primary Benchmark, 95% Bootstrap Confidence Intervals)

| pipeline                   |   Candidate_Recall10 | Hit_at_3             | MRR_at_10            | nDCG_at_3            |   Mean_Gold_Rank | Regression_Rate   | Fix_Rate     |
|:---------------------------|---------------------:|:---------------------|:---------------------|:---------------------|-----------------:|:------------------|:-------------|
| Baseline (Bi-Encoder)      |                0.833 | 0.604 [0.458, 0.750] | 0.524 [0.409, 0.640] | 0.457 [0.343, 0.578] |             4.19 | N/A               | N/A          |
| Re-Ranked (+Cross-Encoder) |                0.833 | 0.750 [0.625, 0.855] | 0.636 [0.524, 0.752] | 0.604 [0.489, 0.711] |             3.42 | 4.2% (2/48)       | 18.8% (9/48) |

## 2. Category-Wise Retrieval Breakdown

| category      |   n |   candidate_recall10 |   base_hit3 |   rerank_hit3 |   fixed |   degraded |   both_ok |   gold_missing |   unanswerable |
|:--------------|----:|---------------------:|------------:|--------------:|--------:|-----------:|----------:|---------------:|---------------:|
| Control       |   6 |                0.667 |       0.667 |         0.5   |       0 |          1 |         3 |              2 |              0 |
| Cross-article |   7 |                1     |       0.714 |         0.714 |       1 |          1 |         4 |              0 |              0 |
| Exception     |   7 |                0.857 |       0.571 |         0.857 |       2 |          0 |         4 |              1 |              0 |
| Keyword trap  |   7 |                0.429 |       0.286 |         0.429 |       1 |          0 |         2 |              4 |              0 |
| Negation      |   7 |                0.857 |       0.571 |         0.857 |       2 |          0 |         4 |              1 |              0 |
| Procedure     |   7 |                1     |       0.571 |         0.857 |       2 |          0 |         4 |              0 |              0 |
| Role limit    |   7 |                1     |       0.857 |         1     |       1 |          0 |         6 |              0 |              0 |
| Unanswerable  |   7 |              nan     |     nan     |       nan     |       0 |          0 |         0 |              0 |              7 |

## 3. Query Outcome Classification

| status             |   count |
|:-------------------|--------:|
| BOTH_OK            |      27 |
| FIXED_BY_RERANK    |       9 |
| GOLD_NOT_IN_TOP10  |       8 |
| UNANSWERABLE       |       7 |
| DEGRADED_BY_RERANK |       2 |
| BOTH_FAIL          |       2 |

## 4. Candidate Pool Size Sweep (k=3, 5, 10, 20: Recall vs Latency Trade-Off)

|   candidate_pool_k |   candidate_recall_at_k |   hit_at_3_after_rerank |   cross_encoder_median_ms |   cross_encoder_mean_ms |
|-------------------:|------------------------:|------------------------:|--------------------------:|------------------------:|
|                  3 |                   0.604 |                   0.604 |                      92.1 |                    94   |
|                  5 |                   0.708 |                   0.667 |                     154.9 |                   155.7 |
|                 10 |                   0.833 |                   0.75  |                     332.1 |                   333.2 |
|                 20 |                   0.854 |                   0.771 |                     659.1 |                   663.1 |

## 5. Latency Profiling (ms)

| stage                                |   median_ms |   mean_ms |
|:-------------------------------------|------------:|----------:|
| embed_query (Bi-Encoder)             |        9.88 |      9.72 |
| faiss_search (IndexFlatIP)           |        0.14 |      0.14 |
| cross_encoder_rerank (10 candidates) |      324.97 |    322.03 |
| total_baseline_retrieval             |       10.02 |      9.85 |
| total_rerank_retrieval               |      333.98 |    331.88 |
| llm_generation_fresh_answer1         |    13160    |  14654.9  |
| llm_generation_fresh_answer2         |    12740    |  13454.3  |

## 6. Empirical Scaling vs Theoretical Projection

|   n_chunks |   faiss_median_ms |   faiss_std_ms |   cross_encoder_median_ms |   cross_encoder_std_ms |
|-----------:|------------------:|---------------:|--------------------------:|-----------------------:|
|         10 |              0.01 |           0.01 |                    346.26 |                   5.88 |
|         50 |              0.01 |           0.01 |                   1647.41 |                  12.96 |
|        100 |              0.01 |           0.01 |                   3455.23 |                  23.76 |
|        250 |              0.02 |           0.01 |                   7924.89 |                  23.82 |
|        500 |              0.02 |           0.01 |                  15840.8  |                  80.78 |
|        880 |              0.05 |           0.02 |                  30359.2  |                  61.24 |

### Linear Extrapolation to Larger Corpora

|   corpus_chunks |   estimated_cross_encoder_seconds |   estimated_faiss_flat_ms | note                                                                                       |
|----------------:|----------------------------------:|--------------------------:|:-------------------------------------------------------------------------------------------|
|           10000 |                            344.99 |                      0.57 | Theoretical linear extrapolation assuming fixed batch size and single-thread CPU execution |
|          100000 |                           3449.9  |                      5.7  | Theoretical linear extrapolation assuming fixed batch size and single-thread CPU execution |
|         1000000 |                          34499    |                     56.99 | Theoretical linear extrapolation assuming fixed batch size and single-thread CPU execution |
|        10000000 |                         344990    |                    569.95 | Theoretical linear extrapolation assuming fixed batch size and single-thread CPU execution |

## 7. Genuine Failure Cases: Candidate Rank Shifts & Answer Impact


### Case Study: K5 (Keyword trap) - 'Can the police force of one State investigate or exercise jurisdiction in another State without permission?'
- **Expected Legal Answer**: Under the Seventh Schedule (Union List, Entry 80), powers and jurisdiction of a State police force cannot be extended to another State so as to enable police to exercise powers without the consent of the Government of that State.
- **Status**: `FIXED_BY_RERANK` | Gold Rank Shift: `#4` → `#1`

|   Base_Rank |   CE_Rank |   Shift |   Base_Score |   CE_Score | Is_Gold   | Article   |   Page | Text_Snippet                                                                                            |
|------------:|----------:|--------:|-------------:|-----------:|:----------|:----------|-------:|:--------------------------------------------------------------------------------------------------------|
|           1 |         2 |      -1 |       0.4455 |    -1.3391 | no        | 262       |    182 | of any dispute or complaint with respect to the use, distribution or control of the waters of, or in... |
|           2 |         4 |      -2 |       0.4445 |    -4.9825 | no        | 259       |    181 | or unconditionally to that Government or to its officers functions in relation to any matter to whic... |
|           3 |         3 |       0 |       0.4394 |    -4.1862 | no        | 131       |     92 | shall, to the exclusion of any other court, have original jurisdiction in any dispute— (a) between t... |
|           4 |         1 |       3 |       0.417  |     4.3196 | YES       | 21        |    346 | the High Courts. 79. Extension of the jurisdiction of a High Court to, and exclusion of the jurisdic... |
|           5 |         7 |      -2 |       0.4087 |    -6.9768 | no        | 21        |    348 | in any court. 97. Any other matter not enumerated in List II or List III including any tax not menti... |
|           6 |         9 |      -3 |       0.3998 |    -8.7061 | no        | 14        |    319 | officers of such courts, and may also appoint such officers as may be necessary for the administrati... |
|           7 |         8 |      -1 |       0.3804 |    -7.9903 | no        | 304       |    202 | or produced in that State are subject, so, however, as not to discriminate between goods so imported... |
|           8 |         5 |       3 |       0.3791 |    -5.0253 | no        | 241       |    149 | as they apply in relation to a High Court referred to in article 214 subject to such modifications o... |
|           9 |         6 |       3 |       0.3774 |    -5.3538 | no        | 212       |    128 | place in the Legislature of a State with respect to the conduct of any Judge of the Supreme Court or... |
|          10 |        10 |       0 |       0.3735 |    -9.4062 | no        | 3B        |    317 | and, if the law is again passed by the said Council with or without amendments it shall be presented... |

**LLM Answer 1 (Baseline Top-3 Context):**
> The provided context does not contain the answer.

**LLM Answer 2 (Re-Ranked Top-3 Context):**
> According to Article 80 of the Constitution, the extension of the powers and jurisdiction of members of a police force belonging to any State to any area outside that State, but not so as to enable the police of one State to exercise powers and jurisdiction in any area outside that State without the consent of the Government of the State in which such area is situated.

**Answer Analysis**: Baseline Completeness: 0.0, Re-Ranked Completeness: 1.0, Correctness Shift: 0.0 → 1.0


### Case Study: E2 (Exception) - 'Despite equality of opportunity in public employment, can the State reserve posts for some groups?'
- **Expected Legal Answer**: Under Article 16(4), nothing shall prevent the State from making any provision for the reservation of appointments or posts in favour of any backward class of citizens not adequately represented in State services.
- **Status**: `FIXED_BY_RERANK` | Gold Rank Shift: `#4` → `#3`

|   Base_Rank |   CE_Rank |   Shift |   Base_Score |   CE_Score | Is_Gold   | Article   |   Page | Text_Snippet                                                                                            |
|------------:|----------:|--------:|-------------:|-----------:|:----------|:----------|-------:|:--------------------------------------------------------------------------------------------------------|
|           1 |         1 |       0 |       0.5702 |     0.4452 | no        | 15        |     39 | case of reservation would be in addition to the existing reservations and subject to a maximum of te... |
|           2 |         2 |       0 |       0.5318 |    -3.2571 | no        | 16        |     39 | posts in favour of any backward class of citizens which, in the opinion of the State, is not adequat... |
|           3 |         7 |      -4 |       0.4597 |    -9.5057 | no        | 15        |     38 | institutions, whether aided or unaided by the State, other than the minority educational institution... |
|           4 |         3 |       1 |       0.4366 |    -5.4503 | YES       | 16        |     39 | place of birth, residence or any of them, be ineligible for, or discriminated against in respect of,... |
|           5 |         6 |      -1 |       0.4314 |    -8.1488 | no        | 15        |     37 | before law.—The State shall not deny to any person equality before the law or the equal protection o... |
|           6 |         5 |       1 |       0.4297 |    -7.4721 | no        | 38        |     52 | as effectively as it may a social order in which justice, social, economic and political, shall info... |
|           7 |         4 |       3 |       0.418  |    -6.9529 | no        | 39        |     52 | in the concentration of wealth and means of production to the common detriment; (d) that there is eq... |
|           8 |        10 |      -2 |       0.4179 |   -11.0727 | no        | 361B      |    251 | has the meaning assigned to it in clause (a) of paragraph 1 of the Tenth Schedule; (b) the expressio... |
|           9 |         8 |       1 |       0.4111 |   -10.0484 | no        | 15        |     38 | to the use of the general public. (3) Nothing in this article shall prevent the State from making an... |
|          10 |         9 |       1 |       0.41   |   -10.4449 | no        | 16        |     39 | of the Scheduled Castes and the Scheduled Tribes which, in the opinion of the State, are not adequat... |

**LLM Answer 1 (Baseline Top-3 Context):**
> Yes, the State can reserve posts for some groups, subject to a maximum of ten per cent. of the total seats in each category, as per Article 16(1) and Article 16(4A). The State can make provisions for reservation in favour of backward classes of citizens, Scheduled Castes, and Scheduled Tribes, which are not adequately represented in the services under the State.

**LLM Answer 2 (Re-Ranked Top-3 Context):**
> Yes, the State can reserve posts for some groups, subject to a maximum of ten per cent. of the total seats in each category, as per Article 15(4) and Article 16(4A).

**Answer Analysis**: Baseline Completeness: 1.0, Re-Ranked Completeness: 0.5, Correctness Shift: 1.0 → 0.6


### Case Study: E5 (Exception) - 'Under what circumstances can Parliament legislate on a subject enumerated in the State List in the national interest?'
- **Expected Legal Answer**: Under Article 249(1), Parliament can legislate on a State List subject if the Council of States declares by resolution supported by not less than two-thirds of the members present and voting that it is necessary or expedient in the national interest.
- **Status**: `FIXED_BY_RERANK` | Gold Rank Shift: `#10` → `#1`

|   Base_Rank |   CE_Rank |   Shift |   Base_Score |   CE_Score | Is_Gold   | Article   |   Page | Text_Snippet                                                                                            |
|------------:|----------:|--------:|-------------:|-----------:|:----------|:----------|-------:|:--------------------------------------------------------------------------------------------------------|
|           1 |         4 |      -3 |       0.6875 |     1.3679 | no        | 246       |    176 | or any part of the State. (2) No law made by Parliament shall be deemed to be invalid on the ground ... |
|           2 |         2 |       0 |       0.6588 |     2.1244 | no        | 246       |    176 | part thereof with respect to any of the matters enumerated in List II in the Seventh Schedule (in th... |
|           3 |         3 |       0 |       0.6384 |     1.763  | no        | 239AA     |    145 | deemed to be a reference to Parliament. 1. Arts. 239AA and 239 AB ins. by the Constitution (Sixty-ni... |
|           4 |         5 |      -1 |       0.6031 |    -0.7376 | no        | 246A      |    177 | with respect to goods and services tax where the supply of goods, or of services, or both takes plac... |
|           5 |         8 |      -3 |       0.6026 |    -5.2827 | no        | 262       |    182 | of any dispute or complaint with respect to the use, distribution or control of the waters of, or in... |
|           6 |         9 |      -3 |       0.5872 |    -5.811  | no        | 244A      |    174 | or in part) all or any of the tribal areas specified in Part I of the table appended to paragraph 20... |
|           7 |        10 |      -3 |       0.5861 |    -7.9274 | no        | 3         |     34 | I.—In this article, in clauses (a) to (e), “State” includes a Union territory, but in the proviso, “... |
|           8 |         7 |       1 |       0.5858 |    -4.9181 | no        | 21        |    350 | of List III. 27. Production, supply and distribution of goods subject to the provisions of entry 33 ... |
|           9 |         6 |       3 |       0.5805 |    -2.4611 | no        | 252       |    178 | of the State shall to the extent of the repugnancy, but so long only as the law made by Parliament c... |
|          10 |         1 |       9 |       0.5794 |     5.9311 | YES       | 249       |    177 | in the Concurrent List or State List. (2) Such power shall include the power of making any law impos... |

**LLM Answer 1 (Baseline Top-3 Context):**
> According to Article 246(4), Parliament has power to make laws with respect to any matter for any part of the territory of India not included in a State, notwithstanding that such matter is a matter enumerated in the State List.

**LLM Answer 2 (Re-Ranked Top-3 Context):**
> According to Article 249(1), Parliament can legislate on a subject enumerated in the State List in the national interest if the Council of States has declared by resolution supported by not less than two-thirds of the members present and voting that it is necessary or expedient in the national interest that Parliament should make laws with respect to that matter.

**Answer Analysis**: Baseline Completeness: 0.0, Re-Ranked Completeness: 1.0, Correctness Shift: 0.2 → 1.0


## 8. Documented Regression Cases (Degraded by Re-Ranking)

- **C4** (Control): 'What is the ordinary duration of the House of the People, and under what condition and for how long can it be extended?' (Gold rank dropped from #1 to #5)
- **X5** (Cross-article): 'How does the retirement age of a Supreme Court judge compare to that of a High Court judge?' (Gold rank dropped from #3 to #4)