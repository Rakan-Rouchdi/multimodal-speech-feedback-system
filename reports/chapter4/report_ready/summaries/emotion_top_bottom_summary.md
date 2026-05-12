# Emotion Top/Bottom Human-Rating Summary

This appendix summary is exploratory and supplementary. It compares predicted dominant emotions with the top and bottom five recordings by human reference score within this small evaluation set only. Emotion labels are not used in the primary MAE/Pearson/Spearman comparison and are not added to the Speech-only/Text-only/Multimodal performance comparison.

1. What emotions appeared most often in the top-confidence group?
   happy appeared most often, appearing alongside 3/5 recordings. Counts: happy=3; calm=2. Recordings: S07_T1; S07_T2; S01_T1; S04_T1; S04_T2.

2. What emotions appeared most often in the bottom-confidence group?
   calm and happy appeared most often, appearing alongside 2/5 recordings. Counts: calm=2; happy=2; sad=1. Recordings: S08_T2; S05_T2; S02_T2; S08_T1; S05_T1.

3. What emotions appeared most often in the top-engagement group?
   happy appeared most often, appearing alongside 3/5 recordings. Counts: happy=3; calm=2. Recordings: S04_T1; S04_T2; S07_T1; S07_T2; S01_T1.

4. What emotions appeared most often in the bottom-engagement group?
   happy and calm appeared most often, appearing alongside 2/5 recordings. Counts: happy=2; calm=2; sad=1. Recordings: S08_T2; S02_T2; S08_T1; S02_T1; S10_T1.

5. Does clarity show any meaningful emotion pattern, or is it more likely linguistic?
   Clarity does not show a strong emotion pattern in this small set. The top-clarity group had happy=3; calm=1; sad=1, while the bottom-clarity group had happy=2; sad=1; calm=1; neutral=1. The mixed labels suggest that clarity is more likely associated with linguistic and delivery features than with a single dominant emotion label here.

6. Is there enough evidence to use emotion as a primary scoring feature?
   No. These results are exploratory, based on 20 recordings, and show emotions appearing alongside human ratings rather than explaining them. There is not enough evidence from this appendix analysis to use emotion as a primary scoring feature or to generalise beyond this evaluation set.

Method note: top and bottom groups use the human reference 0-100 means already aligned in `appendix_emotion_predictions.csv`; ties are resolved by `participant_id` to keep each group at exactly five recordings.
